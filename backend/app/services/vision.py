"""
backend/vision.py — Análisis de imágenes de productos con Gemini Flash Lite
=============================================================================
Envía una imagen a google/gemini-2.5-flash-lite (vía OpenRouter) y extrae
atributos del producto en JSON estructurado.

Función pública:  analyze_product_image(image_b64, mime_type) → dict
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("uvicorn.error")

_VISION_MODEL = os.getenv("VISION_MODEL", "google/gemini-2.5-flash-lite")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Tipos y piedras válidas del catálogo ──────────────────────────

VALID_TYPES = [
    "abalorio", "amuleto", "ángel", "anillo", "árbol", "cabujón", "cadena",
    "canto rodado", "chip", "colgante", "collar", "corazón", "cordón",
    "cuenta suelta", "decoración", "disco", "drusa", "embalaje", "esfera",
    "expositor", "figura", "fornitura", "geoda", "huevo", "japa mala",
    "llavero", "masajeador", "merkaba", "mineral en bruto", "obelisco",
    "orgonita", "ornamento", "otro", "pack", "palm stone", "pendientes",
    "péndulo", "pirámide", "placa", "pulsera", "punta", "reloj", "set",
    "tira de cuentas", "varita", "worry stone",
]

VALID_STONES = [
    "ágata", "aguamarina", "amatista", "amazonita", "ámbar", "angelita",
    "apatita", "apofilita", "auralita", "aventurina", "azurita", "broncita",
    "calcedonia", "calcita", "carneola", "charoita", "cianita", "coral",
    "crisocola", "cuarzo ahumado", "cuarzo citrino", "cuarzo cristal",
    "cuarzo rosa", "cuarzo verde", "esmeralda", "estilbita", "fluorita",
    "fósil", "granate", "heliotropo", "hematite", "howlita", "iolita",
    "jade", "jaspe", "labradorita", "lapislázuli", "larimar", "larvikita",
    "lepidolita", "madera", "malaquita", "metal", "morganita", "obsidiana",
    "ojo de buey", "ojo de gato", "ojo de halcón", "ojo de tigre",
    "ojo turco", "ónix", "opalite", "orgonita", "otro", "piedra bian",
    "piedra luna", "piedra maifan", "piedra natural", "piedra oro",
    "piedra sol", "piedra volcánica", "pirita", "plata", "prehnita",
    "rodocrosita", "rodonita", "rubí", "selenita", "septaria", "serpentina",
    "sílex", "sodalita", "super seven", "turmalina negra", "turmalina rosa",
    "turmalina verde", "turquesa", "unakita", "variado",
]

# ── Categorías disponibles (id → nombre) ─────────────────────────

CATEGORIES = {
    67: "Abalorios", 39: "Ágata", 40: "Amatista", 27: "Amuletos",
    7: "Anillos", 71: "Anillos de plata", 34: "Aventurina", 5: "Bisutería",
    18: "Cadenas de plata", 36: "Cantos Rodados", 22: "Carneola",
    65: "Chips", 13: "Colgantes", 68: "Colgantes de plata", 11: "Collares",
    41: "Corazones", 37: "Cristal de roca", 42: "Cuarzo",
    23: "Cuarzo rosa", 21: "Cuentas de Ágata", 32: "Cuentas de Amatista",
    63: "Cuentas de Turmalina", 24: "Cuentas sueltas",
    9: "Embalajes y complementos", 59: "Esferas y Huevos",
    44: "Figuras y Ángeles", 45: "Formas Geométricas",
    55: "Fornituras y abalórios", 38: "Geodas y drusas", 46: "Horóscopo",
    25: "Jade", 28: "Japa-Mala", 26: "Jaspe", 6: "Joyas de Plata",
    57: "Lapislázuli", 64: "Llaveros y Relojes", 35: "Masajeadores",
    66: "Minerales en Bruto", 58: "Obsidiana", 61: "Ofertas y packs",
    54: "Ojo de tigre", 10: "Ojo Turco", 17: "Orgonitas",
    62: "Otras piedras y abalorios", 15: "Pendientes de plata",
    14: "Péndulos", 29: "Piedras de los Chakras",
    16: "Piedras para engarzar", 20: "Piedras Semipreciosas",
    47: "Pirámides", 8: "Productos esotéricos",
    12: "Pulseras Artesanales", 70: "Pulseras de plata",
    48: "Puntas y obeliscos", 50: "Símbolos Reiki", 56: "Sodalita",
    72: "Tienda de minerales online", 69: "Tienda fisica",
    43: "Turmalina", 33: "Unakita", 31: "Varitas de Poder",
}

# Invertido para búsqueda por nombre
_CAT_BY_NAME = {v.lower(): k for k, v in CATEGORIES.items()}


def _build_vision_prompt() -> str:
    """Construye el prompt del modelo de visión."""
    types_str = ", ".join(VALID_TYPES)
    stones_str = ", ".join(VALID_STONES)
    cats_str = ", ".join(f'"{v}" (id:{k})' for k, v in sorted(CATEGORIES.items(), key=lambda x: x[1]))

    return f"""\
Eres un experto catalogador de minerales, cristales y bisutería para la tienda UniArt Minerales.

Analiza la imagen del producto y devuelve SOLO un JSON válido con estos campos:

{{
  "name": "Nombre corto y descriptivo del producto en español (ej: Colgante de Amatista)",
  "description_short": "Descripción comercial breve de 2-3 frases en español. Describe el producto, su mineral/piedra, forma y posible uso.",
  "type": "tipo del producto — DEBE ser EXACTAMENTE uno de: [{types_str}]",
  "stone": "mineral o piedra — DEBE ser EXACTAMENTE uno de: [{stones_str}]",
  "suggested_category_id": id numérico (int) de la categoría más apropiada,
  "suggested_category_name": "nombre de la categoría",
  "materials": "materiales principales visibles (ej: amatista natural, engaste de plata 925)",
  "shape": "forma del objeto (ej: ovalado, redondo, en bruto, facetado, cabujón)",
  "color": "colores predominantes (ej: violeta intenso, blanco lechoso)",
  "estimated_weight_grams": peso estimado en gramos (número, puede ser null si no es estimable),
  "confidence": número 0.0-1.0 indicando tu confianza en la identificación
}}

Categorías disponibles: [{cats_str}]

REGLAS:
- Los campos "type" y "stone" DEBEN coincidir EXACTAMENTE con una de las opciones listadas.
- Si no puedes identificar la piedra, usa "otro".
- Si no puedes identificar la forma/tipo, usa "otro".
- La descripción debe ser comercial y atractiva, en español.
- Responde SOLO con el JSON, sin markdown ni explicaciones extra."""


def analyze_product_image(
    image_b64: str,
    mime_type: str = "image/jpeg",
) -> dict:
    """Analiza una imagen de producto con Gemini Flash Lite.

    Args:
        image_b64: Imagen codificada en base64.
        mime_type: Tipo MIME de la imagen (image/jpeg, image/png, image/webp).

    Returns:
        dict con los atributos extraídos del producto.

    Raises:
        RuntimeError: Si la API falla o la respuesta no es JSON válido.
    """
    if not _API_KEY:
        raise RuntimeError("OPENAI_API_KEY no configurada — necesaria para visión")

    prompt = _build_vision_prompt()

    payload = {
        "model": _VISION_MODEL,
        "max_tokens": 1024,
        "temperature": 0.2,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
    }

    headers = {
        "Authorization": f"Bearer {_API_KEY}",
        "Content-Type": "application/json",
    }

    # Usar la base URL de OpenRouter
    url = f"{_BASE_URL.rstrip('/')}/chat/completions"

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Limpiar posibles markers de markdown
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        result = json.loads(content)

        # Validar y normalizar type/stone
        if result.get("type", "").lower() not in [t.lower() for t in VALID_TYPES]:
            logger.warning("Vision: tipo '%s' no válido, usando 'otro'", result.get("type"))
            result["type"] = "otro"

        if result.get("stone", "").lower() not in [s.lower() for s in VALID_STONES]:
            logger.warning("Vision: piedra '%s' no válida, usando 'otro'", result.get("stone"))
            result["stone"] = "otro"

        # Normalizar a minúsculas
        result["type"] = result["type"].lower()
        result["stone"] = result["stone"].lower()

        logger.info(
            "Vision OK: %s — %s / %s (conf=%.2f)",
            result.get("name", "?"),
            result.get("type", "?"),
            result.get("stone", "?"),
            result.get("confidence", 0),
        )

        return result

    except httpx.HTTPStatusError as exc:
        logger.error("Vision API HTTP %d: %s", exc.response.status_code, exc.response.text[:500])
        raise RuntimeError(f"Error de la API de visión: HTTP {exc.response.status_code}") from exc
    except json.JSONDecodeError as exc:
        logger.error("Vision: respuesta no es JSON válido: %s", content[:300])
        raise RuntimeError("La respuesta del modelo de visión no es JSON válido") from exc
    except Exception as exc:
        logger.error("Vision error: %s", exc)
        raise RuntimeError(f"Error al analizar la imagen: {exc}") from exc
