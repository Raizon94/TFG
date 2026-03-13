#!/usr/bin/env python3
"""
scripts/extract_attributes.py
==============================
Extrae atributos (piedra, tipo) de TODOS los productos activos usando regex
sobre el nombre del producto y los almacena en ``ps_ai_product_attr``.

Sin dependencias externas (ni LLM ni API).  Ejecución < 1 segundo.

Uso::

    python -m scripts.extract_attributes            # solo los que faltan
    python -m scripts.extract_attributes --force     # reprocesa todo
    python -m scripts.extract_attributes --dry-run   # muestra sin guardar
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from backend.app.db.database import SessionLocal


# =====================================================================
# Normalización de texto
# =====================================================================

def _strip_accents(s: str) -> str:
    """Elimina tildes/diacríticos para matching flexible."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _norm(s: str) -> str:
    """Minúscula + sin tildes."""
    return _strip_accents(s.lower())


# =====================================================================
# Taxonomías  (las más largas primero para greedy match)
# =====================================================================

_STONE_PATTERNS: list[tuple[str, str]] = [
    # -- Multi-palabra (más específico primero) --
    (r"super\s*seven",          "super seven"),
    (r"cuarzo\s+rosa",          "cuarzo rosa"),
    (r"cuarzo\s+cristal",       "cuarzo cristal"),
    (r"cuarzo\s+ahumado",       "cuarzo ahumado"),
    (r"cuarzo\s+citrin[oa]",    "cuarzo citrino"),
    (r"cuarzo\s+verde",         "cuarzo verde"),
    (r"cuarzo\s+azul",          "cuarzo cristal"),
    (r"cuarzo\s+blanco",        "cuarzo cristal"),
    (r"cuarzo\s+rutilado",      "cuarzo cristal"),
    (r"cuarzo\s+fresa",         "cuarzo rosa"),
    (r"turmalina\s+negra",      "turmalina negra"),
    (r"turmalina\s+rosa",       "turmalina rosa"),
    (r"turmalina\s+verde",      "turmalina verde"),
    (r"turmalina\s+sandia",     "turmalina sandía"),
    (r"piedra\s+luna",          "piedra luna"),
    (r"piedra\s+sol",           "piedra sol"),
    (r"piedra\s+volcanica",     "piedra volcánica"),
    (r"piedra\s+bian",          "piedra bian"),
    (r"piedra\s+oro",           "piedra oro"),
    (r"ojo\s+de\s+tigre",      "ojo de tigre"),
    (r"ojo\s+tigre",           "ojo de tigre"),
    (r"ojo\s+de\s+buey",       "ojo de buey"),
    (r"ojo\s+de\s+halcon",     "ojo de halcón"),
    (r"ojo\s+halcon",          "ojo de halcón"),
    (r"auralita\s*23?",        "auralita"),
    # -- Mono-palabra --
    (r"amatista",               "amatista"),
    (r"agata",                  "ágata"),
    (r"jaspe",                  "jaspe"),
    (r"jade(?:ita)?",           "jade"),
    (r"obsidiana",              "obsidiana"),
    (r"turmalina",              "turmalina negra"),
    (r"labradorita",            "labradorita"),
    (r"malaquita",              "malaquita"),
    (r"lapislazuli",            "lapislázuli"),
    (r"lapis\s*lazuli",         "lapislázuli"),
    (r"sodalita",               "sodalita"),
    (r"amazonita",              "amazonita"),
    (r"aventurina",             "aventurina"),
    (r"carneola",               "carneola"),
    (r"cornelina",              "carneola"),
    (r"onix",                   "ónix"),
    (r"hematite",               "hematite"),
    (r"hematita",               "hematite"),
    (r"pirita",                 "pirita"),
    (r"fluorita",               "fluorita"),
    (r"selenita",               "selenita"),
    (r"granate",                "granate"),
    (r"esmeralda",              "esmeralda"),
    (r"aguamarina",             "aguamarina"),
    (r"rodonita",               "rodonita"),
    (r"rodocrosita",            "rodocrosita"),
    (r"unakita",                "unakita"),
    (r"calcita",                "calcita"),
    (r"opalita",                "opalita"),
    (r"opalo",                  "opalita"),
    (r"howlita",                "howlita"),
    (r"turquesa",               "turquesa"),
    (r"kunzita",                "kunzita"),
    (r"moldavita",              "moldavita"),
    (r"charoita",               "charoita"),
    (r"larimar",                "larimar"),
    (r"tanzanita",              "tanzanita"),
    (r"xilopalo",               "xilópalo"),
    (r"madera\s+fosil",         "xilópalo"),
    (r"coral\b",                "coral"),
    (r"ambar",                  "ámbar"),
    (r"cianita",                "cianita"),
    (r"kyanita",                "cianita"),
    (r"lepidolita",             "lepidolita"),
    (r"prehnita",               "prehnita"),
    (r"morganita",              "morganita"),
    (r"apatit[ao]",             "apatita"),
    (r"shungita",               "shungita"),
    (r"schungita",              "shungita"),
    (r"angelita",               "angelita"),
    (r"azurita",                "azurita"),
    (r"crisocola",              "crisocola"),
    (r"auralita",               "auralita"),
    (r"celestita",              "angelita"),
    (r"apofilita",              "apofilita"),
    (r"estilbita",              "estilbita"),
    (r"rubi",                   "rubí"),
    (r"zoisita",                "zoisita"),
    (r"serpentina",             "serpentina"),
    (r"aragonito",              "aragonito"),
    (r"topacio",                "topacio"),
    (r"dumortierita",           "dumortierita"),
    (r"iolita",                 "iolita"),
    (r"citrino",                "cuarzo citrino"),
    (r"maifan",                 "piedra maifan"),
    (r"cristal\s+de\s+roca",    "cuarzo cristal"),
    (r"cuarzo",                 "cuarzo cristal"),
    (r"berilo",                 "esmeralda"),
    # -- Minerales extra --
    (r"lava",                   "piedra volcánica"),
    (r"broncita",               "broncita"),
    (r"larvikita",              "larvikita"),
    (r"calcedonia",             "calcedonia"),
    (r"turquenita",             "howlita"),
    (r"heliotropo",             "heliotropo"),
    (r"sangre\s+de\s+dragon",   "heliotropo"),
    (r"cornalina",              "carneola"),
    (r"septaria",               "septaria"),
    (r"silex",                  "sílex"),
    (r"ojo\s+de\s+gato",        "ojo de gato"),
    (r"ammonite",               "fósil"),
    (r"orthocera",              "fósil"),
    (r"marmol\s+fosil",         "fósil"),
    (r"fosil",                  "fósil"),
    (r"lapis\s+fenix",          "lapislázuli"),
    (r"rondonita",              "rodonita"),
    (r"piedra\s+caliza",        "fósil"),
    (r"pegmatitica",            "cuarzo cristal"),
    (r"rudraksha",              "madera"),
    # -- No-mineral (baja prioridad, al final) --
    (r"ojo\s+turco",            "ojo turco"),
    (r"orgonita",               "orgonita"),
    (r"madera",                 "madera"),
    (r"metal",                  "metal"),
    # -- Catch-all genéricos (última prioridad) --
    (r"piedra[s]?\s+natural",   "piedra natural"),
    (r"piedra[s]?\s+semipreciosa", "piedra natural"),
    (r"mineral(?:es)?",         "piedra natural"),
    (r"gema[s]?",               "piedra natural"),
]

_VARIADO_PATTERNS = [
    r"chakras?",
    r"variados?",
    r"7\s*piedras",
    r"siete\s*piedras",
    r"colores\s+de\s+los",
    r"piedras\s+y\s+signos",
    r"minerales\s+variados",
    r"mezcla\s+de\s+mineral",
    r"multi\s*mineral",
]

_STONE_EXCLUDE: list[str] = [
    # ya no excluimos nada, ojo turco y plata se clasifican
]

_TYPE_PATTERNS: list[tuple[str, str]] = [
    # -- Multi-palabra --
    (r"japa\s*mala",             "japa mala"),
    (r"mala\s+de\s+bola",       "japa mala"),
    (r"mineral\s+en\s+bruto",   "mineral en bruto"),
    (r"palm\s*stone",           "palm stone"),
    (r"worry\s*stone",          "worry stone"),
    (r"canto[s]?\s+rodado[s]?", "canto rodado"),
    (r"tira[s]?\s+de\s+cuenta", "tira de cuentas"),
    (r"hilo\s+de\b",            "tira de cuentas"),
    (r"cuenta[s]?\s+suelta",    "cuenta suelta"),
    (r"cuenta[s]?\s+de\b",      "tira de cuentas"),
    (r"piedra\s+para\s+engarzar", "piedra para engarzar"),
    # -- Mono-palabra --
    (r"anillo",                 "anillo"),
    (r"colgante[s]?",           "colgante"),
    (r"pulsera",                "pulsera"),
    (r"collar\b",               "collar"),
    (r"pendientes?",            "pendientes"),
    (r"pendulo",                "péndulo"),
    (r"esfera",                 "esfera"),
    (r"huevo",                  "huevo"),
    (r"corazon",                "corazón"),
    (r"piramide",               "pirámide"),
    (r"punta",                  "punta"),
    (r"obelisco",               "obelisco"),
    (r"drusa",                  "drusa"),
    (r"geoda",                  "geoda"),
    (r"figura",                 "figura"),
    (r"angel",                  "ángel"),
    (r"arbol",                  "árbol"),
    (r"masajeador",             "masajeador"),
    (r"varita",                 "varita"),
    (r"llavero",                "llavero"),
    (r"amuleto",                "amuleto"),
    (r"disco",                  "disco"),
    (r"merkaba",                "merkaba"),
    (r"chips?\b",               "chip"),
    (r"fornitura",              "fornitura"),
    (r"cabujon",                "cabujón"),
    (r"cadena",                 "cadena"),
    (r"orgonita",               "orgonita"),
    (r"\bset\b",                "set"),
    (r"\bpack\b",               "pack"),
    (r"expositor",              "expositor"),
    (r"ornamento",              "ornamento"),
    (r"abalorio[s]?",           "abalorio"),
    (r"dije[s]?",               "abalorio"),
    (r"charm[s]?",              "abalorio"),
    (r"colgador(?:es|a)?",      "fornitura"),
    (r"enganche[s]?",           "fornitura"),
    (r"adorno",                 "decoración"),
    (r"decoracion",             "decoración"),
    (r"ornamento",              "decoración"),
    (r"placa",                  "placa"),
    (r"cordon",                 "cordón"),
    (r"rosa\s+labrada",         "figura"),
    (r"mensajero\s+del\s+viento", "decoración"),
    (r"biterminado",            "punta"),
    (r"\ben\s+bruto\b",         "mineral en bruto"),
    (r"\bbruto\b",              "mineral en bruto"),
    (r"talla\s+\w+",            "tira de cuentas"),
    (r"bola[s]?\s+\d",          "tira de cuentas"),
    (r"bola[s]?\s+facetada",    "tira de cuentas"),
    (r"bola[s]?\s+agujereada",  "tira de cuentas"),
    (r"nugget",                 "tira de cuentas"),
    (r"rondelle",               "tira de cuentas"),
    (r"facetada[s]?",           "tira de cuentas"),
    (r"hilos?\s+de",            "tira de cuentas"),
    (r"hebras?\s+de",           "tira de cuentas"),
    (r"bola[s]?\s+de",          "tira de cuentas"),
    (r"cuentas?\s+\w+",         "tira de cuentas"),
    (r"mala\s+budista",         "japa mala"),
    (r"reloj",                  "reloj"),
    (r"incienso",               "incienso"),
    (r"cubo[s]?\s+de\s+mineral", "set"),
    (r"bolsa[s]?\s+de\s+organza", "embalaje"),
    (r"lamina[s]?",              "placa"),
    (r"chapa[s]?",              "placa"),
    (r"tiras?\s+de",            "tira de cuentas"),
    (r"alfiler",                "fornitura"),
    (r"mosqueton",              "fornitura"),
    (r"formacion",              "mineral en bruto"),
    (r"semibrut[ao]",           "mineral en bruto"),
    (r"\bde\s+pared\b",         "decoraci\u00f3n"),
    (r"mano\s+de\s+fatima",     "amuleto"),
    (r"mano\s+de\s+metal",      "decoraci\u00f3n"),
    (r"tallad[ao]",             "figura"),
    (r"briolette",             "tira de cuentas"),
    (r"ovalo[s]?",              "tira de cuentas"),
    (r"aceituna[s]?",           "tira de cuentas"),
    (r"rodado[s]?",             "canto rodado"),
    (r"pulid[ao]",              "canto rodado"),
    (r"por\s+peso",             "mineral en bruto"),
    (r"caja\s+de\s+un\s+kilo",  "mineral en bruto"),
    (r"\d+\s*gramos",           "mineral en bruto"),
]

# Fallback de material (solo si no se encontró ningún mineral)
_FALLBACK_STONE = [
    (r"plata",  "plata"),
]

# Pre-compilar todos los patrones
_STONE_RE   = [(re.compile(p, re.IGNORECASE), v) for p, v in _STONE_PATTERNS]
_FALLBACK_RE = [(re.compile(p, re.IGNORECASE), v) for p, v in _FALLBACK_STONE]
_TYPE_RE    = [(re.compile(p, re.IGNORECASE), v) for p, v in _TYPE_PATTERNS]
_VARIADO_RE = [re.compile(p, re.IGNORECASE) for p in _VARIADO_PATTERNS]
_EXCLUDE_RE = [re.compile(p, re.IGNORECASE) for p in _STONE_EXCLUDE]


# =====================================================================
# Clasificador
# =====================================================================

def classify(name: str) -> tuple[str, str]:
    """Devuelve (stone, type) para un nombre de producto."""
    normed = _norm(name)

    # -- Stone --
    stone = "otro"

    # 1) Excluir falsos positivos (ojo turco != mineral)
    excluded = any(rx.search(normed) for rx in _EXCLUDE_RE)

    if not excluded:
        # 2) Variado? (chakras, mix)
        if any(rx.search(normed) for rx in _VARIADO_RE):
            stone = "variado"
        else:
            # 3) Buscar piedra en la taxonomia
            for rx, val in _STONE_RE:
                if rx.search(normed):
                    stone = val
                    break

            # 4) Fallback: plata solo si no hay ningún mineral
            if stone == "otro":
                for rx, val in _FALLBACK_RE:
                    if rx.search(normed):
                        stone = val
                        break

    # -- Type --
    ptype = "otro"
    for rx, val in _TYPE_RE:
        if rx.search(normed):
            ptype = val
            break

    return stone, ptype


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extraer atributos de productos con regex"
    )
    parser.add_argument("--force", action="store_true",
                        help="Reprocesar todos los productos")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostrar sin guardar en BD")
    args = parser.parse_args()

    db = SessionLocal()

    # -- Crear tabla si no existe --
    print("Verificando tabla ps_ai_product_attr...")
    db.execute(text(
        "CREATE TABLE IF NOT EXISTS ps_ai_product_attr ("
        "  id_product INT UNSIGNED PRIMARY KEY,"
        "  stone VARCHAR(80) NOT NULL DEFAULT 'otro',"
        "  type  VARCHAR(80) NOT NULL DEFAULT 'otro',"
        "  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        "    ON UPDATE CURRENT_TIMESTAMP"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    ))
    db.commit()

    # -- Cargar productos --
    if args.force:
        db.execute(text("TRUNCATE TABLE ps_ai_product_attr"))
        db.commit()

    rows = db.execute(text(
        "SELECT p.id_product, pl.name "
        "FROM ps_product p "
        "JOIN ps_product_lang pl ON p.id_product = pl.id_product "
        "  AND pl.id_lang = 3 "
        "WHERE p.active = 1 "
        "  AND p.id_product NOT IN "
        "      (SELECT id_product FROM ps_ai_product_attr) "
        "ORDER BY p.id_product"
    )).fetchall()

    products = [(r[0], r[1]) for r in rows]
    total = len(products)

    if total == 0:
        print("Todos los productos ya tienen atributos extraidos.")
        db.close()
        return

    print(f"{total} productos por clasificar")

    # -- Clasificar --
    processed = 0
    errors = 0

    for pid, name in products:
        stone, ptype = classify(name)

        if args.dry_run:
            print(f"  {pid:>5} | {stone:<20} | {ptype:<20} | {name[:65]}")
        else:
            try:
                db.execute(text(
                    "INSERT INTO ps_ai_product_attr (id_product, stone, type) "
                    "VALUES (:pid, :stone, :type) "
                    "ON DUPLICATE KEY UPDATE "
                    "  stone=VALUES(stone), type=VALUES(type)"
                ), {"pid": pid, "stone": stone, "type": ptype})
                processed += 1
            except Exception as e:
                print(f"  Error guardando {pid}: {e}")
                errors += 1

    if not args.dry_run:
        db.commit()

    db.close()

    if args.dry_run:
        print(f"\nDry-run: {total} productos clasificados (nada guardado)")
    else:
        print(f"\nExtraccion completada: {processed} guardados, {errors} errores")

    # -- Estadisticas --
    _show_stats(args.dry_run, products if args.dry_run else None)


def _show_stats(dry_run: bool, products: list | None = None):
    """Muestra distribucion de atributos."""
    if dry_run and products:
        from collections import Counter
        stone_c: Counter[str] = Counter()
        type_c: Counter[str] = Counter()
        for _, name in products:
            s, t = classify(name)
            stone_c[s] += 1
            type_c[t] += 1

        print("\nTop piedras:")
        for val, cnt in stone_c.most_common(20):
            print(f"  {cnt:>5}x  {val}")
        print("\nTop tipos:")
        for val, cnt in type_c.most_common(20):
            print(f"  {cnt:>5}x  {val}")

        otros_s = stone_c.get("otro", 0)
        otros_t = type_c.get("otro", 0)
        total = sum(stone_c.values())
        pct_s = 100 * otros_s / total if total else 0
        pct_t = 100 * otros_t / total if total else 0
        print(f"\n  Stone 'otro': {otros_s}/{total} ({pct_s:.1f}%)")
        print(f"  Type  'otro': {otros_t}/{total} ({pct_t:.1f}%)")
    else:
        db2 = SessionLocal()
        stone_stats = db2.execute(text(
            "SELECT stone, COUNT(*) as cnt FROM ps_ai_product_attr "
            "GROUP BY stone ORDER BY cnt DESC LIMIT 20"
        )).fetchall()
        type_stats = db2.execute(text(
            "SELECT type, COUNT(*) as cnt FROM ps_ai_product_attr "
            "GROUP BY type ORDER BY cnt DESC LIMIT 20"
        )).fetchall()
        total_row = db2.execute(text(
            "SELECT COUNT(*) FROM ps_ai_product_attr"
        )).scalar()
        otros_s = db2.execute(text(
            "SELECT COUNT(*) FROM ps_ai_product_attr WHERE stone='otro'"
        )).scalar()
        otros_t = db2.execute(text(
            "SELECT COUNT(*) FROM ps_ai_product_attr WHERE type='otro'"
        )).scalar()
        db2.close()

        print("\nTop piedras:")
        for s in stone_stats:
            print(f"  {s[1]:>5}x  {s[0]}")
        print("\nTop tipos:")
        for t in type_stats:
            print(f"  {t[1]:>5}x  {t[0]}")

        pct_s = 100 * otros_s / total_row if total_row else 0
        pct_t = 100 * otros_t / total_row if total_row else 0
        print(f"\n  Stone 'otro': {otros_s}/{total_row} ({pct_s:.1f}%)")
        print(f"  Type  'otro': {otros_t}/{total_row} ({pct_t:.1f}%)")


if __name__ == "__main__":
    main()
