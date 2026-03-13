"""
scripts/build_rag.py — Construye el índice vectorial ChromaDB (knowledge base)
===============================================================================
Indexa SOLO contenido de conocimiento para que el agente pueda responder
preguntas sobre propiedades de minerales, políticas y categorías.

Fuentes indexadas:
  • Descripciones largas de productos (propiedades, usos, significado)
  • Descripciones de categorías
  • Páginas CMS (envíos, devoluciones, políticas, info general)

NO se indexan datos de catálogo (nombre, precio, stock) porque la búsqueda
de productos ahora se hace por SQL con filtros sobre ps_ai_product_attr.

Uso:
    python -m scripts.build_rag          # construir si no existe
    python -m scripts.build_rag --force  # reconstruir desde cero
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import text

# ── Configuración ──────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(PROJECT_DIR, "rag_data")
COLLECTION_NAME = "uniart_knowledge"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

ID_LANG_ES = 3

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Splitter profesional: separa por párrafos → frases → palabras → chars
_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    length_function=len,
)


def _strip_html(html: str | None) -> str:
    """Elimina etiquetas HTML, entidades y espacios redundantes."""
    if not html:
        return ""
    clean = re.sub(r"<[^>]+>", " ", html)
    clean = re.sub(r"&[a-z]+;", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def _chunk_text(text_str: str) -> list[str]:
    """Divide texto en chunks usando RecursiveCharacterTextSplitter de LangChain."""
    text_str = text_str.strip()
    if not text_str:
        return []
    return _splitter.split_text(text_str)


def build_index(force: bool = False) -> None:
    """Construye el índice ChromaDB con textos de conocimiento."""
    sys.path.insert(0, PROJECT_DIR)
    from backend.app.db.database import SessionLocal

    # ── Comprobar si ya existe ──
    if os.path.exists(CHROMA_DIR) and not force:
        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            col = client.get_collection(COLLECTION_NAME)
            n = col.count()
            if n > 0:
                print(f"✅ Índice ya existe con {n} documentos. "
                      f"Usa --force para reconstruir.")
                return
        except Exception:
            pass

    print("🔨 Construyendo índice de conocimiento en ChromaDB…")
    t0 = time.time()

    # ── Limpiar colección previa ──
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_fn,
    )

    db = SessionLocal()
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict] = []

    # ── 1. DESCRIPCIONES LARGAS DE PRODUCTOS (conocimiento mineral) ──
    print("  📦 Cargando descripciones de productos…")
    products = db.execute(text("""
        SELECT pl.id_product, pl.name, pl.description
        FROM ps_product_lang pl
        JOIN ps_product p ON pl.id_product = p.id_product
        WHERE pl.id_lang = :lang AND pl.id_shop = 1
              AND p.active = 1
    """), {"lang": ID_LANG_ES}).fetchall()

    n_prod_chunks = 0
    n_prods_with_desc = 0
    for row in products:
        pid, name, desc_long = row
        desc_l = _strip_html(desc_long)
        if not desc_l or len(desc_l) < 40:
            continue
        n_prods_with_desc += 1

        # Prefijo con nombre del producto para contexto
        prefixed = f"{name}: {desc_l}"
        for i, chunk in enumerate(_chunk_text(prefixed)):
            all_ids.append(f"prod_d_{pid}_{i}")
            all_docs.append(chunk)
            all_meta.append({
                "source": "product_knowledge",
                "id_product": pid,
                "name": name,
            })
            n_prod_chunks += 1

    print(f"    → {n_prod_chunks} chunks de {n_prods_with_desc} productos "
          f"con descripción (de {len(products)} activos)")

    # ── 2. CATEGORÍAS ──────────────────────────────────────────────
    print("  📂 Cargando categorías…")
    categories = db.execute(text("""
        SELECT cl.id_category, cl.name, cl.description
        FROM ps_category_lang cl
        JOIN ps_category c ON cl.id_category = c.id_category
        WHERE cl.id_lang = :lang AND c.active = 1 AND c.level_depth >= 2
    """), {"lang": ID_LANG_ES}).fetchall()

    n_cat_chunks = 0
    for row in categories:
        cid, name, description = row
        desc_clean = _strip_html(description)
        full = f"Categoría: {name}"
        if desc_clean:
            full += f" | {desc_clean}"

        for i, chunk in enumerate(_chunk_text(full)):
            all_ids.append(f"cat_{cid}_{i}")
            all_docs.append(chunk)
            all_meta.append({
                "source": "category",
                "id_category": cid,
                "name": name,
            })
            n_cat_chunks += 1

    print(f"    → {n_cat_chunks} chunks de {len(categories)} categorías")

    # ── 3. PÁGINAS CMS ────────────────────────────────────────────
    print("  📄 Cargando páginas CMS…")
    cms_pages = db.execute(text("""
        SELECT cl.id_cms, cl.meta_title, cl.content
        FROM ps_cms_lang cl
        JOIN ps_cms c ON cl.id_cms = c.id_cms
        WHERE cl.id_lang = :lang AND c.active = 1
    """), {"lang": ID_LANG_ES}).fetchall()

    n_cms_chunks = 0
    for row in cms_pages:
        cms_id, title, content = row
        if not content:
            continue
        title = title or f"CMS #{cms_id}"
        full = f"{title}: {_strip_html(content)}"

        for i, chunk in enumerate(_chunk_text(full)):
            all_ids.append(f"cms_{cms_id}_{i}")
            all_docs.append(chunk)
            all_meta.append({
                "source": "cms",
                "id_cms": cms_id,
                "title": title,
            })
            n_cms_chunks += 1

    print(f"    → {n_cms_chunks} chunks de {len(cms_pages)} páginas CMS")

    db.close()

    # ── Insertar en ChromaDB ──────────────────────────────────────
    total = len(all_ids)
    print(f"\n  💾 Indexando {total} documentos en ChromaDB…")

    BATCH = 500
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        collection.add(
            ids=all_ids[start:end],
            documents=all_docs[start:end],
            metadatas=all_meta[start:end],
        )
        print(f"    batch {start}–{end} insertado")

    elapsed = time.time() - t0
    print(f"\n✅ Índice de conocimiento construido: {collection.count()} documentos "
          f"en {elapsed:.1f}s")
    print(f"   Descripciones producto: {n_prod_chunks} | Categorías: {n_cat_chunks} "
          f"| CMS: {n_cms_chunks}")
    print(f"   Ubicación: {CHROMA_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construir índice RAG")
    parser.add_argument("--force", action="store_true",
                        help="Reconstruir aunque ya exista")
    args = parser.parse_args()
    build_index(force=args.force)
