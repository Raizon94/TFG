# E-Commerce Headless gestionado por Agentes de IA

Sistema e-commerce headless donde **agentes de inteligencia artificial** gestionan tanto la experiencia del cliente como la administración de la tienda. Construido sobre datos reales de [UniArt Minerales](https://uniartminerales.com) (tienda online de minerales, cristales y gemas), combina un backend API con agentes conversacionales LangGraph, un motor de recomendaciones por filtrado colaborativo (SVD), un sistema RAG con ChromaDB, y un frontend Streamlit.

## Tabla de contenidos

- [Arquitectura general](#arquitectura-general)
- [Stack tecnológico](#stack-tecnológico)
- [Puesta en marcha](#puesta-en-marcha)
  - [Opción A: Docker Compose (recomendado)](#opción-a-docker-compose-recomendado)
  - [Opción B: Ejecución local](#opción-b-ejecución-local)
- [Variables de entorno](#variables-de-entorno)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Componentes del sistema](#componentes-del-sistema)
  - [Backend (FastAPI)](#backend-fastapi)
  - [Agente de cliente](#agente-de-cliente)
  - [Agente de administración](#agente-de-administración)
  - [Motor de recomendaciones (SVD)](#motor-de-recomendaciones-svd)
  - [Sistema RAG (ChromaDB)](#sistema-rag-chromadb)
  - [Sistema de boost](#sistema-de-boost)
  - [Guardrails de seguridad](#guardrails-de-seguridad)
  - [Servicio de visión](#servicio-de-visión)
  - [Cross-Encoder Re-ranker](#cross-encoder-re-ranker)
  - [Frontend (Streamlit)](#frontend-streamlit)
- [API REST](#api-rest)
- [Scripts](#scripts)
- [Tests y resultados](#tests-y-resultados)
- [Base de datos](#base-de-datos)

---

## Arquitectura general

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (Streamlit :8501)                    │
│                  ┌──────────────┐  ┌──────────────────┐             │
│                  │ Vista Cliente │  │  Vista Admin      │             │
│                  │  (Chat + UI)  │  │ (Chat + Paneles)  │             │
│                  └──────┬───────┘  └────────┬─────────┘             │
└─────────────────────────┼──────────────────┼────────────────────────┘
                          │ HTTP/SSE         │ HTTP/SSE
┌─────────────────────────┼──────────────────┼────────────────────────┐
│                     Backend (FastAPI :8000)                          │
│  ┌──────────────────────┼──────────────────┼─────────────────────┐  │
│  │                  API Router (/api)                             │  │
│  │    /chat    /products   /admin   /checkout   /customers        │  │
│  └──────┬──────────┬────────┬─────────┬───────────┬──────────────┘  │
│         │          │        │         │           │                  │
│  ┌──────▼──────┐   │  ┌─────▼──────┐  │    ┌──────▼──────┐         │
│  │   Agente    │   │  │   Agente   │  │    │   Motor ML  │         │
│  │  Cliente    │   │  │   Admin    │  │    │   (SVD)     │         │
│  │ (LangGraph) │   │  │ (LangGraph)│  │    │  Fold-in    │         │
│  └──┬──┬──┬────┘   │  └──┬──┬──┬───┘  │    └─────────────┘         │
│     │  │  │        │     │  │  │      │                             │
│  ┌──▼──▼──▼────────▼─────▼──▼──▼──────▼──────────────────────────┐  │
│  │                    Servicios                                   │  │
│  │  ChromaDB (RAG)  │  Re-ranker  │  Vision (Gemini)  │  Boost   │  │
│  └───────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                  Guardrails (LLM + Heurísticas)                │  │
│  └───────────────────────────────────────────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │ SQLAlchemy
                    ┌────────────▼────────────┐
                    │   MySQL (PrestaShop BD)  │
                    │        :3306             │
                    └─────────────────────────┘
```

---

## Stack tecnológico

| Capa | Tecnología |
|------|-----------|
| **Backend API** | FastAPI + Uvicorn (Python 3.13) |
| **Agentes IA** | LangGraph + LangChain + OpenAI-compatible API (OpenRouter) |
| **LLM principal** | Configurable (por defecto `gpt-4o-mini` vía OpenRouter) |
| **Modelo de visión** | Google Gemini 2.5 Flash Lite (vía OpenRouter) |
| **Recomendaciones ML** | TruncatedSVD (scikit-learn) con fold-in en tiempo real |
| **RAG** | ChromaDB + SentenceTransformers (`intfloat/multilingual-e5-large`) |
| **Re-ranking** | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| **Base de datos** | MySQL 9.6 (esquema PrestaShop 1.6 con tablas IA adicionales) |
| **Frontend** | Streamlit |
| **Contenedores** | Docker + Docker Compose |
| **Memoria agente** | SQLite (checkpointer LangGraph) |

---

## Puesta en marcha

### Requisitos previos

- **Docker Desktop** (para la opción Docker) o **Python 3.13** + **MySQL** (para local)
- Una **API key** de OpenRouter u OpenAI para los agentes IA
- ~4 GB de RAM disponible (los modelos de embeddings y re-ranking se cargan en memoria)

### Opción A: Docker Compose (recomendado)

Es la forma más sencilla. Levanta todos los servicios (MySQL, backend, frontend, PhpMyAdmin) con un solo comando.

**1. Clonar el repositorio**

```bash
git clone <url-del-repo>
cd TFG
```

**2. Configurar variables de entorno**

Edita el archivo `.env` en la raíz del proyecto con tu API key:

```env
# API Key de OpenRouter u OpenAI
OPENAI_API_KEY=sk-tu-clave-aqui
OPENAI_MODEL=openai/gpt-4o-mini
OPENAI_BASE_URL=https://openrouter.ai/api/v1

# (Opcional) Modelo de visión para creación de productos por imagen
VISION_MODEL=google/gemini-2.5-flash-lite

# (Opcional) Proxy — dejar vacío si no se usa
PROXY_URL=
```

**3. Levantar los servicios**

```bash
docker compose up --build -d
```

Esto realiza automáticamente:
- Arranca **MySQL** y carga el dump `BD.sql` en el primer inicio
- Construye e inicia el **backend** (FastAPI) en el puerto `8000`
- Construye e inicia el **frontend** (Streamlit) en el puerto `8501`
- Arranca **PhpMyAdmin** en el puerto `8080`
- El índice RAG y los modelos ML se generan automáticamente en el primer arranque si no existen

**4. Acceder**

| Servicio | URL |
|----------|-----|
| Tienda (frontend) | http://localhost:8501 |
| API docs (Swagger) | http://localhost:8000/docs |
| PhpMyAdmin | http://localhost:8080 |

**5. Parar los servicios**

```bash
docker compose down
```

Para borrar también los volúmenes (base de datos, modelos, RAG):

```bash
docker compose down -v
```

### Opción B: Ejecución local

Para desarrollo o si no se dispone de Docker.

**1. Requisitos**

- Python 3.13 (se recomienda usar `pyenv`)
- MySQL corriendo localmente
- PHP (opcional, para PhpMyAdmin local)

**2. Importar la base de datos**

```bash
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS tfg_bd"
mysql -u root -p tfg_bd < BD.sql
```

**3. Instalar dependencias Python**

```bash
pip install -r requirements.txt
```

**4. Configurar `.env`**

Asegúrate de que `DATABASE_URL` apunte a tu MySQL local:

```env
DATABASE_URL=mysql+pymysql://root:tu_password@localhost:3306/tfg_bd
BACKEND_URL=http://localhost:8000
```

**5. Entrenar los modelos ML**

```bash
python -m scripts.train_ml
```

**6. Construir el índice RAG**

```bash
python -m scripts.build_rag
```

**7. Usar el script de arranque**

```bash
chmod +x start.sh
./start.sh
```

El script `start.sh` verifica Python, MySQL, dependencias, modelos y arranca backend + frontend + PhpMyAdmin automáticamente.

Comandos del script:

| Comando | Descripción |
|---------|-------------|
| `./start.sh` | Arranca todo el stack |
| `./start.sh --train` | Arranca y re-entrena los modelos ML |
| `./start.sh --stop` | Para todos los servicios |

---

## Variables de entorno

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `OPENAI_API_KEY` | Clave API para el LLM (OpenRouter/OpenAI) | — (obligatoria) |
| `OPENAI_MODEL` | Modelo LLM a usar | `openai/gpt-4o-mini` |
| `OPENAI_BASE_URL` | Base URL de la API compatible OpenAI | `https://openrouter.ai/api/v1` |
| `OPENAI_TEMPERATURE` | Temperatura del LLM | `0.3` |
| `VISION_MODEL` | Modelo para análisis de imágenes | `google/gemini-2.5-flash-lite` |
| `DATABASE_URL` | Conexión MySQL (SQLAlchemy) | `mysql+pymysql://root:root1234@localhost:3306/tfg_bd` |
| `BACKEND_URL` | URL del backend (para el frontend) | `http://localhost:8000` |
| `EMBEDDING_MODEL` | Modelo de embeddings para RAG | `intfloat/multilingual-e5-large` |
| `RERANKER_MODEL` | Cross-encoder para re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CHUNK_SIZE` | Tamaño de chunks para RAG | `500` |
| `CHUNK_OVERLAP` | Solapamiento entre chunks | `100` |
| `PROXY_URL` | Proxy HTTP/HTTPS (opcional) | — |

---

## Estructura del proyecto

```
TFG/
├── .env                          # Variables de entorno
├── BD.sql                        # Dump de la base de datos PrestaShop
├── docker-compose.yml            # Orquestación Docker
├── Dockerfile.backend            # Imagen del backend
├── Dockerfile.frontend           # Imagen del frontend
├── requirements.txt              # Dependencias Python
├── start.sh                      # Script de arranque local
├── agent_memory.sqlite           # Memoria persistente de los agentes
│
├── backend/
│   └── app/
│       ├── main.py               # Punto de entrada FastAPI (lifespan, CORS, montaje)
│       ├── agents/
│       │   ├── customer/         # Agente de atención al cliente
│       │   │   ├── agent.py      # API pública: chat(), chat_stream()
│       │   │   ├── graph.py      # Grafo LangGraph (guardrail → agent → tools)
│       │   │   ├── prompts.py    # System prompt del asistente
│       │   │   └── tools.py      # Herramientas: search_catalog, RAG, recomendaciones...
│       │   ├── admin/            # Agente ERP de administración
│       │   │   ├── agent.py      # API pública + grafo LangGraph
│       │   │   └── tools.py      # Herramientas: ventas, pedidos, clientes, catálogo...
│       │   └── shared/
│       │       └── guardrails.py # Guardrails de seguridad (heurísticas + LLM)
│       ├── api/
│       │   ├── router.py         # Router principal con todos los sub-routers
│       │   ├── utils.py          # Utilidades (URLs de imágenes, etc.)
│       │   └── routes/
│       │       ├── admin.py      # Endpoints admin (stats, boost, retrain, chat)
│       │       ├── catalog.py    # Endpoints catálogo (productos, recomendaciones)
│       │       ├── chat.py       # Endpoints chat cliente (send, stream, history)
│       │       ├── checkout.py   # Endpoint checkout (creación de pedidos)
│       │       └── customers.py  # Endpoints clientes (top clientes)
│       ├── core/
│       │   └── config.py         # Config compartida: LLM, ChromaDB, proxy, checkpointer
│       ├── db/
│       │   └── database.py       # SQLAlchemy: engine, modelos ORM, ensure_ai_tables()
│       ├── models/               # Artefactos ML (.joblib)
│       │   ├── svd_what.joblib
│       │   ├── fallback_popular.joblib
│       │   └── training_metadata.joblib
│       ├── schemas/              # Modelos Pydantic (request/response)
│       │   ├── admin.py
│       │   ├── chat.py
│       │   ├── order.py
│       │   └── product.py
│       └── services/
│           ├── ml_engine.py      # Inferencia SVD fold-in + reentrenamiento inteligente
│           ├── reranker.py       # Cross-encoder re-ranking
│           └── vision.py         # Análisis de imágenes con Gemini
│
├── frontend/
│   ├── app.py                    # Punto de entrada Streamlit (menú lateral)
│   ├── .streamlit/config.toml    # Configuración Streamlit (tema, etc.)
│   └── views/
│       ├── customer_view.py      # Vista tienda: chat, productos, carrito
│       └── admin_view.py         # Vista admin: chat ERP, stats, boost
│
├── scripts/
│   ├── train_ml.py               # Entrena el modelo SVD de recomendaciones
│   ├── build_rag.py              # Construye el índice ChromaDB
│   ├── evaluate_agent.py         # Evaluación automatizada del agente
│   ├── evaluate_ml.py            # Evaluación del motor ML (métricas offline)
│   ├── evaluate_rag.py           # Evaluación del sistema RAG
│   ├── extract_attributes.py    # Extrae atributos (piedra, tipo) de productos
│   └── anonimizar.py            # Anonimiza datos personales para el repositorio
│
├── docker/
│   ├── mysql/my.cnf              # Configuración MySQL personalizada
│   └── phpmyadmin/servername.conf
│
├── rag_data/                     # Índice ChromaDB persistido
│   ├── chroma.sqlite3
│   └── <colecciones>/            # Vectores HNSW por colección
│
├── product_images/               # Imágenes de productos
│
├── backups/
│   └── BD_original.sql           # Backup de la BD original
│
├── evaluation_results/           # Resultados de evaluaciones
│
└── tests/                        # Suite de tests con resultados
    ├── run_01_api.py ... run_09_rendimiento.py
    └── results_01_api.json ... results_09_rendimiento.json
```

---

## Componentes del sistema

### Backend (FastAPI)

Punto de entrada en `backend/app/main.py`. Al arrancar (lifespan):

1. Crea las tablas auxiliares de IA (`ps_ai_boost`, `ps_ai_product_attr`) si no existen
2. Carga los modelos ML (SVD + fallback popular) desde `.joblib`
3. Opcionalmente precalienta la colección ChromaDB

Expone la API REST bajo `/api` y sirve imágenes de productos como archivos estáticos.

### Agente de cliente

Agente conversacional que actúa como dependiente virtual de la tienda. Implementado como un **grafo LangGraph** con 4 nodos:

```
[input_guardrail] → [agent] ⇄ [tools] → [check_escalation]
```

**Flujo:**
1. **Guardrail de entrada**: comprueba si el mensaje es seguro (heurísticas + LLM clasificador)
2. **Agente**: el LLM razona con el system prompt y decide qué herramientas usar
3. **Tools**: ejecuta las herramientas seleccionadas
4. **Check escalation**: verifica si se ha pedido derivar a un humano

**Herramientas disponibles:**

| Herramienta | Descripción |
|-------------|-------------|
| `search_catalog` | Búsqueda de productos con filtros SQL (piedra, tipo, keyword). Ranking híbrido: relevancia semántica + popularidad/boost + afinidad del cliente |
| `browse_categories` | Exploración del árbol de categorías |
| `infer_minerals_for_intent` | Dado un uso/intención ("amor", "protección"), descubre qué minerales recomienda la tienda usando RAG sobre las descripciones |
| `search_knowledge_base` | Consulta la base de conocimiento RAG (propiedades, envíos, políticas) |
| `get_recommendations` | Recomendaciones personalizadas ML (SVD fold-in) |
| `get_order_status` | Estado de los pedidos del cliente |
| `get_customer_info` | Datos del cliente registrado |
| `escalate_to_human` | Derivación a un agente humano |

**Memoria persistente**: el agente mantiene el historial conversacional entre sesiones usando SQLite como checkpointer de LangGraph.

### Agente de administración

Agente ERP para la gestión interna de la tienda. Grafo LangGraph sin guardrails (uso interno).

**Herramientas disponibles:**

| Herramienta | Descripción |
|-------------|-------------|
| `get_sales_report` | Informes de ventas por período (hoy/semana/mes/año) |
| `get_top_products` | Ranking de productos más vendidos |
| `search_orders` | Búsqueda de pedidos por cliente, referencia o estado |
| `get_order_details` | Detalle completo de un pedido (líneas, historial de estados) |
| `search_customers` | Búsqueda de clientes por nombre o email |
| `get_catalog_stats` | Estadísticas del catálogo (productos activos, precios, categorías) |
| `get_pending_orders` | Pedidos pagados pendientes de envío |
| `get_packing_list` | Lista de empaquetado con dirección de envío |
| `update_order_status` | Cambio de estado de pedidos |
| `list_order_statuses` | Estados disponibles en el sistema |
| `create_product` | Creación de productos en la BD (con soporte para análisis de imagen) |

### Motor de recomendaciones (SVD)

Sistema de **filtrado colaborativo** basado en TruncatedSVD (descomposición en valores singulares).

**Entrenamiento** (`scripts/train_ml.py`):
1. Extrae pedidos reales de la BD PrestaShop
2. Construye una matriz usuario-producto dispersa (sparse)
3. Aplica `log(1 + cantidad)` para normalizar
4. Descompone con TruncatedSVD (50 componentes latentes)
5. Guarda los artefactos `.joblib`

**Inferencia en tiempo real** (`services/ml_engine.py`):
- Técnica **fold-in con ventana deslizante** (MAX_WINDOW=8):
  1. Consulta las compras actuales del usuario desde la BD
  2. Toma solo los últimos 8 productos distintos (por fecha)
  3. Construye un vector binario y lo proyecta al espacio latente: `u = x · Vᵀ`
  4. Reconstruye scores: `s = u · V`
  5. Excluye productos de la ventana y devuelve los top 10
- **Cada nueva compra se refleja inmediatamente** sin reentrenar
- Latencia de inferencia: **~6.5 ms** (p50)

**Reentrenamiento inteligente**:
- Compara el número de pedidos actuales con los del último entrenamiento
- Si hay ≥50 pedidos nuevos, lanza reentrenamiento en un hilo de fondo
- El administrador puede forzar el reentrenamiento desde el panel

### Sistema RAG (ChromaDB)

Base de conocimiento vectorial para que los agentes respondan preguntas sobre propiedades de minerales, políticas de la tienda y categorías.

**Construcción del índice** (`scripts/build_rag.py`):
- **Fuentes indexadas**:
  - Descripciones largas de productos (propiedades, usos, significado)
  - Descripciones de categorías
  - Páginas CMS (envíos, devoluciones, políticas)
- **Chunking**: `RecursiveCharacterTextSplitter` (500 chars, 100 overlap)
- **Embeddings**: `intfloat/multilingual-e5-large` (SentenceTransformers)
- **Almacenamiento**: ChromaDB con persistencia local (directorio `rag_data/`)

**No se indexan** datos del catálogo (nombre, precio, stock) porque la búsqueda de productos se hace por SQL con filtros sobre la tabla `ps_ai_product_attr`.

### Sistema de boost

Mecanismo para que el administrador **priorice manualmente ciertos productos** en las respuestas del agente.

**Tabla `ps_ai_boost`**: almacena el `id_product`, un `boost_factor` (1.0–5.0) y un motivo opcional.

**Cómo funciona en la búsqueda**:
1. La query SQL trae el `boost_factor` de cada producto (`LEFT JOIN ps_ai_boost`)
2. Al calcular el score de popularidad: `popularidad = log(1 + ventas) × boost_factor`
3. Un producto con `boost_factor=2.0` duplica su señal de popularidad

**CRUD completo** vía API: `GET/POST/DELETE /api/admin/boost` + búsqueda de productos para asignar boost.

### Guardrails de seguridad

Sistema de protección en dos niveles para el agente de cliente:

1. **Heurísticas rápidas (sin LLM)**: detecta patrones obvios de jailbreak/prompt injection ("olvida tus instrucciones", "act as", "developer mode", etc.)
2. **Clasificador LLM**: un modelo pequeño clasifica los mensajes ambiguos como seguros o bloqueados (categorías: jailbreak, injection, off-topic, inapropiado)

**Política fail-open**: si el clasificador falla, el mensaje pasa por defecto para no bloquear usuarios legítimos.

### Servicio de visión

Análisis de imágenes de productos usando **Gemini 2.5 Flash Lite** (vía OpenRouter):

- El administrador sube una foto de un producto
- El modelo de visión extrae: nombre, tipo de producto, mineral/piedra, color, forma, materiales, descripción comercial, categoría sugerida
- Los atributos extraídos se presentan al admin para confirmación
- Tras confirmación, se invoca `create_product` para insertar en la BD

### Cross-Encoder Re-ranker

Reordena los resultados del RAG usando un modelo cross-encoder (`ms-marco-MiniLM-L-6-v2`):

- Recibe la query del usuario y los chunks candidatos de ChromaDB
- Calcula un score de relevancia par-a-par (query, documento)
- Reordena por score descendente
- Fallback: si el re-ranker no está disponible, usa el orden original de ChromaDB

### Frontend (Streamlit)

Interfaz web con dos vistas:

**Vista Cliente** (`views/customer_view.py`):
- Chat conversacional con el agente de la tienda
- Tarjetas de producto con imagen, precio y enlace
- Sistema de carrito de compras
- Identificación de cliente (opcional, para personalización)

**Vista Admin** (`views/admin_view.py`):
- Chat con el agente ERP (informes, pedidos, clientes)
- Panel de estadísticas (pedidos totales, ingresos, ticket medio)
- Panel de gestión de boosts (listar, crear, eliminar)
- Estado del modelo ML y botón de reentrenamiento
- Subida de imágenes para crear productos automáticamente

---

## API REST

Documentación interactiva disponible en `http://localhost:8000/docs` (Swagger UI).

### Endpoints principales

| Grupo | Endpoint | Método | Descripción |
|-------|----------|--------|-------------|
| **Chat** | `/api/chat/send` | POST | Envía mensaje al agente cliente |
| | `/api/chat/stream` | POST | Chat con streaming SSE |
| | `/api/chat/history/{thread_id}` | GET | Historial de conversación |
| **Catálogo** | `/api/products` | GET | Lista de productos destacados |
| | `/api/recommendations/{id_customer}` | GET | Recomendaciones personalizadas |
| **Admin** | `/api/admin/stats` | GET | Estadísticas generales |
| | `/api/admin/chat` | POST | Chat con agente ERP |
| | `/api/admin/chat/stream` | POST | Chat admin con streaming |
| | `/api/admin/model-status` | GET | Estado del modelo ML |
| | `/api/admin/retrain` | POST | Lanzar reentrenamiento ML |
| | `/api/admin/boost` | GET/POST | Listar/crear boosts |
| | `/api/admin/boost/{id}` | DELETE | Eliminar boost |
| | `/api/admin/boost/search` | GET | Buscar productos para boost |
| **Checkout** | `/api/checkout` | POST | Crear pedido |
| **Clientes** | `/api/customers/top` | GET | Top clientes por gasto |

---

## Scripts

| Script | Descripción | Uso |
|--------|-------------|-----|
| `train_ml.py` | Entrena el modelo SVD de recomendaciones | `python -m scripts.train_ml` |
| `build_rag.py` | Construye/reconstruye el índice ChromaDB | `python -m scripts.build_rag [--force]` |
| `evaluate_agent.py` | Evaluación automatizada del agente con preguntas predefinidas | `python -m scripts.evaluate_agent` |
| `evaluate_ml.py` | Evaluación offline del motor ML (métricas de ranking) | `python -m scripts.evaluate_ml` |
| `evaluate_rag.py` | Evaluación del sistema RAG (precisión de recuperación) | `python -m scripts.evaluate_rag` |
| `extract_attributes.py` | Extrae atributos (piedra, tipo) de productos y los guarda en `ps_ai_product_attr` | `python -m scripts.extract_attributes` |
| `anonimizar.py` | Anonimiza datos personales de la BD para el repositorio público | `python -m scripts.anonimizar` |

---

## Tests y resultados

Suite de 7 baterías de tests. Los resultados se guardan en `tests/results_*.json`.

### Test 01 — API Endpoints

**12/12 pasados** · Latencia media: 25 ms

Verifica todos los endpoints: productos, recomendaciones (usuario conocido y fallback), top clientes, stats admin, boost CRUD, y validación de errores en checkout.

### Test 02 — Motor ML (SVD)

**14/14 pasados**

| Métrica | Valor |
|---------|-------|
| Componentes SVD | 50 |
| Usuarios entrenados | 2.622 |
| Productos en modelo | 1.794 |
| Varianza explicada | 57,4% |
| Cobertura catálogo (100 queries) | 19,0% |
| Diversidad recomendaciones | 100% (listas únicas) |
| Latencia de inferencia | 0,92 ms |

### Test 03 — Evaluación temporal SVD vs Popularidad

SVD supera consistentemente al baseline de popularidad en todos los cortes temporales:

| Métrica (corte 60%) | Popularidad | SVD (k=20) | Mejora |
|----------------------|-------------|------------|--------|
| Hit Rate@5 | 2,4% | 42,6% | ×17,7 |
| MRR | 0,012 | 0,261 | ×21,7 |
| Precision@5 | 0,73% | 13,2% | ×18,1 |
| NDCG@5 | 0,007 | 0,138 | ×19,7 |

### Test 04 — Sistema RAG

**8/8 pasados** · Latencia media: 24,4 s

| Tipo de query | Keywords encontradas | Tasa |
|---------------|---------------------|------|
| Conceptual (intención → minerales) | 14/16 | 87,5% |
| Directa (producto concreto) | 2/2 | 100% |
| Knowledge base (FAQ tienda) | 7/8 | 87,5% |

### Test 05 — Agente de cliente

**8/8 conversaciones exitosas** · Latencia media: 11,8 s

Ejemplo: *"Busco algo para la ansiedad"* → el agente infiere amatista y labradorita, busca en catálogo y devuelve 12 productos relevantes.

### Test 06 — Agente admin

**8/8 conversaciones exitosas** · Latencia media: 5,6 s

Ejemplo: *"Informe de ventas de este mes"* → 30 pedidos, 1.947,82 EUR, ticket medio 64,93 EUR, desglose por estado.

### Test 09 — Rendimiento

**0 errores** en todas las pruebas.

| Endpoint | p50 | p95 |
|----------|-----|-----|
| `/api/products` | 89,8 ms | 91,8 ms |
| `/api/recommendations/{id}` | 4,5 ms | 5,6 ms |
| `/api/admin/stats` | 9,4 ms | 11,0 ms |

| Concurrencia | Workers | RPS | Latencia media | Errores |
|--------------|---------|-----|----------------|---------|
| Baja | 5 | 14,0 | 344 ms | 0 |
| Media | 20 | 16,2 | 958 ms | 0 |
| Alta | 50 | 16,2 | 1.939 ms | 0 |

Inferencia ML: **6,5 ms** de media (p50: 6,4 ms, p95: 8,3 ms).

---

## Base de datos

Esquema basado en **PrestaShop 1.6** con dos tablas auxiliares creadas automáticamente por el sistema:

- **`ps_ai_boost`**: productos con boost de visibilidad para el agente (id_product, boost_factor, reason)
- **`ps_ai_product_attr`**: atributos extraídos de productos (id_product, stone, type) usados para la búsqueda por filtros SQL

El dump completo está en `BD.sql` y se importa automáticamente en el primer arranque (tanto en Docker como en local).

Los datos personales han sido **anonimizados** con el script `scripts/anonimizar.py` usando la librería Faker.
