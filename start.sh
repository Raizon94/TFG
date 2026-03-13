#!/usr/bin/env bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  start.sh — Arranca todo el stack del TFG E-Commerce IA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Uso:
#    chmod +x start.sh   (solo la primera vez)
#    ./start.sh           → para y reinicia todos los servicios
#    ./start.sh --train   → además re-entrena los modelos ML
#    ./start.sh --stop    → solo para todos los servicios
#    ./start.sh --logs    → muestra el log del agente en tiempo real
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -euo pipefail

# ─── Configuración ──────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/.pyenv/versions/3.13.6/bin/python"
BACKEND_PORT=8000
FRONTEND_PORT=8501
PID_DIR="$PROJECT_DIR/.pids"

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ─── Funciones auxiliares ───────────────────────────────────────

log_ok()   { echo -e "${GREEN}  ✅ $1${NC}"; }
log_info() { echo -e "${BLUE}  ℹ️  $1${NC}"; }
log_warn() { echo -e "${YELLOW}  ⚠️  $1${NC}"; }
log_err()  { echo -e "${RED}  ❌ $1${NC}"; }

header() {
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  🤖 E-Commerce IA — TFG${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
}

save_pid() {
    mkdir -p "$PID_DIR"
    echo "$2" > "$PID_DIR/$1.pid"
}

# ─── Comando: --stop ────────────────────────────────────────────

stop_all() {
    header
    echo ""
    log_info "Parando todos los servicios..."

    kill_port $BACKEND_PORT
    log_ok "Backend (puerto $BACKEND_PORT) parado"

    kill_port $FRONTEND_PORT
    log_ok "Frontend (puerto $FRONTEND_PORT) parado"

    kill_port 8080
    log_ok "PhpMyAdmin (puerto 8080) parado"

    rm -rf "$PID_DIR"
    echo ""
    log_ok "Todos los servicios detenidos."
    echo ""
    exit 0
}

# ─── Parsear argumentos ────────────────────────────────────────

TRAIN=false
for arg in "$@"; do
    case $arg in
        --stop)  stop_all ;;
        --train) TRAIN=true ;;
        --logs)
            log_err "Los logs se muestran directamente en la terminal."
            exit 0
            ;;
        *)       log_err "Argumento desconocido: $arg"; exit 1 ;;
    esac
done

# ─── Inicio ────────────────────────────────────────────────────

header

# 1. Verificar Python
echo ""
echo -e "${BOLD}[1/5] Verificando Python...${NC}"
if [ ! -f "$PYTHON" ]; then
    log_err "Python no encontrado en $PYTHON"
    log_info "Instálalo con: pyenv install 3.13.6"
    exit 1
fi
PYVER=$("$PYTHON" --version 2>&1)
log_ok "$PYVER detectado"

# 2. Verificar MySQL
echo ""
echo -e "${BOLD}[2/5] Verificando MySQL...${NC}"
if ! command -v mysql &>/dev/null; then
    log_err "MySQL no está instalado. Ejecuta: brew install mysql"
    exit 1
fi

if ! mysqladmin ping -u root -proot1234 --silent 2>/dev/null; then
    log_warn "MySQL no está corriendo. Intentando arrancar..."
    brew services start mysql 2>/dev/null || brew services start mysql@9.6 2>/dev/null
    sleep 3
    if ! mysqladmin ping -u root -proot1234 --silent 2>/dev/null; then
        log_err "No se pudo arrancar MySQL."
        exit 1
    fi
fi
log_ok "MySQL corriendo"

# Verificar que la BD existe
if ! mysql -u root -proot1234 -e "USE tfg_bd" 2>/dev/null; then
    log_err "La base de datos 'tfg_bd' no existe. Importa BD.sql primero."
    exit 1
fi
log_ok "Base de datos 'tfg_bd' accesible"

# 3. Instalar dependencias si hace falta
echo ""
echo -e "${BOLD}[3/5] Verificando dependencias Python...${NC}"
if ! "$PYTHON" -c "import fastapi, streamlit, sklearn, joblib" 2>/dev/null; then
    log_warn "Faltan dependencias. Instalando..."
    "$PYTHON" -m pip install -q -r "$PROJECT_DIR/requirements.txt"
    log_ok "Dependencias instaladas"
else
    log_ok "Todas las dependencias OK"
fi

# 4. Entrenar modelos ML (si se pide o no existen)
echo ""
echo -e "${BOLD}[4/5] Modelos de Machine Learning...${NC}"
WHAT_MODEL="$PROJECT_DIR/backend/models/svd_what.joblib"

if [ "$TRAIN" = true ] || [ ! -f "$WHAT_MODEL" ]; then
    if [ "$TRAIN" = true ]; then
        log_info "Re-entrenamiento solicitado con --train"
    else
        log_warn "No se encontró el modelo SVD. Entrenando por primera vez..."
    fi
    echo ""
    cd "$PROJECT_DIR"
    "$PYTHON" -m scripts.train_ml
    echo ""
    log_ok "Modelo entrenado y guardado"
else
    WHAT_SIZE=$(du -h "$WHAT_MODEL" | cut -f1 | xargs)
    log_ok "Modelo SVD ya existe ($WHAT_SIZE)"
    log_info "Usa ./start.sh --train para re-entrenar"
fi

# 5. Arrancar servicios
echo ""
echo -e "${BOLD}[5/5] Reiniciando servicios...${NC}"

# Parar todo lo que pueda estar corriendo
kill_port $BACKEND_PORT && log_info "Backend anterior parado" || true
kill_port $FRONTEND_PORT && log_info "Frontend anterior parado" || true
kill_port 8080 && log_info "PhpMyAdmin anterior parado" || true
sleep 1

# Backend (FastAPI + Uvicorn)
# Cargar proxy si está definido en .env
PROXY_URL=""
if [ -f "$PROJECT_DIR/.env" ]; then
    PROXY_URL=$(grep -E '^PROXY_URL=' "$PROJECT_DIR/.env" | cut -d= -f2- | tr -d '"\r' || true)
fi

cd "$PROJECT_DIR"
if [ -n "$PROXY_URL" ]; then
    log_info "Proxy activo: $PROXY_URL"
    http_proxy="$PROXY_URL" https_proxy="$PROXY_URL" \
    $PYTHON -m uvicorn backend.main:app \
        --host 0.0.0.0 \
        --port $BACKEND_PORT \
        --reload \
        --log-level info \
        &
else
    $PYTHON -m uvicorn backend.main:app \
        --host 0.0.0.0 \
        --port $BACKEND_PORT \
        --reload \
        --log-level info \
        &
fi
BACKEND_PID=$!
save_pid "backend" $BACKEND_PID

# Esperar a que el backend esté listo
echo -ne "  Esperando backend"
for i in $(seq 1 15); do
    if curl -s "http://localhost:$BACKEND_PORT/docs" > /dev/null 2>&1; then
        break
    fi
    echo -n "."
    sleep 1
done
echo ""
log_ok "Backend arrancado (PID: $BACKEND_PID) → http://localhost:$BACKEND_PORT"
log_info "Documentación API → http://localhost:$BACKEND_PORT/docs"

# Frontend (Streamlit)
cd "$PROJECT_DIR/frontend"
$PYTHON -m streamlit run app.py \
    --server.port $FRONTEND_PORT \
    --server.headless true \
    --server.address 0.0.0.0 \
    &
FRONTEND_PID=$!
save_pid "frontend" $FRONTEND_PID

sleep 3
log_ok "Frontend arrancado (PID: $FRONTEND_PID) → http://localhost:$FRONTEND_PORT"

# PhpMyAdmin (puerto 8080) — servido con PHP built-in server
PHPMYADMIN_DIR="/opt/homebrew/share/phpmyadmin"
if [ -f "$PHPMYADMIN_DIR/index.php" ]; then
    kill_port 8080
    log_info "Arrancando PhpMyAdmin en http://localhost:8080 ..."
    php -S localhost:8080 -t "$PHPMYADMIN_DIR" > /dev/null 2>&1 &
    PHPMYADMIN_PID=$!
    save_pid "phpmyadmin" "$PHPMYADMIN_PID"
    sleep 1
    log_ok "PhpMyAdmin arrancado (PID: $PHPMYADMIN_PID) → http://localhost:8080"
else
    log_warn "PhpMyAdmin no encontrado en $PHPMYADMIN_DIR. Instálalo con: brew install phpmyadmin"
fi

# ─── Resumen final ─────────────────────────────────────────────

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  ✅ Todo arrancado correctamente${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  🛒 Tienda:       ${BOLD}http://localhost:$FRONTEND_PORT${NC}"
echo -e "  ⚙️  API:          ${BOLD}http://localhost:$BACKEND_PORT${NC}"
echo -e "  📖 API Docs:     ${BOLD}http://localhost:$BACKEND_PORT/docs${NC}"
echo ""
echo -e "  Para parar todo:  ${BOLD}./start.sh --stop${NC}"
echo -e "  Re-entrenar ML:   ${BOLD}./start.sh --train${NC}"
echo ""
