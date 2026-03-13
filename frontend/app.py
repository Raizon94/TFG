from __future__ import annotations
import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
import streamlit as st

# ─── Configuración de página ────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── URL base de la API ─────────────────────────────────────────────
API_BASE: str = f"{BACKEND_URL}"

# Guardar en session_state para que las vistas lo usen
if "api_base" not in st.session_state:
    st.session_state["api_base"] = API_BASE

# ─── Menú lateral ───────────────────────────────────────────────────
st.sidebar.markdown("## 🤖 E-Commerce IA")
st.sidebar.markdown("---")

modo: str = st.sidebar.radio(
    "Navegación",
    options=["🛒 Tienda (Modo Cliente)", "⚙️ Panel de Control (Admin)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption("TFG — E-commerce Headless gestionado por IA")

# ─── Renderizar vista seleccionada ──────────────────────────────────
if modo == "🛒 Tienda (Modo Cliente)":
    from views.customer_view import render_customer_view
    render_customer_view()
else:
    from views.admin_view import render_admin_view
    render_admin_view()
