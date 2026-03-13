"""
frontend/views/admin_view.py — Vista panel de administración
=============================================================
Layout 70/30:
  • Izquierda: Chat amplio con el Agente ERP IA
  • Derecha: Métricas (st.metric) desde /api/admin/stats
"""

from __future__ import annotations

import base64

import requests
import streamlit as st


import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def _api(endpoint: str) -> str:
    """Construye la URL completa de la API."""
    return f"{st.session_state['api_base']}{endpoint}"


# ─── Panel de métricas (columna derecha) ────────────────────────────
def _render_stats_panel() -> None:
    """Obtiene y muestra las métricas del negocio."""
    st.markdown("### 📊 Métricas")

    try:
        resp = requests.get(_api("/api/admin/stats"), timeout=10)
        stats: dict = resp.json()
    except requests.RequestException:
        st.error("⚠️ No se pudieron cargar las métricas.")
        return

    st.metric("🛒 Total Pedidos", f"{stats['total_pedidos']:,}")
    st.metric("💰 Ingresos del Mes", f"{stats['ingresos_mes']:,.2f} €")
    st.metric("👥 Clientes Activos", f"{stats['clientes_activos']:,}")
    st.metric("📦 Productos Activos", f"{stats['productos_activos']:,}")
    st.metric("📈 Tasa de Conversión", f"{stats['tasa_conversion']}%")
    st.metric("🧾 Ticket Medio", f"{stats['ticket_medio']:.2f} €")

    st.divider()
    st.markdown("### 🧠 Motor de Recomendaciones")

    if st.button("🔁 Reentrenar modelo", use_container_width=True):
        try:
            resp = requests.post(_api("/api/admin/retrain"), timeout=30)
            payload = resp.json()
            if payload.get("started"):
                st.success("✅ Reentrenamiento lanzado en segundo plano.")
            else:
                st.info(payload.get("message", "ℹ️ Reentrenamiento no iniciado."))
        except requests.RequestException:
            st.error("⚠️ No se pudo lanzar el reentrenamiento. ¿Backend activo?")

    with st.expander("📜 Ver log de reentrenamiento", expanded=False):
        if st.button("🔄 Actualizar log", use_container_width=True):
            try:
                resp = requests.get(
                    _api("/api/admin/retrain-log?lines=300"),
                    timeout=10,
                )
                st.session_state["retrain_log_text"] = resp.json().get("log", "")
            except requests.RequestException:
                st.session_state["retrain_log_text"] = "⚠️ No se pudo leer el log."

        log_to_show = st.session_state.get(
            "retrain_log_text",
            "Pulsa 'Actualizar log' para cargar.",
        )
        st.text_area("Log", log_to_show, height=260)



# ─── Panel de gestión de boosts ─────────────────────────────────────
def _render_boost_panel() -> None:
    """Sección para gestionar los productos con boost de visibilidad IA."""
    st.markdown("### 🚀 Boost de Productos")
    st.caption("Los productos con boost aparecen antes en las respuestas del asistente cuando ya son relevantes para la búsqueda.")

    # ── Listado actual ──────────────────────────────────────────────
    with st.expander("📋 Productos con boost activo", expanded=True):
        if st.button("🔄 Actualizar lista", key="refresh_boosts", use_container_width=True):
            st.session_state.pop("cached_boosts", None)

        if "cached_boosts" not in st.session_state:
            try:
                resp = requests.get(_api("/api/admin/boost"), timeout=10)
                st.session_state["cached_boosts"] = resp.json() if resp.ok else []
            except requests.RequestException:
                st.session_state["cached_boosts"] = []

        boosts: list[dict] = st.session_state.get("cached_boosts", [])
        if not boosts:
            st.info("No hay productos con boost configurado.")
        else:
            for item in boosts:
                cols = st.columns([6, 1])
                with cols[0]:
                    ref_str = f" · ref: `{item['reference']}`" if item.get("reference") else ""
                    st.markdown(
                        f"**{item['name']}**{ref_str} (ID {item['id_product']})  \n"
                        f"Factor: `×{item['boost_factor']}`"
                        + (f"  \n_{item['reason']}_" if item.get("reason") else "")
                    )
                with cols[1]:
                    if st.button("🗑️", key=f"del_boost_{item['id_product']}", help="Eliminar boost"):
                        try:
                            resp = requests.delete(
                                _api(f"/api/admin/boost/{item['id_product']}"),
                                timeout=10,
                            )
                            if resp.status_code == 204:
                                st.success(f"Boost eliminado para producto {item['id_product']}")
                                st.session_state.pop("cached_boosts", None)
                                st.rerun()
                            else:
                                st.error(f"Error: {resp.text}")
                        except requests.RequestException:
                            st.error("⚠️ No se pudo conectar con el backend.")

    # ── Añadir / actualizar boost ───────────────────────────────────
    with st.expander("➕ Añadir / Actualizar boost", expanded=False):
        # Paso 1: buscador
        search_col, btn_col = st.columns([5, 1])
        with search_col:
            search_q = st.text_input(
                "Buscar producto por referencia o ID",
                key="boost_search_q",
                placeholder="Ej: AM-001, 42…",
            )
        with btn_col:
            st.markdown("<br>", unsafe_allow_html=True)  # alinear verticalmente
            do_search = st.button("🔍 Buscar", key="boost_do_search", use_container_width=True)

        if do_search and search_q.strip():
            query = search_q.strip()
            if len(query) < 2 and not query.isdigit():
                st.session_state["boost_search_results"] = []
                st.session_state.pop("boost_selected_id", None)
                st.warning("Introduce un ID numérico o una referencia de al menos 2 caracteres.")
            else:
                try:
                    resp = requests.get(
                        _api(f"/api/admin/boost/search?q={query}"),
                        timeout=10,
                    )
                    st.session_state["boost_search_results"] = resp.json() if resp.ok else []
                    st.session_state.pop("boost_selected_id", None)
                except requests.RequestException:
                    st.session_state["boost_search_results"] = []
                    st.error("⚠️ No se pudo conectar con el backend.")

        results: list[dict] = st.session_state.get("boost_search_results", [])
        if results:
            normalized_results = []
            for row in results:
                if not isinstance(row, dict):
                    continue
                product_id = row.get("id_product")
                if product_id is None:
                    product_id = row.get("id")
                if product_id is None:
                    continue
                normalized_results.append({
                    "id_product": product_id,
                    "name": row.get("name", f"Producto {product_id}"),
                    "reference": row.get("reference"),
                    "price": float(row.get("price", 0.0) or 0.0),
                })
            options = {}
            for row in normalized_results:
                product_id = row.get("id_product")
                if product_id is None:
                    continue
                label = (
                    f"{row.get('name', f'Producto {product_id}')}  "
                    f"(ref: {row.get('reference') or '—'} · "
                    f"ID {product_id} · {float(row.get('price', 0.0) or 0.0):.2f} €)"
                )
                options[label] = product_id
            if options:
                chosen_label = st.selectbox("Selecciona un producto", list(options.keys()), key="boost_select_box")
                chosen_id = options[chosen_label]
                st.session_state["boost_selected_id"] = chosen_id
            else:
                st.session_state.pop("boost_selected_id", None)
                st.warning("Los resultados de búsqueda no traen un identificador válido.")
        elif do_search:
            st.warning("No se encontraron productos para esa búsqueda.")

        # Paso 2: formulario de boost (solo si hay producto seleccionado)
        selected_id: int | None = st.session_state.get("boost_selected_id")
        if selected_id:
            st.divider()
            with st.form("add_boost_form", clear_on_submit=False):
                st.caption(f"Configurando boost para el producto **ID {selected_id}**")
                new_factor = st.number_input("Factor de boost", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
                new_reason = st.text_input("Motivo (opcional)", placeholder="Ej: Promoción de temporada")
                submitted = st.form_submit_button("💾 Guardar boost", use_container_width=True)
                if submitted:
                    try:
                        resp = requests.post(
                            _api("/api/admin/boost"),
                            json={
                                "id_product": selected_id,
                                "boost_factor": new_factor,
                                "reason": new_reason or None,
                            },
                            timeout=10,
                        )
                        if resp.status_code in (200, 201):
                            data = resp.json()
                            st.success(f"✅ Boost ×{new_factor} aplicado a «{data['name']}»")
                            st.session_state.pop("cached_boosts", None)
                            st.session_state.pop("boost_search_results", None)
                            st.session_state.pop("boost_selected_id", None)
                            st.rerun()
                        else:
                            st.error(f"Error {resp.status_code}: {resp.text}")
                    except requests.RequestException:
                        st.error("⚠️ No se pudo conectar con el backend.")


# ─── CSS para el chat estilo ChatGPT ───────────────────────────────
_CHAT_CSS = """
<style>
/* Contenedor de mensajes: quita el scroll horizontal */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px;
}
/* Burbujas de usuario */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #1e293b;
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 4px;
}
/* Burbujas del asistente */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent;
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 4px;
}
/* Input sticky */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    font-size: 0.95rem;
}
</style>
"""

# ─── Chat ERP (columna izquierda) ──────────────────────────────────
def _render_admin_chat() -> None:
    """Interfaz de chat estilo ChatGPT para el agente ERP."""
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    # Inicializar historial y thread_id
    if "chat_admin" not in st.session_state:
        st.session_state["chat_admin"] = []
    if "admin_thread_id" not in st.session_state:
        st.session_state["admin_thread_id"] = None

    # ── cabecera con botón de limpiar ──
    hdr_left, hdr_right = st.columns([5, 1])
    with hdr_left:
        st.markdown("### 🤖 Agente ERP — UniArt Minerales")
    with hdr_right:
        if st.button("🗑️ Limpiar", key="clear_admin_chat", use_container_width=True,
                     help="Borrar historial e iniciar nueva conversación"):
            st.session_state["chat_admin"] = []
            st.session_state["admin_thread_id"] = None
            st.session_state.pop("admin_uploaded_image", None)
            st.rerun()

    # ── Uploader de imagen para productos ──
    with st.expander("📸 Subir imagen de producto", expanded=False):
        uploaded = st.file_uploader(
            "Sube una foto de un producto para análisis automático",
            type=["jpg", "jpeg", "png", "webp"],
            key="admin_image_upload",
            help="La IA analizará la imagen y propondrá una ficha de producto.",
        )
        img_message = st.text_input(
            "Mensaje adicional (opcional)",
            key="admin_image_msg",
            placeholder="Ej: Es un colgante de plata 925, ponlo a 15€…",
        )
        if st.button("📤 Analizar imagen", key="admin_send_image",
                     use_container_width=True, disabled=uploaded is None):
            if uploaded is not None:
                st.session_state["chat_admin"].append({
                    "role": "user",
                    "content": f"📸 *Imagen subida: {uploaded.name}*"
                    + (f"\n{img_message}" if img_message.strip() else ""),
                })
                with st.spinner("🔍 Analizando imagen con IA…"):
                    try:
                        files = {"image": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                        data = {
                            "message": img_message or "",
                            "thread_id": st.session_state["admin_thread_id"] or "",
                        }
                        resp = requests.post(
                            _api("/api/admin/analyze-product-image"),
                            files=files,
                            data=data,
                            timeout=120,
                        )
                        result = resp.json()
                        reply = result.get("reply", "Sin respuesta.")
                        st.session_state["admin_thread_id"] = result.get("thread_id")
                        # Guardar vision_data para referencia
                        if result.get("vision_data"):
                            st.session_state["admin_last_vision"] = result["vision_data"]
                    except requests.RequestException:
                        reply = "⚠️ No se pudo conectar con el backend para analizar la imagen."
                st.session_state["chat_admin"].append({"role": "assistant", "content": reply})
                st.rerun()

    # ── input sticky al fondo (declarar ANTES del área de mensajes) ──
    user_input: str | None = st.chat_input(
        "Escribe tu consulta… (ventas, pedidos, clientes, catálogo…)",
        key="admin_chat_input",
    )

    # ── procesar el nuevo turno antes de renderizar ──
    if user_input:
        st.session_state["chat_admin"].append({"role": "user", "content": user_input})
        with st.spinner("🧠 Pensando…"):
            try:
                resp = requests.post(
                    _api("/api/chat/admin"),
                    json={
                        "message": user_input,
                        "thread_id": st.session_state["admin_thread_id"],
                    },
                    timeout=120,
                )
                data = resp.json()
                reply: str = data.get("reply", "Sin respuesta.")
                st.session_state["admin_thread_id"] = data.get("thread_id")
            except requests.RequestException:
                reply = "⚠️ No se pudo conectar con el Agente ERP. ¿Está el backend arrancado?"
        st.session_state["chat_admin"].append({"role": "assistant", "content": reply})

    # ── área de mensajes con altura fija y scroll automático ──
    messages_area = st.container(height=560, border=False)
    with messages_area:
        if not st.session_state["chat_admin"]:
            st.markdown(
                """
                <div style="
                    display:flex; flex-direction:column; align-items:center;
                    justify-content:center; height:420px; color:#6b7280; text-align:center;
                ">
                    <div style="font-size:3.5rem; margin-bottom:12px">🤖</div>
                    <div style="font-size:1.2rem; font-weight:600; margin-bottom:8px">
                        Asistente ERP listo
                    </div>
                    <div style="font-size:0.9rem; max-width:340px; line-height:1.5">
                        Puedes preguntarme sobre ventas, pedidos, clientes,
                        catálogo o listas de empaquetado.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state["chat_admin"]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])


# ─── Vista principal ────────────────────────────────────────────────
def render_admin_view() -> None:
    """Renderiza el panel de administración."""

    st.title("⚙️ Panel de Control — Admin")
    st.caption("Gestión del e-commerce mediante agentes de IA")

    col_chat, col_stats = st.columns([7, 3])

    with col_chat:
        _render_admin_chat()

    with col_stats:
        _render_stats_panel()

    st.divider()
    _render_boost_panel()
