"""
frontend/views/customer_view.py — Vista modo cliente
=====================================================
Secciones:
  1. Recomendados para ti (3 columnas)
  2. Catálogo de productos (cuadrícula 4 columnas)
  3. Asistente Virtual (chat en sidebar)
  4. Carrito con cantidades y checkout real contra BD
"""

from __future__ import annotations

import uuid
import json

import requests
import streamlit as st


import os
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def _api(endpoint: str) -> str:
    """Construye la URL completa de la API."""
    return f"{st.session_state['api_base']}{endpoint}"


# ─── Gestión del carrito ────────────────────────────────────────────

def _init_cart() -> None:
    """Inicializa el carrito en session_state si no existe.
    Estructura: dict[product_id] -> {id, name, price, image_url, qty}
    """
    if "carrito" not in st.session_state:
        st.session_state["carrito"] = {}


def _add_to_cart(product: dict) -> None:
    """Añade un producto al carrito (o incrementa cantidad)."""
    _init_cart()
    pid = product["id"]
    cart = st.session_state["carrito"]
    if pid in cart:
        cart[pid]["qty"] += 1
    else:
        cart[pid] = {
            "id": pid,
            "name": product["name"],
            "price": product["price"],
            "image_url": product["image_url"],
            "qty": 1,
        }


# ─── Tarjeta de producto ───────────────────────────────────────────

def _render_product_card(product: dict, key_prefix: str) -> None:
    """Renderiza una tarjeta de producto con imagen, nombre, precio y botón."""
    st.image(product["image_url"], width="stretch")
    st.markdown(f"**{product['name'][:60]}**")
    st.markdown(f"\U0001f4b0 **{product['price']:.2f} \u20ac**")
    if st.button("\U0001f6d2 Añadir al carrito", key=f"{key_prefix}_{product['id']}"):
        _add_to_cart(product)
        st.toast(f"\u2705 {product['name'][:30]}\u2026 añadido al carrito")



# ─── CSS chat principal (estilo ChatGPT) ────────────────────────────
_CUSTOMER_CHAT_CSS = """
<style>
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #1e293b;
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 4px;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) span,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) li,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div {
    color: white !important;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent;
    border-radius: 12px;
    padding: 4px 8px;
    margin-bottom: 4px;
}
[data-testid="stTextInput"] input {
    border-radius: 24px !important;
    font-size: 0.95rem;
}
[data-testid="stForm"] {
    border: none !important;
    padding: 0 !important;
}
</style>
"""


# ─── Sección: Chat Asistente Virtual (área principal) ──────────────

def _render_chat_main() -> None:
    """Interfaz de chat del asistente virtual en el área principal."""
    st.markdown(_CUSTOMER_CHAT_CSS, unsafe_allow_html=True)

    # ── Estado de sesión ───────────────────────────────────────────
    active_customer = st.session_state.get("active_customer")
    current_cid = active_customer.get("id_customer") if active_customer else None

    # Thread estable por cliente (sobrevive recargas porque siempre se
    # recalcula igual).  Anónimos usan un UUID generado una vez.
    stable_tid = f"customer_{current_cid}" if current_cid else None
    if "chat_thread_id" not in st.session_state:
        st.session_state["chat_thread_id"] = stable_tid or str(uuid.uuid4())

    # Detectar cambio de cliente → resetear a thread estable del nuevo
    prev_cid = st.session_state.get("_chat_last_cid", "__uninit__")
    if current_cid != prev_cid:
        st.session_state["_chat_last_cid"] = current_cid
        st.session_state["chat_customer"] = []
        st.session_state["chat_thread_id"] = stable_tid or str(uuid.uuid4())
        st.session_state.pop("_chat_hist_loaded", None)

    if "chat_customer" not in st.session_state:
        st.session_state["chat_customer"] = []

    # ── Recuperar historial del backend (1 vez por sesión) ─────────
    if "_chat_hist_loaded" not in st.session_state:
        st.session_state["_chat_hist_loaded"] = True
        try:
            hist_resp = requests.post(
                _api("/api/chat/customer/history"),
                json={"thread_id": st.session_state["chat_thread_id"]},
                timeout=15,
            )
            if hist_resp.ok:
                messages = hist_resp.json().get("messages", [])
                if messages:
                    st.session_state["chat_customer"] = messages
        except requests.RequestException:
            pass

    # ── Cabecera ───────────────────────────────────────────────────
    hdr_col, btn_col = st.columns([5, 1])
    with hdr_col:
        st.markdown("### 💬 Asistente Virtual")
    with btn_col:
        if st.button(
            "🗑️ Nuevo chat",
            key="btn_new_chat_main",
            use_container_width=True,
            help="Borrar historial e iniciar una nueva conversación",
        ):
            # Borrar historial en el backend (LangGraph SQLite)
            try:
                requests.post(
                    _api("/api/chat/customer/clear"),
                    json={"thread_id": st.session_state["chat_thread_id"]},
                    timeout=10,
                )
            except requests.RequestException:
                pass
            # Resetear estado local (mantener el mismo thread estable)
            st.session_state["chat_customer"] = []
            st.rerun()

    # ── Área de mensajes con scroll ────────────────────────────────
    messages_area = st.container(height=520, border=False)
    with messages_area:
        if not st.session_state["chat_customer"]:
            st.markdown(
                """
                <div style="
                    display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:380px;color:#6b7280;text-align:center;
                ">
                    <div style="font-size:3rem;margin-bottom:12px">💬</div>
                    <div style="font-size:1.1rem;font-weight:600;margin-bottom:8px">
                        ¿En qué puedo ayudarte?
                    </div>
                    <div style="font-size:0.88rem;max-width:360px;line-height:1.6">
                        Puedo recomendarte piedras y minerales, responder dudas
                        sobre productos, consultar tu estado de pedido o ayudarte
                        a encontrar el regalo perfecto.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg_idx, msg in enumerate(st.session_state["chat_customer"]):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # Tarjetas de producto inline bajo la respuesta
                    for p_idx, product in enumerate(msg.get("products", [])):
                        link = product.get("link", "")
                        img_url = product.get("image_url", "")
                        name = product["name"][:55]
                        price = f"{product['price']:.2f} €"
                        img_html = (
                            f'<img src="{img_url}" style="width:52px;height:52px;'
                            f'object-fit:cover;border-radius:8px;flex-shrink:0;" />'
                            if img_url else ""
                        )
                        name_html = (
                            f'<a href="{link}" target="_blank" style="'
                            f'text-decoration:none;color:inherit;font-weight:600;'
                            f'font-size:0.87em;">{name}</a>'
                            if link
                            else f'<b style="font-size:0.87em;">{name}</b>'
                        )
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:10px;'
                            f'padding:6px 0;border-bottom:1px solid #2d3748;">'
                            f'{img_html}'
                            f'<div style="flex:1;min-width:0;">'
                            f'{name_html}<br/>'
                            f'<span style="color:#9ca3af;font-size:0.8em;">{price}</span>'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )
                        if st.button(
                            "🛒 Añadir",
                            key=f"main_chat_add_{msg_idx}_{p_idx}",
                        ):
                            _add_to_cart({
                                "id": product["id"],
                                "name": product["name"],
                                "price": product["price"],
                                "image_url": img_url,
                            })
                            st.toast(f"✅ {product['name'][:30]}… añadido al carrito")

    # ── Input inline (no sticky) ───────────────────────────────────
    with st.form("customer_chat_form", clear_on_submit=True):
        in_col, btn_col = st.columns([9, 1])
        with in_col:
            typed = st.text_input(
                "chat_input_label",
                placeholder="Pregúntame sobre productos, piedras, envíos…",
                label_visibility="collapsed",
                key="customer_chat_text",
            )
        with btn_col:
            submitted = st.form_submit_button("➤", use_container_width=True)
    user_input: str | None = typed.strip() if submitted and typed.strip() else None

    # ── Procesar nuevo mensaje ─────────────────────────────────────
    if user_input:
        st.session_state["chat_customer"].append({"role": "user", "content": user_input})

        payload: dict = {
            "message": user_input,
            "thread_id": st.session_state["chat_thread_id"],
        }
        if active_customer:
            payload["id_customer"] = active_customer["id_customer"]

        status_ph = st.empty()
        try:
            resp = requests.post(
                _api("/api/chat/customer/stream"),
                json=payload,
                timeout=180,
                stream=True,
            )
            reply = "Sin respuesta."
            products: list = []
            status_lines: list[str] = []

            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "status":
                    status_lines.append(event["content"])
                    recent = status_lines[-5:]
                    status_ph.markdown("\n\n".join(
                        f"**{s}**" if i == len(recent) - 1 else s
                        for i, s in enumerate(recent)
                    ))
                elif event.get("type") == "result":
                    data = event["data"]
                    reply = data.get("reply", "Sin respuesta.")
                    products = data.get("products", [])
                    if data.get("thread_id"):
                        st.session_state["chat_thread_id"] = data["thread_id"]

            status_ph.empty()

        except requests.RequestException as e:
            status_ph.empty()
            reply = (
                "⚠️ No se pudo conectar con el asistente. ¿Está el backend arrancado?\n\n"
                f"```\n{type(e).__name__}: {e}\n```"
            )
            products = []

        st.session_state["chat_customer"].append(
            {"role": "assistant", "content": reply, "products": products}
        )
        st.rerun()


# ─── Sección: Carrito completo (sidebar) ───────────────────────────

def _render_cart_sidebar() -> None:
    """Carrito interactivo en sidebar con cantidades, eliminar y checkout."""
    _init_cart()
    cart: dict = st.session_state["carrito"]

    n_items = sum(item["qty"] for item in cart.values()) if cart else 0
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### \U0001f6d2 Carrito ({n_items})")

    if not cart:
        st.sidebar.caption("Tu carrito est\u00e1 vac\u00edo.")
        return

    total: float = 0.0
    items_to_remove: list[int] = []
    needs_rerun: bool = False

    for pid, item in list(cart.items()):
        subtotal = item["price"] * item["qty"]
        total += subtotal

        st.sidebar.markdown(
            f"**{item['name'][:35]}**\n\n"
            f"{item['qty']} \u00d7 {item['price']:.2f} \u20ac = **{subtotal:.2f} \u20ac**"
        )

        col1, col2, col3 = st.sidebar.columns([1, 1, 1])
        with col1:
            if st.button("\u2796", key=f"minus_{pid}"):
                if item["qty"] > 1:
                    item["qty"] -= 1
                else:
                    items_to_remove.append(pid)
                needs_rerun = True
        with col2:
            if st.button("\u2795", key=f"plus_{pid}"):
                item["qty"] += 1
                needs_rerun = True
        with col3:
            if st.button("\U0001f5d1\ufe0f", key=f"del_{pid}"):
                items_to_remove.append(pid)
                needs_rerun = True

    for pid in items_to_remove:
        cart.pop(pid, None)

    if needs_rerun:
        st.rerun()

    shipping = 5.99
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"\U0001f4e6 Env\u00edo: **{shipping:.2f} \u20ac**")
    st.sidebar.markdown(f"### \U0001f4b6 Total: **{total + shipping:.2f} \u20ac**")

    col_vaciar, col_comprar = st.sidebar.columns(2)
    with col_vaciar:
        if st.button("\U0001f5d1\ufe0f Vaciar", key="vaciar_carrito"):
            st.session_state["carrito"] = {}
            st.rerun()
    with col_comprar:
        if st.button("\U0001f4b3 Comprar", key="btn_comprar", type="primary"):
            st.session_state["show_checkout"] = True
            st.rerun()


# ─── Sección: Confirmación de compra ───────────────────────────────

def _render_checkout_dialog() -> None:
    """Muestra la confirmación y procesa el checkout contra la API/BD."""
    if not st.session_state.get("show_checkout"):
        return

    cart: dict = st.session_state.get("carrito", {})
    if not cart:
        st.session_state["show_checkout"] = False
        return

    st.markdown("---")
    st.markdown("## \U0001f4b3 Confirmar Compra")

    total = 0.0
    st.markdown("| Producto | Cant. | Precio ud. | Subtotal |")
    st.markdown("|----------|------:|-----------:|---------:|")
    for item in cart.values():
        subtotal = item["price"] * item["qty"]
        total += subtotal
        st.markdown(
            f"| {item['name'][:40]} | {item['qty']} | "
            f"{item['price']:.2f} \u20ac | {subtotal:.2f} \u20ac |"
        )

    shipping = 5.99
    total_final = total + shipping
    st.markdown(f"\n\U0001f4e6 **Env\u00edo:** {shipping:.2f} \u20ac")
    st.markdown(f"### \U0001f4b6 Total a pagar: **{total_final:.2f} \u20ac**")
    
    active_customer = st.session_state.get("active_customer")
    if active_customer:
        st.info(f"\U0001f464 **Comprando como:** {active_customer['name']} ({active_customer['email']})")
    else:
        st.warning("\u26a0\ufe0f No hay un cliente seleccionado. Selecciona uno en el menú lateral.")

    st.caption("\U0001f512 Pago simulado \u2014 se registrar\u00e1 como pedido real en la BD.")

    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("\u274c Cancelar", key="cancel_checkout"):
            st.session_state["show_checkout"] = False
            st.rerun()
    with col_confirm:
        if st.button("\u2705 Confirmar pedido", key="confirm_checkout", type="primary", disabled=not active_customer):
            items_payload = []
            for item in cart.values():
                items_payload.append({
                    "product_id": item["id"],
                    "product_name": item["name"],
                    "quantity": item["qty"],
                    "unit_price": item["price"],
                })

            payload = {
                "items": items_payload,
                "id_customer": active_customer["id_customer"],
                "id_address": active_customer["id_address"]
            }

            try:
                resp = requests.post(
                    _api("/api/checkout"),
                    json=payload,
                    timeout=15,
                )
                data = resp.json()

                if resp.status_code == 200 and data.get("success"):
                    st.session_state["carrito"] = {}
                    st.session_state["show_checkout"] = False
                    # Guardar mensaje de éxito para mostrarlo tras rerun
                    st.session_state["last_order_msg"] = (
                        f"\U0001f389 **\u00a1Pedido creado!**\n\n"
                        f"- **N\u00ba Pedido:** {data['order_id']}\n"
                        f"- **Referencia:** {data['reference']}\n"
                        f"- **Total pagado:** {data['total_paid']:.2f} \u20ac\n\n"
                        f"Puedes verificar el pedido en phpMyAdmin \u2192 tabla `ps_orders`."
                    )
                    # Rerun para refrescar recomendaciones LIVE
                    st.rerun()
                else:
                    detail = data.get("detail", "Error desconocido")
                    st.error(f"\u274c Error al crear el pedido: {detail}")

            except requests.RequestException as e:
                st.error(f"\u26a0\ufe0f No se pudo conectar con el backend: {e}")


# ─── Vista principal ────────────────────────────────────────────────

def render_customer_view() -> None:
    """Renderiza la vista completa del modo cliente."""

    st.title("\U0001f916 E-Commerce IA")
    st.caption("E-commerce headless gestionado por agentes de IA")

    # ── Selector de Cliente (sidebar) ──
    st.sidebar.markdown("### \U0001f464 Cliente Activo")
    try:
        resp = requests.get(_api("/api/customers/top"), timeout=10)
        top_customers = resp.json()
    except requests.RequestException:
        top_customers = []
        st.sidebar.warning("No se pudieron cargar los clientes.")

    if top_customers:
        # Crear un diccionario para mapear el nombre a los datos del cliente
        customer_options = {
            "👤 Cliente no identificado": None,
            **{
                f"{c['name']} ({c['email']}) - {c['total_spent']:.2f}€": c
                for c in top_customers
            },
        }
        selected_label = st.sidebar.selectbox(
            "Selecciona un cliente:",
            options=list(customer_options.keys())
        )
        st.session_state["active_customer"] = customer_options[selected_label]
    else:
        st.session_state["active_customer"] = None

    st.sidebar.markdown("---")

    # ── Carrito en sidebar ──
    _render_cart_sidebar()

    # ── Diálogo de checkout (cuerpo principal) ──
    _render_checkout_dialog()

    # ── Mensaje de último pedido (tras rerun) ──
    if "last_order_msg" in st.session_state:
        st.balloons()
        st.success(st.session_state.pop("last_order_msg"))

    # ── Recomendados para ti ──
    st.markdown("## \u2b50 Recomendados para ti")

    active_cust = st.session_state.get("active_customer")
    rec_user_id = active_cust["id_customer"] if active_cust else 0

    try:
        resp = requests.get(_api(f"/api/recommendations/{rec_user_id}"), timeout=10)
        rec_data: dict = resp.json()
        recomendados: list[dict] = rec_data.get("recommendations", [])
    except requests.RequestException:
        recomendados = []
        st.warning("\u26a0\ufe0f No se pudieron cargar las recomendaciones. \u00bfEst\u00e1 el backend arrancado?")

    if recomendados:
        cols = st.columns(3)
        for i, product in enumerate(recomendados):
            with cols[i]:
                _render_product_card(product, key_prefix="rec")

    st.markdown("---")

    # ── Asistente Virtual ──
    _render_chat_main()

    st.markdown("---")

    # ── Catálogo ──
    st.markdown("## \U0001f4e6 Cat\u00e1logo de Productos")

    try:
        resp = requests.get(_api("/api/products"), timeout=10)
        productos: list[dict] = resp.json()
    except requests.RequestException:
        productos = []
        st.warning("\u26a0\ufe0f No se pudo cargar el cat\u00e1logo. \u00bfEst\u00e1 el backend arrancado?")

    if productos:
        # Primera fila: novedades (4 más recientes)
        st.markdown("#### 🆕 Novedades")
        cols = st.columns(4)
        for j, col in enumerate(cols):
            if j < len(productos):
                with col:
                    _render_product_card(productos[j], key_prefix="cat")

        # Filas restantes: más vendidos
        if len(productos) > 4:
            st.markdown("#### 🔥 Más vendidos")
            for row_start in range(4, len(productos), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols):
                    idx = row_start + j
                    if idx < len(productos):
                        with col:
                            _render_product_card(productos[idx], key_prefix="cat")
