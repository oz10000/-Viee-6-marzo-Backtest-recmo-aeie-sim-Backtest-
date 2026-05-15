#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNRT Auto-Reserva – Streamlit App
Busca el pasaje más próximo desde LORETO y permite reservarlo.
Datos hardcodeados. Repositorio privado recomendado.
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time, random, re, logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURACIÓN FIJA
# ============================================================
BASE_URL = "https://reservapasajes.cnrt.gob.ar"
CONFIG = {
    "dni": "28799045",
    "cud": "3351466694",
    "sexo": "1",
    "telefono": "1125569223",
    "email": "Walterarmandoponce28799045@gmail.com",
    "origen_nombre": "LORETO",
    "cantidad_pasajes": "1"
}
TIMEOUT = 30
MAX_RETRIES = 3
THREADS = 4
MAX_DIAS = 30

# ============================================================
# SESIÓN GLOBAL (inicializada en cada ejecución)
# ============================================================
def crear_sesion():
    session = requests.Session()
    retries = Retry(total=MAX_RETRIES, backoff_factor=1,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["GET", "POST"])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Linux; Android 14; SM-G610M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36",
        "Accept-Language": "es-AR,es;q=0.9"
    })
    return session

# ============================================================
# FUNCIONES DEL BOT
# ============================================================
def limpiar(txt):
    return " ".join(str(txt).split()).strip() if txt else ""

def login(session):
    log = logging.getLogger("CNRT")
    log.info("Iniciando sesión...")
    payload = {
        "documentoOpciones": "1",
        "nroDocumento": CONFIG["dni"],
        "tipoCredencial": "CUD",
        "nroCredencial": CONFIG["cud"],
        "sexo": CONFIG["sexo"]
    }
    try:
        r = session.post(f"{BASE_URL}/web/ingresar", data=payload,
                         headers={"Referer": f"{BASE_URL}/web/ingresar"}, timeout=TIMEOUT)
        if "Bienvenido" in r.text:
            log.info("Login exitoso.")
            return r.text
        log.error("Login fallido.")
        return None
    except Exception as e:
        log.error(f"Login: {e}")
        return None

def confirmar_datos(session, html):
    log = logging.getLogger("CNRT")
    log.info("Confirmando datos...")
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form")
    if not form:
        return None, None
    token_input = soup.find("input", {"name": "token"})
    id_input = soup.find("input", {"name": "id"})
    if not token_input or not id_input:
        return None, None
    token = token_input.get("value")
    beneficiario = id_input.get("value")
    action = form.get("action")
    if not action.startswith("http"):
        action = BASE_URL + action
    payload = {
        "token": token,
        "id": beneficiario,
        "nroTelefono": CONFIG["telefono"],
        "email": CONFIG["email"],
        "verifica_email": CONFIG["email"]
    }
    try:
        r = session.post(action, data=payload,
                         headers={"Referer": f"{BASE_URL}/web/ingresar"}, timeout=TIMEOUT)
        log.info("Datos confirmados.")
        return token, beneficiario
    except Exception as e:
        log.error(f"Confirmación: {e}")
        return None, None

def abrir_buscador(session, token, beneficiario):
    log = logging.getLogger("CNRT")
    log.info("Abriendo buscador...")
    url = f"{BASE_URL}/web/buscarServicios?beneficiarioCnrtId={beneficiario}&token={token}"
    try:
        r = session.get(url, headers={"Referer": f"{BASE_URL}/web/ingresar"}, timeout=TIMEOUT)
        soup = BeautifulSoup(r.text, "html.parser")
        csrf_input = soup.find("input", {"name": "_csrf"})
        token_csrf = csrf_input.get("value") if csrf_input else ""
        form = soup.find("form", {"id": "formBuscar"}) or soup.find("form")
        search_action = form.get("action") if form else f"{BASE_URL}/web/buscarServicios"
        if not search_action.startswith("http"):
            search_action = BASE_URL + search_action
        log.info(f"Buscador listo. CSRF={'Sí' if token_csrf else 'No'}")
        return token_csrf, search_action, beneficiario
    except Exception as e:
        log.error(f"Buscador: {e}")
        return None, None, None

def obtener_localidades(session):
    log = logging.getLogger("CNRT")
    log.info("Descargando localidades...")
    headers = {
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json",
        "Referer": f"{BASE_URL}/web/buscarServicios"
    }
    todas = []
    page = 1
    while True:
        try:
            r = session.get(f"{BASE_URL}/web/getLocalidades?page={page}", headers=headers, timeout=TIMEOUT)
            data = r.json()
            items = data.get("items", [])
            if not items:
                break
            for item in items:
                if "id" in item and "text" in item:
                    todas.append((item["id"], item["text"].strip()))
            page += 1
            time.sleep(0.1)
        except Exception as e:
            log.error(f"Paginación {page}: {e}")
            break
    log.info(f"Total localidades: {len(todas)}")
    if not todas:
        return None, None, None

    # Buscar origen LORETO
    origen_norm = CONFIG["origen_nombre"].upper().strip()
    origen_id = None
    # 1) exacto
    for id_loc, nombre in todas:
        if nombre.upper() == origen_norm:
            origen_id = id_loc
            break
    if not origen_id:
        # 2) sin provincia
        for id_loc, nombre in todas:
            if nombre.upper().replace("(SANTIAGO DEL ESTERO)","").strip() == origen_norm:
                origen_id = id_loc
                break
    if not origen_id:
        # 3) más corta que contenga
        candidatos = [(id_loc, nombre) for id_loc, nombre in todas if origen_norm in nombre.upper()]
        if candidatos:
            candidatos.sort(key=lambda x: len(x[1]))
            origen_id = candidatos[0][0]
    if not origen_id:
        log.error("No se encontró el origen LORETO.")
        return None, None, None

    destinos = [(id_loc, nombre) for id_loc, nombre in todas if id_loc != origen_id]
    return origen_id, destinos, todas

def parsear_resultados(html, fecha, destino_nombre):
    if "No se encontraron servicios disponibles" in html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    tabla = soup.find("table", class_=re.compile("resultados|servicios|table", re.I))
    if not tabla:
        contenedor = soup.find("div", class_=re.compile("resultados|lista", re.I))
        if contenedor:
            tabla = contenedor.find("table")
    if not tabla:
        return []
    pasajes = []
    for fila in tabla.find_all("tr")[1:]:
        celdas = fila.find_all("td")
        if len(celdas) < 3:
            continue
        try:
            hora = limpiar(celdas[0].get_text())
            empresa = limpiar(celdas[1].get_text())
            butacas = limpiar(celdas[2].get_text()) if len(celdas) >= 3 else "?"
            link_reserva = None
            for a in fila.find_all("a", href=True):
                if "reservar" in a["href"].lower():
                    link = a["href"]
                    link_reserva = link if link.startswith("http") else BASE_URL + link
                    break
            pasajes.append({
                "destino": destino_nombre,
                "fecha": fecha,
                "hora": hora,
                "empresa": empresa,
                "butacas": butacas,
                "link_reserva": link_reserva,
            })
        except:
            continue
    return pasajes

def buscar_todos_los_pasajes(session, origen_id, destinos, token_csrf, search_action):
    log = logging.getLogger("CNRT")
    encontrados = []
    hoy = datetime.now()
    for delta in range(0, MAX_DIAS + 1):
        fecha_obj = hoy + timedelta(days=delta)
        fecha_str = fecha_obj.strftime("%d/%m/%Y")
        log.info(f"Buscando {fecha_str} (día {delta+1}/{MAX_DIAS+1})...")

        payload_base = {
            "origen": origen_id,
            "cantidadPasajes": CONFIG["cantidad_pasajes"],
            "fechaSalida": fecha_str,
        }
        if token_csrf:
            payload_base["_csrf"] = token_csrf

        def consultar(dest_id, dest_nombre):
            payload = payload_base.copy()
            payload["destino"] = dest_id
            try:
                r = session.post(search_action, data=payload, timeout=TIMEOUT)
                return parsear_resultados(r.text, fecha_str, dest_nombre)
            except Exception as e:
                return []

        with ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = {executor.submit(consultar, d[0], d[1]): d[1] for d in destinos}
            for futuro in as_completed(futures):
                try:
                    encontrados.extend(futuro.result())
                except Exception as e:
                    pass

        # Si ya tenemos al menos uno, cortamos para no seguir recorriendo fechas
        if encontrados:
            break
        time.sleep(random.uniform(1, 3))
    return encontrados

def seleccionar_mas_proximo(pasajes):
    """Elige el pasaje con menor diferencia positiva entre fecha/hora y ahora."""
    ahora = datetime.now()
    mejor = None
    mejor_delta = None
    for p in pasajes:
        try:
            fecha_hora_str = f"{p['fecha']} {p['hora']}"
            fecha_hora = datetime.strptime(fecha_hora_str, "%d/%m/%Y %H:%M")
            delta = fecha_hora - ahora
            if delta.total_seconds() > 0:
                if mejor_delta is None or delta < mejor_delta:
                    mejor_delta = delta
                    mejor = p
        except:
            continue
    return mejor, mejor_delta

def reservar_pasaje(session, pasaje):
    log = logging.getLogger("CNRT")
    log.info(f"Reservando: {pasaje['destino']} - {pasaje['fecha']} {pasaje['hora']}")
    if not pasaje.get("link_reserva"):
        log.error("No hay enlace de reserva.")
        return False
    try:
        r = session.get(pasaje["link_reserva"], headers={"Referer": BASE_URL + "/web/buscarServicios"}, timeout=TIMEOUT)
        soup = BeautifulSoup(r.text, "html.parser")
        form = soup.find("form")
        if form:
            action = form.get("action")
            if action:
                if not action.startswith("http"):
                    action = BASE_URL + action
                payload = {}
                for inp in form.find_all("input"):
                    name = inp.get("name")
                    value = inp.get("value", "")
                    if name:
                        payload[name] = value
                r2 = session.post(action, data=payload, headers={"Referer": pasaje["link_reserva"]}, timeout=TIMEOUT)
                if r2.status_code == 200 and ("confirmación" in r2.text.lower() or "reserva" in r2.text.lower()):
                    log.info("Reserva exitosa.")
                    return True
                else:
                    log.error("Fallo la confirmación.")
                    return False
            else:
                log.error("Formulario sin acción.")
                return False
        else:
            if "confirmación" in r.text.lower() or "reserva" in r.text.lower():
                log.info("Reserva por GET.")
                return True
            else:
                log.error("No se encontró confirmación.")
                return False
    except Exception as e:
        log.error(f"Error al reservar: {e}")
        return False

# ============================================================
# APLICACIÓN STREAMLIT
# ============================================================
def main():
    st.set_page_config(page_title="CNRT Auto Reserva", page_icon="🚌")
    st.title("🚌 CNRT Auto Reserva")
    st.markdown("**Origen:** LORETO (Santiago del Estero)")

    # Inicializar estado
    if "session" not in st.session_state:
        st.session_state.session = crear_sesion()
    if "localidades" not in st.session_state:
        st.session_state.localidades = None
    if "pasajes" not in st.session_state:
        st.session_state.pasajes = None
    if "pasaje_seleccionado" not in st.session_state:
        st.session_state.pasaje_seleccionado = None
    if "tiempo_restante" not in st.session_state:
        st.session_state.tiempo_restante = None

    # Botón de búsqueda
    if st.button("🔍 Buscar pasaje más próximo"):
        with st.spinner("Iniciando sesión y buscando localidades..."):
            session = st.session_state.session

            # 1. Login
            html_login = login(session)
            if not html_login:
                st.error("No se pudo iniciar sesión. Verificá los datos.")
                return

            # 2. Confirmar datos
            token, beneficiario = confirmar_datos(session, html_login)
            if not token or not beneficiario:
                st.error("No se pudo confirmar los datos.")
                return

            # 3. Abrir buscador
            token_csrf, search_action, _ = abrir_buscador(session, token, beneficiario)
            if not search_action:
                st.error("No se pudo abrir el buscador.")
                return

            # 4. Obtener localidades (cacheado en st.session_state)
            if not st.session_state.localidades:
                origen_id, destinos, _ = obtener_localidades(session)
                if not origen_id:
                    st.error("No se encontró la localidad de origen (LORETO).")
                    return
                st.session_state.localidades = (origen_id, destinos, token_csrf, search_action)
            else:
                origen_id, destinos, token_csrf, search_action = st.session_state.localidades

            st.info(f"Buscando entre {len(destinos)} destinos...")

            # 5. Buscar pasajes (día por día, deteniéndose al primer hallazgo)
            pasajes = buscar_todos_los_pasajes(session, origen_id, destinos, token_csrf, search_action)
            if not pasajes:
                st.warning("No se encontraron pasajes en los próximos 30 días.")
                return

            # 6. Seleccionar el más próximo en tiempo real
            mejor, delta = seleccionar_mas_proximo(pasajes)
            if not mejor:
                st.warning("No se pudo determinar el pasaje más próximo.")
                return

            st.session_state.pasaje_seleccionado = mejor
            st.session_state.tiempo_restante = delta

        st.success("¡Búsqueda completada!")

    # Mostrar resultado si existe
    if st.session_state.pasaje_seleccionado:
        p = st.session_state.pasaje_seleccionado
        delta = st.session_state.tiempo_restante
        horas, rem = divmod(delta.seconds, 3600)
        minutos, _ = divmod(rem, 60)
        dias = delta.days

        st.markdown("## 🎯 Pasaje más próximo encontrado")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Destino", p["destino"])
            st.metric("Fecha", p["fecha"])
            st.metric("Hora", p["hora"])
        with col2:
            st.metric("Empresa", p["empresa"])
            st.metric("Butacas", p["butacas"])
            if dias > 0:
                st.metric("Sale en", f"{dias}d {horas}h {minutos}m")
            else:
                st.metric("Sale en", f"{horas}h {minutos}m")

        # Botón de reserva
        if st.button("✅ Reservar este pasaje"):
            with st.spinner("Realizando la reserva..."):
                if reservar_pasaje(st.session_state.session, p):
                    st.success("¡Reserva completada! Revisá tu correo electrónico.")
                    # Limpiar para nueva búsqueda
                    st.session_state.pasaje_seleccionado = None
                    st.session_state.tiempo_restante = None
                else:
                    st.error("No se pudo completar la reserva. Intentá de nuevo o hacelo manualmente desde la web de CNRT.")

    # Footer
    st.markdown("---")
    st.caption("Bot privado – Datos hardcodeados. Usar solo con fines personales.")

if __name__ == "__main__":
    main()
