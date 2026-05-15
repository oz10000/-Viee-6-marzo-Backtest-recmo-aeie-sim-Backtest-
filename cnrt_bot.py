#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CNRT Bot – Busca y reserva el pasaje más próximo desde LORETO
           Datos hardcodeados (repositorio privado recomendado)
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time, random, re, logging, sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURACIÓN FIJA (hardcodeada)
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
THREADS = 2
MAX_DIAS = 30
TOP_N = 1                    # reservamos el primer pasaje encontrado

# ============================================================
# SESIÓN CON REINTENTOS
# ============================================================
session = requests.Session()
retries = Retry(total=MAX_RETRIES, backoff_factor=1, status_forcelist=[429,500,502,503,504], allowed_methods=["GET","POST"])
session.mount("http://", HTTPAdapter(max_retries=retries))
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Linux; Android 14; SM-G610M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36",
    "Accept-Language": "es-AR,es;q=0.9"
})

origen_id = None
destinos = []
token_csrf = ""
search_action = ""
beneficiario_id = ""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("CNRT")

def limpiar(txt):
    return " ".join(str(txt).split()).strip() if txt else ""

# ----------------------------
# 1. Login
# ----------------------------
def login():
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

# ----------------------------
# 2. Confirmar datos
# ----------------------------
def confirmar_datos(html):
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

# ----------------------------
# 3. Abrir buscador
# ----------------------------
def abrir_buscador(token, beneficiario):
    global token_csrf, search_action, beneficiario_id
    beneficiario_id = beneficiario
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
        return True
    except Exception as e:
        log.error(f"Buscador: {e}")
        return False

# ----------------------------
# 4. Obtener localidades (paginación completa)
# ----------------------------
def obtener_localidades():
    global origen_id, destinos
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
        return False

    # Buscar "LORETO" en la lista
    origen_norm = CONFIG["origen_nombre"].upper().strip()
    # 1) exacto
    for id_loc, nombre in todas:
        if nombre.upper() == origen_norm:
            origen_id = id_loc
            log.info(f"Origen exacto: {nombre} (ID {id_loc})")
            break
    if not origen_id:
        # 2) sin provincia "(SANTIAGO DEL ESTERO)"
        for id_loc, nombre in todas:
            if nombre.upper().replace("(SANTIAGO DEL ESTERO)","").strip() == origen_norm:
                origen_id = id_loc
                log.info(f"Origen sin provincia: {nombre} (ID {id_loc})")
                break
    if not origen_id:
        # 3) coincidencia más corta que contenga "LORETO"
        candidatos = [(id_loc, nombre) for id_loc, nombre in todas if origen_norm in nombre.upper()]
        if candidatos:
            candidatos.sort(key=lambda x: len(x[1]))
            origen_id, nombre_origen = candidatos[0]
            log.warning(f"Origen más corto: {nombre_origen} (ID {origen_id})")
    if not origen_id:
        log.error("No se encontró el origen LORETO.")
        return False

    destinos = [(id_loc, nombre) for id_loc, nombre in todas if id_loc != origen_id]
    log.info(f"Destinos a buscar: {len(destinos)}")
    return True

# ----------------------------
# 5. Parseo de resultados (obtiene link/form de reserva)
# ----------------------------
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
            form_reserva = None
            for a in fila.find_all("a", href=True):
                if "reservar" in a["href"].lower():
                    link = a["href"]
                    link_reserva = link if link.startswith("http") else BASE_URL + link
                    break
            if not link_reserva:
                for f in fila.find_all("form"):
                    if "reservar" in f.get("action", "").lower():
                        form_reserva = f
                        break
            pasajes.append({
                "destino": destino_nombre,
                "fecha": fecha,
                "hora": hora,
                "empresa": empresa,
                "butacas": butacas,
                "link_reserva": link_reserva,
                "form_reserva": form_reserva
            })
        except:
            continue
    return pasajes

# ----------------------------
# 6. Búsqueda por día (paralelo)
# ----------------------------
def buscar_dia(fecha_str):
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
            log.debug(f"Error {dest_nombre}: {e}")
            return []

    resultados = []
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        futures = {executor.submit(consultar, d[0], d[1]): d[1] for d in destinos}
        for futuro in as_completed(futures):
            try:
                resultados.extend(futuro.result())
            except Exception as e:
                log.debug(f"Hilo: {e}")
    return resultados

# ----------------------------
# 7. Búsqueda completa hasta encontrar al menos uno
# ----------------------------
def buscar_pasajes():
    encontrados = []
    hoy = datetime.now()
    for delta in range(0, MAX_DIAS + 1):
        fecha_str = (hoy + timedelta(days=delta)).strftime("%d/%m/%Y")
        log.info(f"Buscando {fecha_str} (día {delta+1}/{MAX_DIAS+1})...")
        pasajes = buscar_dia(fecha_str)
        encontrados.extend(pasajes)
        encontrados.sort(key=lambda p: (p["fecha"], p["hora"]))
        log.info(f"Total acumulado: {len(encontrados)}")
        if len(encontrados) >= TOP_N:
            break
        time.sleep(random.uniform(2, 4))
    return encontrados[:TOP_N]

# ----------------------------
# 8. Reserva automática del pasaje
# ----------------------------
def reservar_pasaje(pasaje):
    log.info(f"Reservando: {pasaje['destino']} - {pasaje['fecha']} {pasaje['hora']} ({pasaje['empresa']})")
    # Método 1: enlace directo
    if pasaje.get("link_reserva"):
        log.info(f"Accediendo a: {pasaje['link_reserva']}")
        try:
            r = session.get(pasaje["link_reserva"], headers={"Referer": search_action}, timeout=TIMEOUT)
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
                    log.info("Enviando formulario de confirmación...")
                    r2 = session.post(action, data=payload, headers={"Referer": pasaje["link_reserva"]}, timeout=TIMEOUT)
                    if r2.status_code == 200 and ("confirmación" in r2.text.lower() or "reserva" in r2.text.lower()):
                        log.info("✅ Reserva completada exitosamente.")
                        return True
                    else:
                        log.error("❌ Falló la confirmación (POST).")
                        return False
                else:
                    log.error("Formulario sin acción.")
                    return False
            else:
                if "confirmación" in r.text.lower() or "reserva" in r.text.lower():
                    log.info("✅ Reserva hecha (GET).")
                    return True
                else:
                    log.error("No se encontró formulario ni confirmación en la página.")
                    return False
        except Exception as e:
            log.error(f"Error en reserva: {e}")
            return False

    # Método 2: formulario incrustado en la fila
    if pasaje.get("form_reserva"):
        log.info("Usando formulario de la fila...")
        form = pasaje["form_reserva"]
        action = form.get("action")
        if not action.startswith("http"):
            action = BASE_URL + action
        payload = {}
        for inp in form.find_all("input"):
            name = inp.get("name")
            value = inp.get("value", "")
            if name:
                payload[name] = value
        try:
            r = session.post(action, data=payload, timeout=TIMEOUT)
            if r.status_code == 200 and ("confirmación" in r.text.lower() or "reserva" in r.text.lower()):
                log.info("✅ Reserva exitosa.")
                return True
            else:
                log.error("Falló la reserva con formulario incrustado.")
                return False
        except Exception as e:
            log.error(f"Error: {e}")
            return False

    log.error("No se encontró método de reserva.")
    return False

# ----------------------------
# MAIN
# ----------------------------
def main():
    print("="*70)
    print("   BOT CNRT – RESERVA DEL PASAJE MÁS PRÓXIMO DESDE LORETO")
    print("="*70)
    print(f" Fecha/hora: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("="*70)

    # 1. Login
    html_login = login()
    if not html_login:
        sys.exit(1)

    # 2. Confirmar datos
    token, beneficiario = confirmar_datos(html_login)
    if not token or not beneficiario:
        sys.exit(1)

    # 3. Abrir buscador
    if not abrir_buscador(token, beneficiario):
        sys.exit(1)

    # 4. Obtener localidades
    if not obtener_localidades():
        sys.exit(1)

    # 5. Buscar el pasaje más próximo
    pasajes = buscar_pasajes()
    if not pasajes:
        log.warning("No se encontraron pasajes disponibles.")
        sys.exit(0)

    # 6. Reservar el primero
    primero = pasajes[0]
    print("\n--- Pasaje seleccionado ---")
    print(f"Destino: {primero['destino']}")
    print(f"Fecha: {primero['fecha']}")
    print(f"Hora: {primero['hora']}")
    print(f"Empresa: {primero['empresa']}")
    print("---------------------------")

    if reservar_pasaje(primero):
        log.info("¡RESERVA EXITOSA! Revisá tu correo.")
    else:
        log.error("No se pudo completar la reserva automática.")
        sys.exit(1)

if __name__ == "__main__":
    main()
