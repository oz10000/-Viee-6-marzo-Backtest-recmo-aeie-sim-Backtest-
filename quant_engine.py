import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
INITIAL_CAPITAL = 1000
RISK_PER_TRADE = 0.01

TIMEFRAMES = ["3m"]
ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","ADA/USDT",
    "AVAX/USDT","DOGE/USDT","DOT/USDT","MATIC/USDT"
]

STATE_FILE = "state.json"
LIVE_FILE = "live_trade_log.txt"

LOOKBACK = 50
DEVIATION_THRESHOLD = 0.003
FEE = 0.0  # sin fee

# Colores ASCII para visualización
COLOR_MAP = {"LONG": "\033[32m", "SHORT": "\033[31m", "NEUTRAL": "\033[37m"}  # verde, rojo, gris
EXTRA_COLORS = ["\033[34m", "\033[35m"]  # azul, violeta

# =========================================================
# EXCHANGE ROUTER
# =========================================================
def get_exchange():
    exchanges = [ccxt.kucoin(), ccxt.kraken(), ccxt.binance(), ccxt.bybit()]
    for ex in exchanges:
        try:
            ex.load_markets()
            return ex
        except:
            continue
    raise Exception("no exchange available")

# =========================================================
# FETCH DATA
# =========================================================
def fetch_ohlcv(exchange, symbol, tf):
    data = exchange.fetch_ohlcv(symbol, tf, limit=LOOKBACK)
    df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
    return df

# =========================================================
# PERSISTENCE
# =========================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"capital": INITIAL_CAPITAL}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# =========================================================
# LIVE LOOP CON SCORING ACUMULATIVO
# =========================================================
def live_loop():
    exchange = get_exchange()
    state = load_state()
    capital = state["capital"]

    # Diccionario de scoring acumulativo
    scoring_state = {symbol: [] for symbol in ASSETS}

    while True:
        for symbol in ASSETS:
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = ticker["last"]

                # Traer últimas velas
                df = fetch_ohlcv(exchange, symbol, "3m")
                ema = df['close'].ewm(span=20).mean()
                deviation = (df['close'] - ema) / ema
                last_dev = deviation.iloc[-1]

                # Determinar señal
                if last_dev < -DEVIATION_THRESHOLD:
                    signal = "LONG"
                    tp_price = price*(1+0.008)
                    sl_price = price*(1-0.016)
                elif last_dev > DEVIATION_THRESHOLD:
                    signal = "SHORT"
                    tp_price = price*(1-0.008)
                    sl_price = price*(1+0.016)
                else:
                    signal = "NEUTRAL"
                    tp_price = sl_price = price

                # Actualizar scoring acumulativo (máx 3 niveles)
                scoring_state[symbol].append(signal)
                if len(scoring_state[symbol]) > 3:
                    scoring_state[symbol].pop(0)

                # Preparar display con colores ASCII
                display_signal = ""
                for idx, s in enumerate(scoring_state[symbol]):
                    if s == "NEUTRAL":
                        display_signal += COLOR_MAP[s] + "N "
                    else:
                        color = COLOR_MAP[s] if idx==0 else EXTRA_COLORS[idx-1]
                        display_signal += color + ("L " if s=="LONG" else "S ")
                display_signal += "\033[0m"  # reset

                # Mostrar en terminal
                print(f"{datetime.utcnow()} | {symbol} | Price: {price:.2f} | Capital: {capital:.2f} | Signals: {display_signal}")

                # Guardar log
                with open(LIVE_FILE, "a") as f:
                    f.write(f"{datetime.utcnow()} {symbol} price={price:.2f} signal_seq={scoring_state[symbol]} capital={capital:.2f}\n")

            except Exception as e:
                print(f"Error con {symbol}: {e}")
                continue

        # Guardar estado
        save_state({"capital": capital})
        time.sleep(60)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Starting live simulation with cumulative scoring...")
    live_loop()
