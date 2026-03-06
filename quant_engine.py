import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from numba import njit

# =========================================================
# CONFIG
# =========================================================
INITIAL_CAPITAL = 1000
RISK_PER_TRADE = 0.01

TIMEFRAMES = ["3m","5m","4h"]
TP_VALUES = np.arange(0.001,0.02,0.001)
SL_VALUES = np.arange(0.001,0.02,0.001)
EMA_VALUES = [10,20,30,50]
TRAIL_VALUES = np.arange(0.002,0.02,0.002)

ASSETS = [
    "BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","ADA/USDT",
    "AVAX/USDT","DOGE/USDT","DOT/USDT","MATIC/USDT","LINK/USDT","ATOM/USDT",
    "NEAR/USDT","FTM/USDT","APT/USDT","OP/USDT","ARB/USDT","SUI/USDT",
    "SEI/USDT","INJ/USDT","RUNE/USDT","KAS/USDT","GRT/USDT","AAVE/USDT",
    "UNI/USDT","LDO/USDT","PEPE/USDT","SHIB/USDT","ICP/USDT","FIL/USDT",
    "RNDR/USDT","TIA/USDT","PYTH/USDT","JUP/USDT","BONK/USDT","BLUR/USDT",
    "STX/USDT","ORDI/USDT","WLD/USDT","IMX/USDT","FLOW/USDT","EGLD/USDT",
    "KAVA/USDT","ZIL/USDT","CHZ/USDT","ENS/USDT","DYDX/USDT","SAND/USDT",
    "MANA/USDT","AXS/USDT"
]

STATE_FILE = "state.json"
BACKTEST_FILE = "backtest_results.txt"
LIVE_FILE = "live_trade_log.txt"

LOOKBACK = 1000
LOOKAHEAD = 50
DEVIATION_THRESHOLD = 0.003
FEE = 0.0  # sin fee

DATA_CACHE = {}

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
# INDICATORS
# =========================================================
def indicators(df, ema):
    df["ema"] = df.close.ewm(span=ema).mean()
    ema12 = df.close.ewm(span=12).mean()
    ema26 = df.close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df.macd.ewm(span=9).mean()
    return df

# =========================================================
# NUMBA BACKTEST CORE
# =========================================================
@njit
def backtest_loop(close, high, low, tp, sl):
    capital = 1.0
    trades = 0
    wins = 0
    for i in range(50, len(close)-50):
        entry = close[i]
        tp_price = entry*(1+tp)
        sl_price = entry*(1-sl)
        for j in range(1, 50):
            if high[i+j] >= tp_price:
                capital *= (1+tp)
                wins +=1
                trades+=1
                break
            if low[i+j] <= sl_price:
                capital *= (1-sl)
                trades+=1
                break
    return capital, trades, wins

# =========================================================
# BACKTEST WRAPPER
# =========================================================
def run_backtest(symbol):
    exchange = get_exchange()
    results = []
    for tf in TIMEFRAMES:
        df = fetch_ohlcv(exchange, symbol, tf)
        close = df.close.values
        high = df.high.values
        low = df.low.values
        for tp in TP_VALUES:
            for sl in SL_VALUES:
                capital, trades, wins = backtest_loop(close, high, low, tp, sl)
                if trades == 0:
                    continue
                winrate = wins/trades
                results.append((symbol, tf, tp, sl, capital, winrate, trades))
    return results

# =========================================================
# MULTICPU BACKTEST
# =========================================================
def run_full_backtest():
    with ProcessPoolExecutor() as exe:
        all_results = list(exe.map(run_backtest, ASSETS[:3]))
    flat = [item for sub in all_results for item in sub]
    df = pd.DataFrame(flat, columns=["symbol","tf","tp","sl","capital","winrate","trades"])
    df = df.sort_values("capital", ascending=False)
    df.to_csv(BACKTEST_FILE, index=False)
    print("backtest complete")

# =========================================================
# PERSISTENCE
# =========================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"capital": INITIAL_CAPITAL, "trades": []}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

# =========================================================
# LIVE SIMULATOR CON SEÑALES VALIDAS
# =========================================================
def live_loop():
    exchange = get_exchange()
    state = load_state()
    capital = state["capital"]

    while True:
        for symbol in ASSETS[:10]:
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = ticker["last"]

                # Traer últimas 50 velas
                df = fetch_ohlcv(exchange, symbol, "3m")
                ema = df['close'].ewm(span=20).mean()
                deviation = (df['close'] - ema) / ema
                last_dev = deviation.iloc[-1]

                # Señal
                if last_dev < -0.003:
                    signal = "LONG"
                    tp_price = price*(1+0.008)  # TP validado 0.8%
                    sl_price = price*(1-0.016)  # SL validado 1.6%
                elif last_dev > 0.003:
                    signal = "SHORT"
                    tp_price = price*(1-0.008)
                    sl_price = price*(1+0.016)
                else:
                    signal = "NEUTRAL"
                    tp_price = sl_price = price

                # Imprimir y guardar log
                print(f"{datetime.utcnow()} | {symbol} | Price: {price:.2f} | Capital: {capital:.2f} | Signal: {signal} | TP: {tp_price:.2f} | SL: {sl_price:.2f}")
                with open(LIVE_FILE, "a") as f:
                    f.write(f"{datetime.utcnow()} {symbol} {price} capital={capital} signal={signal} TP={tp_price:.2f} SL={sl_price:.2f}\n")

            except:
                pass

        save_state({"capital": capital})
        time.sleep(60)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Running backtest...")
    run_full_backtest()
    print("Starting live simulation...")
    live_loop()
