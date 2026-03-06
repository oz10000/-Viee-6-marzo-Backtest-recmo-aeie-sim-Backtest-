import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================
# CONFIG
# =========================================================
TIMEFRAMES = ["3m"]
TP_VALUES = [0.005, 0.006, 0.007]  # TakeDrop 0.5%,0.6%,0.7%
SL_PCT = 0.016
EMA_SPAN = 20
LOOKBACK = 1000
LOOKAHEAD = 50

RESULTS_FILE = "backtest_take_drop.txt"

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

# =========================================================
# EXCHANGE
# =========================================================
def get_exchange(name="bybit"):
    if name.lower() == "bybit":
        ex = ccxt.bybit()
    else:
        ex = ccxt.binance()
    ex.load_markets()
    return ex

# =========================================================
# FETCH DATA
# =========================================================
def fetch_ohlcv(exchange, symbol, tf):
    try:
        data = exchange.fetch_ohlcv(symbol, tf, limit=LOOKBACK)
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume"])
        return df
    except:
        return None

# =========================================================
# SIGNALS
# =========================================================
def generate_signals(df):
    ema = df['close'].ewm(span=EMA_SPAN).mean()
    deviation = (df['close'] - ema)/ema
    signals = []
    for d in deviation:
        if d < -0.003:
            signals.append(1)   # LONG
        elif d > 0.003:
            signals.append(-1)  # SHORT
        else:
            signals.append(0)   # NEUTRAL
    return np.array(signals)

# =========================================================
# BACKTEST POR PATRÓN
# =========================================================
def backtest_pattern(df, pattern, tp_pct):
    signals = generate_signals(df)
    trades = 0
    wins = 0
    losses = 0
    capital = 1.0
    maes = []
    mfes = []

    i = 0
    while i < len(signals) - len(pattern):
        if np.array_equal(signals[i:i+len(pattern)], pattern):
            entry_price = df['close'].iloc[i+len(pattern)-1]
            direction = pattern[-1]
            tp_price = entry_price * (1 + tp_pct * direction)
            sl_price = entry_price * (1 - SL_PCT * direction)

            high_range = df['high'].iloc[i+len(pattern):i+len(pattern)+LOOKAHEAD]
            low_range = df['low'].iloc[i+len(pattern):i+len(pattern)+LOOKAHEAD]

            if direction == 1:  # LONG
                mae = (entry_price - low_range.min()) / entry_price
                mfe = (high_range.max() - entry_price) / entry_price
                maes.append(mae)
                mfes.append(mfe)
                if high_range.max() >= tp_price:
                    capital *= (1 + tp_pct)
                    wins +=1
                elif low_range.min() <= sl_price:
                    capital *= (1 - SL_PCT)
                    losses +=1
            elif direction == -1:  # SHORT
                mae = (high_range.max() - entry_price) / entry_price
                mfe = (entry_price - low_range.min()) / entry_price
                maes.append(mae)
                mfes.append(mfe)
                if low_range.min() <= tp_price:
                    capital *= (1 + tp_pct)
                    wins +=1
                elif high_range.max() >= sl_price:
                    capital *= (1 - SL_PCT)
                    losses +=1
            trades +=1
            i += len(pattern)
        else:
            i +=1

    winrate = wins / trades if trades > 0 else 0
    avg_mae = np.mean(maes) if maes else 0
    avg_mfe = np.mean(mfes) if mfes else 0

    return {"trades": trades, "wins": wins, "losses": losses,
            "winrate": winrate, "capital": capital,
            "avg_mae": avg_mae, "avg_mfe": avg_mfe}

# =========================================================
# PATRONES
# =========================================================
patterns = [
    [1],
    [1,0,1],
    [0,1],
    [-1],
    [-1,0,-1],
    [0,-1]
]

# =========================================================
# BACKTEST POR ACTIVO (PARALELO)
# =========================================================
def backtest_asset(asset):
    exchange_bybit = get_exchange("bybit")
    exchange_binance = get_exchange("binance")

    df = fetch_ohlcv(exchange_bybit, asset, "3m")
    if df is None:
        df = fetch_ohlcv(exchange_binance, asset, "3m")
    if df is None:
        return {"asset": asset, "status": "no data"}

    results = []
    for tp_pct in TP_VALUES:
        for pattern in patterns:
            res = backtest_pattern(df, pattern, tp_pct)
            res.update({"asset": asset, "pattern": pattern, "tp_pct": tp_pct})
            results.append(res)

    return {"asset": asset, "status": "ok", "results": results}

# =========================================================
# BACKTEST COMPLETO MULTIPROCESO
# =========================================================
def run_backtest_full():
    all_results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(backtest_asset, asset) for asset in ASSETS[:50]]
        for future in as_completed(futures):
            res = future.result()
            all_results.append(res)

    # Guardar en TXT
    with open(RESULTS_FILE, "w") as f:
        for r in all_results:
            f.write(f"Asset: {r['asset']} | Status: {r['status']}\n")
            if r['status'] == "ok":
                for trade in r['results']:
                    f.write(f"  Pattern: {trade['pattern']}, TP: {trade['tp_pct']}, Trades: {trade['trades']}, Wins: {trade['wins']}, Losses: {trade['losses']}, Winrate: {trade['winrate']:.2f}, Capital: {trade['capital']:.4f}, MAE: {trade['avg_mae']:.4f}, MFE: {trade['avg_mfe']:.4f}\n")
            f.write("\n")

    print("Backtest completo guardado en:", RESULTS_FILE)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_backtest_full()
