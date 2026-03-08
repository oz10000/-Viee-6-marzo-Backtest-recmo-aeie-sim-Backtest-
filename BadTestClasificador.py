import ccxt
import pandas as pd
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================================================
# CONFIG
# =========================================================
TIMEFRAMES = ["3m"]
TP_VALUES = [0.005, 0.006, 0.007]
SL_PCT = 0.016
EMA_SPAN = 20
LOOKBACK = 1000
LOOKAHEAD = 50

RESULTS_FILE = "backtest_take_drop.txt"

MAX_WORKERS = 3

# =========================================================
# EXCHANGES (FALLBACK SYSTEM)
# =========================================================
EXCHANGES = ["kucoin", "kraken", "cryptocom", "binance"]

# =========================================================
# ASSETS
# =========================================================
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
# CREATE EXCHANGE
# =========================================================
def create_exchange(name):
    try:
        exchange_class = getattr(ccxt, name)
        exchange = exchange_class({
            "enableRateLimit": True
        })
        exchange.load_markets()
        return exchange
    except:
        return None


# =========================================================
# FETCH OHLCV WITH FALLBACK
# =========================================================
def fetch_ohlcv(symbol, tf):

    for ex_name in EXCHANGES:
        exchange = create_exchange(ex_name)

        if exchange is None:
            continue

        try:
            if symbol not in exchange.symbols:
                continue

            data = exchange.fetch_ohlcv(symbol, tf, limit=LOOKBACK)

            df = pd.DataFrame(
                data,
                columns=["time","open","high","low","close","volume"]
            )

            time.sleep(exchange.rateLimit / 1000)

            return df

        except Exception:
            continue

    return None


# =========================================================
# SIGNALS
# =========================================================
def generate_signals(df):

    ema = df["close"].ewm(span=EMA_SPAN).mean()

    deviation = (df["close"] - ema) / ema

    signals = []

    for d in deviation:

        if d < -0.003:
            signals.append(1)

        elif d > 0.003:
            signals.append(-1)

        else:
            signals.append(0)

    return np.array(signals)


# =========================================================
# BACKTEST
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

            entry_price = df["close"].iloc[i+len(pattern)-1]

            direction = pattern[-1]

            tp_price = entry_price * (1 + tp_pct * direction)

            sl_price = entry_price * (1 - SL_PCT * direction)

            high_range = df["high"].iloc[
                i+len(pattern):i+len(pattern)+LOOKAHEAD
            ]

            low_range = df["low"].iloc[
                i+len(pattern):i+len(pattern)+LOOKAHEAD
            ]

            if direction == 1:

                mae = (entry_price - low_range.min()) / entry_price
                mfe = (high_range.max() - entry_price) / entry_price

                if high_range.max() >= tp_price:

                    capital *= (1 + tp_pct)
                    wins += 1

                elif low_range.min() <= sl_price:

                    capital *= (1 - SL_PCT)
                    losses += 1

            else:

                mae = (high_range.max() - entry_price) / entry_price
                mfe = (entry_price - low_range.min()) / entry_price

                if low_range.min() <= tp_price:

                    capital *= (1 + tp_pct)
                    wins += 1

                elif high_range.max() >= sl_price:

                    capital *= (1 - SL_PCT)
                    losses += 1

            maes.append(mae)
            mfes.append(mfe)

            trades += 1

            i += len(pattern)

        else:
            i += 1

    winrate = wins / trades if trades > 0 else 0
    avg_mae = np.mean(maes) if maes else 0
    avg_mfe = np.mean(mfes) if mfes else 0

    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "capital": capital,
        "avg_mae": avg_mae,
        "avg_mfe": avg_mfe
    }


# =========================================================
# PATTERNS
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
# BACKTEST ASSET
# =========================================================
def backtest_asset(asset):

    df = fetch_ohlcv(asset, "3m")

    if df is None:
        return {"asset": asset, "status": "no data"}

    results = []

    for tp_pct in TP_VALUES:

        for pattern in patterns:

            res = backtest_pattern(df, pattern, tp_pct)

            res.update({
                "asset": asset,
                "pattern": pattern,
                "tp_pct": tp_pct
            })

            results.append(res)

    return {"asset": asset, "status": "ok", "results": results}


# =========================================================
# FULL BACKTEST
# =========================================================
def run_backtest_full():

    all_results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(backtest_asset, asset)
            for asset in ASSETS
        ]

        for future in as_completed(futures):

            res = future.result()

            all_results.append(res)


    with open(RESULTS_FILE, "w") as f:

        for r in all_results:

            f.write(f"Asset: {r['asset']} | Status: {r['status']}\n")

            if r["status"] == "ok":

                for trade in r["results"]:

                    f.write(
                        f"Pattern: {trade['pattern']} "
                        f"TP:{trade['tp_pct']} "
                        f"Trades:{trade['trades']} "
                        f"Wins:{trade['wins']} "
                        f"Losses:{trade['losses']} "
                        f"Winrate:{trade['winrate']:.2f} "
                        f"Capital:{trade['capital']:.4f} "
                        f"MAE:{trade['avg_mae']:.4f} "
                        f"MFE:{trade['avg_mfe']:.4f}\n"
                    )

            f.write("\n")

    print("Backtest terminado →", RESULTS_FILE)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    run_backtest_full()
