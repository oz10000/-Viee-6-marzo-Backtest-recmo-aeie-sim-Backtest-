import ccxt
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# =========================================================
# CONFIG
# =========================================================
TIMEFRAME = "3m"
DAYS = 30
LIMIT = 1000

DATA_FOLDER = "data"
LOG_FILE = "download_log.txt"

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
# EXCHANGES
# =========================================================
def create_exchanges():

    exchanges = []

    ex1 = ccxt.binance({"enableRateLimit": True})
    ex1.load_markets()
    exchanges.append(ex1)

    ex2 = ccxt.kucoin({"enableRateLimit": True})
    ex2.load_markets()
    exchanges.append(ex2)

    ex3 = ccxt.okx({"enableRateLimit": True})
    ex3.load_markets()
    exchanges.append(ex3)

    return exchanges

# =========================================================
# FETCH OHLCV
# =========================================================
def fetch_ohlcv_all(exchange, symbol, timeframe, since):

    all_candles = []

    while True:
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=LIMIT)

        if not candles:
            break

        all_candles += candles
        since = candles[-1][0] + 1

        if len(candles) < LIMIT:
            break

        time.sleep(exchange.rateLimit / 1000)

    return all_candles

# =========================================================
# DOWNLOAD ASSET
# =========================================================
def download_asset(symbol, exchanges, since):

    for ex in exchanges:

        try:
            print("Trying", ex.id, symbol)

            candles = fetch_ohlcv_all(ex, symbol, TIMEFRAME, since)

            if len(candles) == 0:
                continue

            df = pd.DataFrame(
                candles,
                columns=["time","open","high","low","close","volume"]
            )

            df["time"] = pd.to_datetime(df["time"], unit="ms")

            return df, ex.id

        except Exception as e:
            print("Fail:", ex.id, symbol)

    return None, None

# =========================================================
# SAVE DATA
# =========================================================
def save_csv(symbol, df):

    name = symbol.replace("/","_")

    path = os.path.join(DATA_FOLDER, f"{name}_{TIMEFRAME}.csv")

    df.to_csv(path, index=False)

# =========================================================
# MAIN
# =========================================================
def main():

    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    exchanges = create_exchanges()

    since = int((datetime.utcnow() - timedelta(days=DAYS)).timestamp() * 1000)

    with open(LOG_FILE, "w") as log:

        for asset in ASSETS:

            print("Downloading", asset)

            df, ex_used = download_asset(asset, exchanges, since)

            if df is None:

                log.write(f"{asset} | FAILED\n")
                continue

            save_csv(asset, df)

            log.write(
                f"{asset} | OK | {ex_used} | rows={len(df)}\n"
            )

            print("Saved", asset, len(df))

    print("Download finished")

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    main()
