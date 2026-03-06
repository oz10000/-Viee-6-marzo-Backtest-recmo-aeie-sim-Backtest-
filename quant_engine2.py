import ccxt
import pandas as pd
import numpy as np
import time
import json
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

############################
# CONFIG
############################

INITIAL_CAPITAL = 1000
BACKTEST_DAYS = 30

TP_VALUES = np.arange(0.1,1.1,0.1)
SL_VALUES = np.arange(0.1,1.1,0.1)
TS_VALUES = [0,0.1,0.2,0.3,0.4,0.5]

BACKTEST_SYMBOLS = ["BTC/USDT","ETH/USDT","SOL/USDT"]

SCAN_SYMBOLS = [
"BTC/USDT","ETH/USDT","SOL/USDT","BNB/USDT","XRP/USDT","ADA/USDT","DOGE/USDT",
"AVAX/USDT","LINK/USDT","DOT/USDT","MATIC/USDT","LTC/USDT","BCH/USDT","ATOM/USDT",
"FIL/USDT","APT/USDT","ARB/USDT","OP/USDT","INJ/USDT","SEI/USDT","SUI/USDT",
"NEAR/USDT","ICP/USDT","TRX/USDT","ETC/USDT","XLM/USDT","VET/USDT","HBAR/USDT",
"THETA/USDT","EOS/USDT","AAVE/USDT","UNI/USDT","CRV/USDT","DYDX/USDT","GMX/USDT",
"RUNE/USDT","KAS/USDT","TIA/USDT","WIF/USDT","PEPE/USDT","FLOKI/USDT","BONK/USDT",
"JUP/USDT","PYTH/USDT","ORDI/USDT","BLUR/USDT","MAGIC/USDT","STX/USDT","IMX/USDT",
"GALA/USDT"
]

STATE_FILE = "state.json"
TRADES_FILE = "trades_accumulated.txt"

exchange = ccxt.kucoin()

############################
# DATA
############################

def fetch_ohlcv(symbol,timeframe="5m",limit=500):
    data = exchange.fetch_ohlcv(symbol,timeframe,limit=limit)
    df = pd.DataFrame(data,columns=["time","open","high","low","close","volume"])
    return df

############################
# INDICATORS
############################

def ema(series,period):
    return series.ewm(span=period).mean()

def strategy(df):

    df["ema_fast"]=ema(df["close"],9)
    df["ema_slow"]=ema(df["close"],21)

    df["signal"]=0

    df.loc[df["ema_fast"]>df["ema_slow"],"signal"]=1
    df.loc[df["ema_fast"]<df["ema_slow"],"signal"]=-1

    return df

############################
# BACKTEST
############################

def run_backtest(symbol,tp,sl,ts):

    df = fetch_ohlcv(symbol)
    df = strategy(df)

    capital = INITIAL_CAPITAL
    wins=0
    trades=0

    entry=0
    position=0

    for i in range(1,len(df)):

        price=df["close"][i]
        signal=df["signal"][i]

        if position==0:

            if signal==1:

                position=1
                entry=price

        else:

            pnl=(price-entry)/entry*100

            if pnl>=tp:
                capital*=1+pnl/100
                wins+=1
                trades+=1
                position=0

            elif pnl<=-sl:
                capital*=1+pnl/100
                trades+=1
                position=0

    winrate=0
    if trades>0:
        winrate=wins/trades*100

    return {
        "symbol":symbol,
        "tp":tp,
        "sl":sl,
        "ts":ts,
        "winrate":winrate,
        "trades":trades,
        "capital":capital
    }

############################
# OPTIMIZER
############################

def optimize_symbol(symbol):

    results=[]

    for tp in TP_VALUES:
        for sl in SL_VALUES:
            for ts in TS_VALUES:

                r=run_backtest(symbol,tp,sl,ts)
                results.append(r)

    return results

############################
# SAVE RESULTS
############################

def save_optimization(results):

    with open("optimization_matrix.txt","a") as f:

        for r in results:

            line=f"{r['symbol']},{r['tp']},{r['sl']},{r['ts']},{r['winrate']},{r['trades']},{r['capital']}\n"
            f.write(line)

############################
# LIVE TRADER
############################

class LiveTrader:

    def __init__(self):

        self.capital=INITIAL_CAPITAL
        self.open_trade=None

        if os.path.exists(STATE_FILE):
            self.load_state()

    def save_state(self):

        state={
            "capital":self.capital,
            "open_trade":self.open_trade
        }

        with open(STATE_FILE,"w") as f:
            json.dump(state,f)

    def load_state(self):

        with open(STATE_FILE) as f:
            s=json.load(f)

        self.capital=s["capital"]
        self.open_trade=s["open_trade"]

    def log_trade(self,trade):

        with open(TRADES_FILE,"a") as f:
            f.write(trade+"\n")

    def console(self,trade):

        print("""
+----------------------------------+
SYMBOL: {}
SIDE: {}
ENTRY: {}
PRICE: {}
TP: {}
SL: {}
PNL: {}
+----------------------------------+
""".format(
trade["symbol"],
trade["side"],
trade["entry"],
trade["price"],
trade["tp"],
trade["sl"],
trade["pnl"]
))

    def scan(self):

        for symbol in SCAN_SYMBOLS:

            df=fetch_ohlcv(symbol)
            df=strategy(df)

            signal=df["signal"].iloc[-1]
            price=df["close"].iloc[-1]

            if signal==1 and self.open_trade is None:

                self.open_trade={
                    "symbol":symbol,
                    "side":"LONG",
                    "entry":price,
                    "tp":price*1.005,
                    "sl":price*0.995
                }

                print("OPEN TRADE",symbol)

    def update_trade(self):

        if self.open_trade is None:
            return

        symbol=self.open_trade["symbol"]

        ticker=exchange.fetch_ticker(symbol)
        price=ticker["last"]

        entry=self.open_trade["entry"]

        pnl=(price-entry)/entry*100

        if price>=self.open_trade["tp"] or price<=self.open_trade["sl"]:

            self.capital*=1+pnl/100

            trade_line=f"{datetime.utcnow()},{symbol},{entry},{price},{pnl},{self.capital}"
            self.log_trade(trade_line)

            print("CLOSE TRADE",symbol,pnl)

            self.open_trade=None

        else:

            self.console({
                "symbol":symbol,
                "side":"LONG",
                "entry":entry,
                "price":price,
                "tp":self.open_trade["tp"],
                "sl":self.open_trade["sl"],
                "pnl":pnl
            })

    def run(self):

        while True:

            if self.open_trade is None:
                self.scan()

            self.update_trade()

            self.save_state()

            time.sleep(10)

############################
# MAIN
############################

def run_backtests():

    with ProcessPoolExecutor() as exe:

        futures=[]

        for s in BACKTEST_SYMBOLS:
            futures.append(exe.submit(optimize_symbol,s))

        for f in futures:
            res=f.result()
            save_optimization(res)

def main():

    print("BACKTEST START")

    run_backtests()

    print("LIVE SIMULATOR START")

    trader=LiveTrader()
    trader.run()

if __name__=="__main__":
    main()
