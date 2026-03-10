import ccxt
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime

# ==========================================
# CONFIG
# ==========================================

INITIAL_CAPITAL = 1000
LOOKBACK = 1200

TIMEFRAMES = ["1m","2m","3m"]
ACCUM_LEVELS = [1,2,3]

TP_VALUES = [0.004,0.006,0.008]
SL_VALUES = [0.006,0.01,0.016]

EMA_PERIOD = 20
DEV_THRESHOLD = 0.003

REPORT_FILE = "quant_research_report.txt"

exchange = ccxt.binance()

# ==========================================
# TOP 100 CRYPTO ASSETS
# ==========================================

SYMBOLS = [
"BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT","DOGE/USDT",
"AVAX/USDT","DOT/USDT","MATIC/USDT","LTC/USDT","BCH/USDT","LINK/USDT","ATOM/USDT",
"FIL/USDT","APT/USDT","ARB/USDT","OP/USDT","INJ/USDT","SEI/USDT","SUI/USDT",
"NEAR/USDT","ICP/USDT","TRX/USDT","ETC/USDT","XLM/USDT","VET/USDT","HBAR/USDT",
"THETA/USDT","EOS/USDT","AAVE/USDT","UNI/USDT","CRV/USDT","DYDX/USDT","GMX/USDT",
"RUNE/USDT","KAS/USDT","TIA/USDT","WIF/USDT","PEPE/USDT","FLOKI/USDT","BONK/USDT",
"JUP/USDT","PYTH/USDT","ORDI/USDT","BLUR/USDT","MAGIC/USDT","STX/USDT","IMX/USDT",
"GALA/USDT","CHZ/USDT","SAND/USDT","MANA/USDT","ENJ/USDT","AXS/USDT","FLOW/USDT",
"MINA/USDT","1INCH/USDT","COMP/USDT","SNX/USDT","ZIL/USDT","KAVA/USDT","ROSE/USDT",
"ALGO/USDT","XTZ/USDT","DASH/USDT","ZEC/USDT","BAT/USDT","OMG/USDT","ANKR/USDT",
"CELO/USDT","GLM/USDT","SKL/USDT","LRC/USDT","OCEAN/USDT","API3/USDT","ENS/USDT",
"MASK/USDT","RNDR/USDT","ILV/USDT","YFI/USDT","BAL/USDT","SUSHI/USDT","COTI/USDT",
"RSR/USDT","LDO/USDT","FXS/USDT","AR/USDT","AGIX/USDT","WOO/USDT","CTSI/USDT",
"ICX/USDT","DGB/USDT","RVN/USDT","SC/USDT","ZEN/USDT","KNC/USDT"
]

# ==========================================
# DATA
# ==========================================

def fetch_data(symbol,timeframe):

    data = exchange.fetch_ohlcv(symbol,timeframe,limit=LOOKBACK)

    df = pd.DataFrame(data,columns=[
        "time","open","high","low","close","volume"
    ])

    return df

# ==========================================
# STRATEGY
# ==========================================

def compute_signals(df):

    df["ema"] = df["close"].ewm(span=EMA_PERIOD).mean()

    dev = (df["close"]-df["ema"])/df["ema"]

    signals = []

    for d in dev:

        if d < -DEV_THRESHOLD:
            signals.append(1)

        elif d > DEV_THRESHOLD:
            signals.append(-1)

        else:
            signals.append(0)

    df["signal"] = signals

    return df

# ==========================================
# ACCUMULATION
# ==========================================

def check_accum(signals,i,n):

    if i-n+1 < 0:
        return 0

    seq = signals[i-n+1:i+1]

    if all(s == 1 for s in seq):
        return 1

    if all(s == -1 for s in seq):
        return -1

    return 0

# ==========================================
# METRICS
# ==========================================

def compute_metrics(trades):

    if len(trades)==0:
        return 0,0,0,0

    wins=[t for t in trades if t>0]
    losses=[t for t in trades if t<=0]

    winrate=len(wins)/len(trades)

    gross_profit=sum(wins)
    gross_loss=abs(sum(losses))

    profit_factor=0
    if gross_loss>0:
        profit_factor=gross_profit/gross_loss

    expectancy=np.mean(trades)

    sharpe=0
    if np.std(trades)>0:
        sharpe=np.mean(trades)/np.std(trades)

    return winrate,profit_factor,expectancy,sharpe

# ==========================================
# BACKTEST
# ==========================================

def run_backtest(df,accum,tp,sl):

    capital=INITIAL_CAPITAL
    peak=capital
    drawdown=0

    position=0
    entry=0

    trades=[]

    signals=df["signal"].values
    prices=df["close"].values

    for i in range(len(df)):

        price=prices[i]

        if position==0:

            sig=check_accum(signals,i,accum)

            if sig==1:
                position=1
                entry=price

            elif sig==-1:
                position=-1
                entry=price

        else:

            pnl=(price-entry)/entry if position==1 else (entry-price)/entry

            if pnl>=tp or pnl<=-sl:

                capital*=1+pnl
                trades.append(pnl)

                position=0

            if capital>peak:
                peak=capital

            dd=(peak-capital)/peak

            if dd>drawdown:
                drawdown=dd

    winrate,pf,expectancy,sharpe=compute_metrics(trades)

    edge_score = expectancy * pf * (1-drawdown)

    return {
        "trades":len(trades),
        "winrate":winrate*100,
        "profit_factor":pf,
        "expectancy":expectancy,
        "sharpe":sharpe,
        "drawdown":drawdown*100,
        "capital":capital,
        "edge":edge_score
    }

# ==========================================
# MAIN RESEARCH LOOP
# ==========================================

results=[]

for symbol in SYMBOLS:

    print("Analyzing",symbol)

    for tf in TIMEFRAMES:

        df=fetch_data(symbol,tf)
        df=compute_signals(df)

        for accum,tp,sl in product(
            ACCUM_LEVELS,
            TP_VALUES,
            SL_VALUES
        ):

            stats=run_backtest(df,accum,tp,sl)

            results.append({
                "symbol":symbol,
                "tf":tf,
                "accum":accum,
                "tp":tp,
                "sl":sl,
                **stats
            })

# ==========================================
# RANKING
# ==========================================

df=pd.DataFrame(results)

df=df.sort_values("edge",ascending=False)

top_strategies=df.head(50)

# ==========================================
# ASSET CLASSIFICATION
# ==========================================

asset_scores=df.groupby("symbol")["edge"].mean()

asset_rank=asset_scores.sort_values(ascending=False)

# ==========================================
# REPORT
# ==========================================

with open(REPORT_FILE,"w") as f:

    f.write("QUANT RESEARCH REPORT\n")
    f.write(str(datetime.utcnow())+"\n\n")

    f.write("TOP STRATEGIES\n")

    f.write(top_strategies.to_string())

    f.write("\n\nASSET RANKING\n")

    f.write(asset_rank.to_string())

print("Report generated:",REPORT_FILE)
