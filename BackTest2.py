# -*- coding: utf-8 -*-
"""
PRO Swing Trading Backtester V2 (FINAL FIXED)
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np

# ==============================
# CONFIG
# ==============================
SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "WIPRO",
    "HCLTECH", "ULTRACEMCO", "BAJFINANCE", "NESTLEIND", "TITAN",
    "TECHM", "SUNPHARMA", "ONGC", "NTPC", "POWERGRID",
    "M&M", "BAJAJFINSV", "TATAMOTORS", "TATASTEEL", "ADANIENT",
    "JSWSTEEL", "HINDALCO", "COALINDIA", "DRREDDY", "CIPLA",
    "DIVISLAB", "GRASIM", "HEROMOTOCO", "EICHERMOT", "BPCL",
    "BRITANNIA", "INDUSINDBK", "APOLLOHOSP", "TATACONSUM", "DABUR",
    "PIDILITIND", "SIEMENS", "HAVELLS", "BERGEPAINT", "GODREJCP",
    "MARICO", "MUTHOOTFIN", "BANDHANBNK", "PNB", "CANBK",
    "BANKBARODA", "LUPIN", "BIOCON", "TORNTPHARM", "IPCALAB",
    "GLAND", "ALKEM", "AUROPHARMA", "ZYDUSLIFE", "ABBOTINDIA",
    "SANOFI", "PFIZER", "GLAXO", "CHOLAFIN", "BAJAJ-AUTO",
    "TVSMOTOR", "MOTHERSON", "BOSCHLTD", "BALKRISIND", "MRF",
    "CUMMINSIND", "ABB", "BHEL", "CONCOR", "ADANIPORTS",
    "GMRINFRA", "IRCTC", "DMART", "NYKAA", "ZOMATO",
    "PAYTM", "POLICYBZR", "NAVINFLUOR", "SRF", "AAVAS",
    "CROMPTON", "VOLTAS", "WHIRLPOOL", "BLUESTAR", "VGUARD",
    "HFCL", "RVNL", "IRFC", "RAILTEL", "HUDCO"]
INTERVAL = "15m"
PERIOD = "60d"

EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14

MIN_SCORE = 5
LOOKAHEAD = 100

# ==============================
# INDICATORS
# ==============================
def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(n).mean()
    loss = -delta.clip(upper=0).rolling(n).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(df, n=14):
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()

# ✅ FINAL FIXED ADX (NO ERROR GUARANTEED)
def adx(df, n=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = minus_dm.where(minus_dm > 0, 0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val = tr.rolling(n).mean()

    plus_di = 100 * (plus_dm.rolling(n).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr_val)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0) * 100
    adx_val = dx.rolling(n).mean()

    # Force Series
    return adx_val.astype(float), plus_di.astype(float), minus_di.astype(float)

# ==============================
# MARKET TREND
# ==============================
def get_market_trend():
    df = yf.download("^NSEI", period=PERIOD, interval=INTERVAL, progress=False)

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)

    return "UP" if df.iloc[-1]['EMA20'] > df.iloc[-1]['EMA50'] else "DOWN"

# ==============================
# HIGHER TIMEFRAME TREND
# ==============================
def get_htf_trend(symbol):
    df = yf.download(symbol + ".NS", period=PERIOD, interval="1h", progress=False)

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)

    return "UP" if df.iloc[-1]['EMA20'] > df.iloc[-1]['EMA50'] else "DOWN"

# ==============================
# SIGNAL LOGIC
# ==============================
def get_signal(df, i, market_trend, htf_trend):
    row = df.iloc[i]

    close = row['Close']
    ema20 = row['EMA20']
    ema50 = row['EMA50']
    rsi_val = row['RSI']
    adx_val = row['ADX']
    atr_val = row['ATR']
    volume = row['Volume']
    plus_di = row['PLUS_DI']
    minus_di = row['MINUS_DI']

    avg_vol = df['Volume'].iloc[i-50:i].mean()

    score = 0
    trend = "UP" if ema20 > ema50 else "DOWN"

    # Trend alignment
    if trend != market_trend or trend != htf_trend:
        return None

    # Volume
    if volume > avg_vol * 1.8:
        score += 2

    # RSI continuation
    if 40 < rsi_val < 60:
        score += 1

    # ADX strength
    if adx_val > 25:
        if trend == "UP" and plus_di > minus_di:
            score += 2
        elif trend == "DOWN" and minus_di > plus_di:
            score += 2

    # Breakout
    recent_high = df['High'].iloc[i-20:i].max()
    recent_low = df['Low'].iloc[i-20:i].min()

    breakout_buy = close > recent_high * 0.995
    breakout_sell = close < recent_low * 1.005

    if trend == "UP" and breakout_buy:
        score += 3
        signal = "BUY"
    elif trend == "DOWN" and breakout_sell:
        score += 3
        signal = "SELL"
    else:
        return None

    if score < MIN_SCORE:
        return None

    rr = 2 if adx_val > 30 else 1.5

    if signal == "BUY":
        sl = close - 1.5 * atr_val
        tp = close + (close - sl) * rr
    else:
        sl = close + 1.5 * atr_val
        tp = close - (sl - close) * rr

    return {"signal": signal, "entry": close, "sl": sl, "tp": tp}

# ==============================
# BACKTEST
# ==============================
def backtest(symbol):
    df = yf.download(symbol + ".NS", period=PERIOD, interval=INTERVAL, progress=False)

    if df is None or len(df) < 200:
        return []

    # FIX multi-index columns
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)
    df['RSI'] = rsi(df['Close'])
    df['ATR'] = atr(df)

    # SAFE ADX assignment
    adx_val, plus_di, minus_di = adx(df)
    df['ADX'] = adx_val
    df['PLUS_DI'] = plus_di
    df['MINUS_DI'] = minus_di

    df.dropna(inplace=True)

    market_trend = get_market_trend()
    htf_trend = get_htf_trend(symbol)

    trades = []
    i = 100

    while i < len(df) - LOOKAHEAD:
        sig = get_signal(df, i, market_trend, htf_trend)

        if not sig:
            i += 1
            continue

        future = df.iloc[i:i+LOOKAHEAD]

        for _, r in future.iterrows():
            if sig['signal'] == "BUY":
                if r['Low'] <= sig['sl']:
                    trades.append(-1)
                    break
                if r['High'] >= sig['tp']:
                    trades.append(1)
                    break
            else:
                if r['High'] >= sig['sl']:
                    trades.append(-1)
                    break
                if r['Low'] <= sig['tp']:
                    trades.append(1)
                    break

        i += LOOKAHEAD  # prevent overtrading

    return trades

# ==============================
# MAIN
# ==============================
def main():
    all_results = []

    for sym in SYMBOLS:
        print(f"\nBacktesting {sym}...")
        trades = backtest(sym)

        wins = trades.count(1)
        losses = trades.count(-1)
        total = len(trades)

        win_rate = (wins / total * 100) if total else 0

        print(f"Trades: {total}")
        print(f"Wins: {wins} | Losses: {losses}")
        print(f"Win Rate: {win_rate:.2f}%")

        all_results.extend(trades)

    print("\n===== OVERALL =====")
    wins = all_results.count(1)
    total = len(all_results)

    if total:
        print(f"Total Trades: {total}")
        print(f"Win Rate: {(wins/total*100):.2f}%")
    else:
        print("No trades found")

if __name__ == "__main__":
    main()