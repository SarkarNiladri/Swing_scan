# -*- coding: utf-8 -*-
"""
PRO Swing Trading Backtester V4.2 (Relaxed Filters for More Trades)
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd

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

# --- MODIFIED PARAMETERS (for more trades) ---
LOOKAHEAD = 50                  # Reduced from 100 (faster exit)
MIN_SCORE = 3                   # Lowered from 4 (ADX condition now optional)
MAX_TRADES_PER_DAY = 3          # Increased from 2 (allow more opportunities)

# Volatility filter – lowered threshold (0.3% instead of 0.7%)
MIN_ATR_PERCENT = 0.3

# Strong candle requirement – relaxed from 60% to 40% body/range
STRONG_CANDLE_RATIO = 0.4

# Market trend alignment – set to False to disable (more trades)
REQUIRE_MARKET_TREND = True     # Keep True if you want alignment with Nifty

# Debug mode – prints why signals are rejected
DEBUG = False

# ==============================
# INDICATORS (unchanged)
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

def adx(df, n=14):
    high = df['High']
    low = df['Low']
    close = df['Close']

    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)

    atr_val = tr.rolling(n).mean()

    plus_di = 100 * (plus_dm.rolling(n).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(n).mean() / atr_val)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).fillna(0) * 100
    adx_val = dx.rolling(n).mean()

    return adx_val, plus_di, minus_di

# ==============================
# MARKET TREND (optional)
# ==============================
def get_market_trend():
    df = yf.download("^NSEI", period=PERIOD, interval=INTERVAL, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)

    return "UP" if df.iloc[-1]['EMA20'] > df.iloc[-1]['EMA50'] else "DOWN"

# ==============================
# SIGNAL LOGIC (relaxed filters)
# ==============================
def get_signal(df, i, market_trend):
    row = df.iloc[i]
    prev = df.iloc[i-1]

    close = row['Close']
    open_ = row['Open']
    high = row['High']
    low = row['Low']

    ema20 = row['EMA20']
    ema50 = row['EMA50']
    atr_val = row['ATR']
    adx_val = row['ADX']
    plus_di = row['PLUS_DI']
    minus_di = row['MINUS_DI']

    trend = "UP" if ema20 > ema50 else "DOWN"

    # --- Market trend filter (optional) ---
    if REQUIRE_MARKET_TREND and trend != market_trend:
        if DEBUG:
            print(f"{df.index[i]}: Market trend mismatch ({trend} vs {market_trend})")
        return None

    # --- Volatility filter (lowered threshold) ---
    atr_percent = (atr_val / close) * 100
    if atr_percent < MIN_ATR_PERCENT:
        if DEBUG:
            print(f"{df.index[i]}: Low volatility ({atr_percent:.2f}% < {MIN_ATR_PERCENT}%)")
        return None

    score = 0

    # --- Trend strength (ADX > 20 plus DI crossover) ---
    if adx_val > 20:
        if trend == "UP" and plus_di > minus_di:
            score += 2
        elif trend == "DOWN" and minus_di > plus_di:
            score += 2

    # --- Breakout detection (unchanged) ---
    recent_high = df['High'].iloc[i-20:i].max()
    recent_low = df['Low'].iloc[i-20:i].min()

    breakout_recent = (
        df['Close'].iloc[i-5:i].max() > recent_high or
        df['Close'].iloc[i-5:i].min() < recent_low
    )

    if not breakout_recent:
        if DEBUG:
            print(f"{df.index[i]}: No recent breakout")
        return None

    # --- Pullback (unchanged) ---
    pullback_buy = prev['Close'] < df['Close'].iloc[i-2]
    pullback_sell = prev['Close'] > df['Close'].iloc[i-2]

    # --- Strong candle confirmation (relaxed ratio) ---
    body = abs(close - open_)
    candle_range = high - low

    if candle_range == 0:
        return None

    strong_candle = body > STRONG_CANDLE_RATIO * candle_range

    # --- Final entry logic ---
    if trend == "UP" and breakout_recent and pullback_buy and close > prev['Close'] and strong_candle:
        signal = "BUY"
        score += 3

    elif trend == "DOWN" and breakout_recent and pullback_sell and close < prev['Close'] and strong_candle:
        signal = "SELL"
        score += 3

    else:
        if DEBUG:
            print(f"{df.index[i]}: Entry conditions failed")
        return None

    if score < MIN_SCORE:
        if DEBUG:
            print(f"{df.index[i]}: Score too low ({score} < {MIN_SCORE})")
        return None

    # --- Risk management (unchanged) ---
    rr = 1.2
    if signal == "BUY":
        sl = close - 1.2 * atr_val
        tp = close + (close - sl) * rr
    else:
        sl = close + 1.2 * atr_val
        tp = close - (sl - close) * rr

    return {"signal": signal, "sl": sl, "tp": tp, "entry": close}

# ==============================
# BACKTEST (modified trade recording)
# ==============================
def backtest(symbol):
    df = yf.download(symbol + ".NS", period=PERIOD, interval=INTERVAL, progress=False)

    if df is None or len(df) < 200:
        return []

    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df['EMA20'] = ema(df['Close'], 20)
    df['EMA50'] = ema(df['Close'], 50)
    df['ATR'] = atr(df)

    adx_val, plus_di, minus_di = adx(df)
    df['ADX'] = adx_val
    df['PLUS_DI'] = plus_di
    df['MINUS_DI'] = minus_di

    df.dropna(inplace=True)

    market_trend = get_market_trend() if REQUIRE_MARKET_TREND else None

    trades = []
    i = 100

    trades_today = 0
    current_day = None

    while i < len(df) - LOOKAHEAD:
        day = df.index[i].date()

        if current_day != day:
            current_day = day
            trades_today = 0

        if trades_today >= MAX_TRADES_PER_DAY:
            i += 1
            continue

        sig = get_signal(df, i, market_trend)

        if not sig:
            i += 1
            continue

        # Simulate trade outcome
        future = df.iloc[i:i+LOOKAHEAD]
        hit_sl = False
        hit_tp = False
        exit_price = None

        for idx, r in future.iterrows():
            if sig['signal'] == "BUY":
                if r['Low'] <= sig['sl']:
                    hit_sl = True
                    exit_price = sig['sl']
                    break
                if r['High'] >= sig['tp']:
                    hit_tp = True
                    exit_price = sig['tp']
                    break
            else:  # SELL
                if r['High'] >= sig['sl']:
                    hit_sl = True
                    exit_price = sig['sl']
                    break
                if r['Low'] <= sig['tp']:
                    hit_tp = True
                    exit_price = sig['tp']
                    break

        # If neither SL nor TP hit within lookahead, close at last bar
        if not hit_sl and not hit_tp:
            exit_price = future.iloc[-1]['Close']
            # Determine outcome: win if price moved in direction
            if sig['signal'] == "BUY":
                if exit_price > sig['entry']:
                    hit_tp = True   # treat as win for simplicity
                else:
                    hit_sl = True   # treat as loss
            else:  # SELL
                if exit_price < sig['entry']:
                    hit_tp = True
                else:
                    hit_sl = True

        if hit_tp:
            trades.append(1)
        elif hit_sl:
            trades.append(-1)
        else:
            # Should not happen, but if it does, skip
            pass

        trades_today += 1
        # Move forward by 1 bar instead of LOOKAHEAD to allow more signals
        i += 1

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