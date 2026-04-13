# -*- coding: utf-8 -*-
"""
Automated Swing Trading Scanner
================================
Features:
  - Nifty 100 stocks scanned every 15 minutes during market hours
  - Indicators : RSI, MACD, EMA, Bollinger Bands, ADX
  - Filters    : Volume spike, Nifty 50 trend, News sentiment
  - S/R        : Swing highs/lows, 52-week high/low, round numbers
  - Candles    : Hammer, Engulfing, Morning/Evening Star, Shooting Star
  - Stop Loss  : ATR-based dynamic
  - Scoring    : Out of 13 points, High confidence = 8+
  - Output     : Console table, no duplicate signals per day

Requirements:
    pip install yfinance pandas talib requests rich feedparser pytz

Run:
    python swing_scanner_auto.py
"""

import warnings
warnings.filterwarnings("ignore")
import Symbols
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import time
import pytz
from datetime import datetime, date
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

SCAN_INTERVAL_SECONDS = 15 * 60
IST                   = pytz.timezone("Asia/Kolkata")
MARKET_OPEN           = (9, 15)
MARKET_CLOSE          = (15, 30)

# Indicator settings
RSI_PERIOD     = 14
RSI_OVERSOLD   = 40
BB_PERIOD      = 20
BB_STD         = 2
EMA_SHORT      = 20
EMA_LONG       = 50
ADX_PERIOD     = 14
ADX_THRESHOLD  = 25
ATR_PERIOD     = 14
VOLUME_MULT    = 1.5
SR_ZONE_PCT    = 0.015
RISK_REWARD    = 2.0
MIN_SCORE      = 8

console = Console()

# ─────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────
def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi, index=data.index)

def calculate_ema(data, period):
    """Calculate EMA indicator"""
    return data.ewm(span=period, adjust=False).mean()

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bbands(data, period=20, num_std=2):
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def calculate_atr(df, period=14):
    """Calculate ATR (Average True Range)"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(df, period=14):
    """Calculate ADX (Average Directional Index)"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    
    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di
    di_sum[di_sum == 0] = 1
    
    dx = 100 * (di_diff / di_sum)
    adx = dx.rolling(window=period).mean()
    
    return adx, plus_di, minus_di

console = Console()

# ─────────────────────────────────────────────
# NIFTY 100 SYMBOLS
# ─────────────────────────────────────────────
NIFTY100_SYMBOLS = Symbols.symbols


# ─────────────────────────────────────────────
# NEWS SENTIMENT
# ─────────────────────────────────────────────
def get_news_sentiment(symbol: str) -> tuple:
    positive_words = [
        "surge", "rally", "gain", "profit", "growth", "beat", "upgrade",
        "outperform", "buy", "strong", "record", "boost", "expansion",
        "deal", "order", "positive", "rise", "bullish", "momentum",
        "breakout", "revenue", "acquisition"
    ]
    negative_words = [
        "fall", "drop", "loss", "decline", "downgrade", "miss", "weak",
        "sell", "cut", "reduce", "concern", "risk", "fraud", "penalty",
        "fine", "negative", "bearish", "crash", "debt", "warning",
        "lawsuit", "probe", "slump", "plunge"
    ]
    try:
        url  = f"https://news.google.com/rss/search?q={symbol}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        entries = feed.entries[:10]
        if not entries:
            return 0.0, "Neutral"
        score = 0
        for entry in entries:
            text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
            for w in positive_words:
                if w in text: score += 1
            for w in negative_words:
                if w in text: score -= 1
        normalized = score / max(len(entries), 1)
        if normalized > 0.3:    label = "Positive"
        elif normalized < -0.3: label = "Negative"
        else:                   label = "Neutral"
        return round(normalized, 2), label
    except Exception:
        return 0.0, "Neutral"


# ─────────────────────────────────────────────
# NIFTY 50 TREND
# ─────────────────────────────────────────────
def get_nifty_trend() -> str:
    try:
        df = yf.download("^NSEI", period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        
        df['EMA_20'] = calculate_ema(df['Close'], 20)
        df['EMA_50'] = calculate_ema(df['Close'], 50)
        
        df.dropna(inplace=True)
        last  = df.iloc[-1]
        ema20 = float(last.get("EMA_20", 0))
        ema50 = float(last.get("EMA_50", 0))
        close = float(last["Close"])
        if close > ema20 > ema50:   return "bullish"
        elif close < ema20 < ema50: return "bearish"
        else:                       return "neutral"
    except Exception:
        return "neutral"


# ─────────────────────────────────────────────
# SUPPORT / RESISTANCE
# ─────────────────────────────────────────────
def find_sr_levels(df: pd.DataFrame, close: float) -> tuple:
    levels = []

    # Swing highs/lows
    highs = df["High"].rolling(5, center=True).max()
    lows  = df["Low"].rolling(5, center=True).min()
    levels.extend(df["High"][df["High"] == highs].tail(50).tolist())
    levels.extend(df["Low"][df["Low"] == lows].tail(50).tolist())

    # 52-week high/low
    levels.append(float(df["High"].tail(252 * 26).max()))
    levels.append(float(df["Low"].tail(252 * 26).min()))

    # Round numbers
    for base in [50, 100]:
        for offset in [-1, 0, 1]:
            levels.append((round(close / base) + offset) * base)

    zone = close * SR_ZONE_PCT
    near_support    = any(abs(close - l) <= zone and l <= close for l in levels if l > 0)
    near_resistance = any(abs(close - l) <= zone and l >= close for l in levels if l > 0)
    return near_support, near_resistance


# ─────────────────────────────────────────────
# CANDLESTICK PATTERNS
# ─────────────────────────────────────────────
def detect_candle_pattern(df: pd.DataFrame) -> tuple:
    if len(df) < 3:
        return False, False, ""

    c1 = df.iloc[-3]
    c2 = df.iloc[-2]
    c3 = df.iloc[-1]

    o3, h3, l3, c3c = float(c3["Open"]), float(c3["High"]), float(c3["Low"]), float(c3["Close"])
    o2, h2, l2, c2c = float(c2["Open"]), float(c2["High"]), float(c2["Low"]), float(c2["Close"])
    o1, h1, l1, c1c = float(c1["Open"]), float(c1["High"]), float(c1["Low"]), float(c1["Close"])

    body3         = abs(c3c - o3)
    range3        = h3 - l3 if h3 != l3 else 0.001
    upper_wick3   = h3 - max(c3c, o3)
    lower_wick3   = min(c3c, o3) - l3

    # Bullish
    hammer        = lower_wick3 >= 2 * body3 and upper_wick3 <= 0.1 * range3 and body3 > 0
    bull_engulf   = c2c < o2 and c3c > o3 and o3 <= c2c and c3c >= o2
    morning_star  = (c1c < o1 and
                     abs(c2c - o2) < 0.3 * (h2 - l2 if h2 != l2 else 0.001) and
                     c3c > o3 and c3c > (o1 + c1c) / 2)

    # Bearish
    shooting_star = upper_wick3 >= 2 * body3 and lower_wick3 <= 0.1 * range3 and body3 > 0
    bear_engulf   = c2c > o2 and c3c < o3 and o3 >= c2c and c3c <= o2
    evening_star  = (c1c > o1 and
                     abs(c2c - o2) < 0.3 * (h2 - l2 if h2 != l2 else 0.001) and
                     c3c < o3 and c3c < (o1 + c1c) / 2)

    bullish = hammer or bull_engulf or morning_star
    bearish = shooting_star or bear_engulf or evening_star

    if hammer:          name = "Hammer"
    elif bull_engulf:   name = "Bullish Engulfing"
    elif morning_star:  name = "Morning Star"
    elif shooting_star: name = "Shooting Star"
    elif bear_engulf:   name = "Bearish Engulfing"
    elif evening_star:  name = "Evening Star"
    else:               name = ""

    return bullish, bearish, name


# ─────────────────────────────────────────────
# CORE STOCK ANALYSIS
# ─────────────────────────────────────────────
def analyze_stock(symbol: str, nifty_trend: str) -> dict | None:
    ticker = symbol + ".NS"
    try:
        df = yf.download(ticker, period="60d", interval="15m",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < EMA_LONG + 20:
            return None

        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)

        # Calculate indicators
        df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
        macd, macd_signal, macd_hist = calculate_macd(df['Close'], 12, 26, 9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACDh'] = macd_hist
        bb_upper, bb_middle, bb_lower = calculate_bbands(df['Close'], BB_PERIOD, BB_STD)
        df['BB_Upper'] = bb_upper
        df['BB_Lower'] = bb_lower
        df['EMA_Short'] = calculate_ema(df['Close'], EMA_SHORT)
        df['EMA_Long'] = calculate_ema(df['Close'], EMA_LONG)
        df['ADX'], df['Plus_DI'], df['Minus_DI'] = calculate_adx(df, ADX_PERIOD)
        df['ATR'] = calculate_atr(df, ATR_PERIOD)

        df.dropna(inplace=True)
        if len(df) < 5:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close     = float(last["Close"])
        volume    = float(last["Volume"])
        rsi       = float(last.get("RSI", 50))
        macd_hist = float(last.get("MACDh", 0))
        prev_macd = float(prev.get("MACDh", 0))
        bb_lower  = float(last.get("BB_Lower", close))
        bb_upper  = float(last.get("BB_Upper", close))
        ema_short = float(last.get("EMA_Short", close))
        ema_long  = float(last.get("EMA_Long", close))
        adx       = float(last.get("ADX", 0))
        atr       = float(last.get("ATR", close * 0.01))

        # Volume filter — must be > 1.5x average
        avg_vol = df["Volume"].tail(20 * 26).mean()
        if volume <= avg_vol * VOLUME_MULT:
            return None

        buy_score  = 0
        sell_score = 0
        reasons    = []

        # 1. RSI (+2)
        if rsi < RSI_OVERSOLD:
            buy_score += 2; reasons.append(f"RSI({rsi:.0f}) oversold")
        elif rsi > (100 - RSI_OVERSOLD):
            sell_score += 2; reasons.append(f"RSI({rsi:.0f}) overbought")

        # 2. MACD (+2)
        if macd_hist > 0 and prev_macd <= 0:
            buy_score += 2; reasons.append("MACD up cross")
        elif macd_hist < 0 and prev_macd >= 0:
            sell_score += 2; reasons.append("MACD down cross")
        elif macd_hist > 0:
            buy_score += 1
        elif macd_hist < 0:
            sell_score += 1

        # 3. Bollinger Bands (+2)
        if close <= bb_lower:
            buy_score += 2; reasons.append("At BB lower")
        elif close >= bb_upper:
            sell_score += 2; reasons.append("At BB upper")

        # 4. EMA trend (+1)
        if ema_short > ema_long:
            buy_score += 1; reasons.append("EMA bullish")
        else:
            sell_score += 1; reasons.append("EMA bearish")

        # 5. ADX (+2)
        if adx > ADX_THRESHOLD:
            if ema_short > ema_long:
                buy_score += 2
            else:
                sell_score += 2
            reasons.append(f"ADX({adx:.0f}) strong")

        # 6. Support / Resistance (+2)
        near_sup, near_res = find_sr_levels(df, close)
        if near_sup:
            buy_score += 2; reasons.append("Near support")
        if near_res:
            sell_score += 2; reasons.append("Near resistance")

        # 7. Candlestick pattern (+2)
        bull_c, bear_c, c_name = detect_candle_pattern(df)
        if bull_c:
            buy_score += 2; reasons.append(c_name)
        elif bear_c:
            sell_score += 2; reasons.append(c_name)

        # Determine signal
        if buy_score >= MIN_SCORE:
            signal, score = "BUY", buy_score
        elif sell_score >= MIN_SCORE:
            signal, score = "SELL", sell_score
        else:
            return None

        # Nifty trend filter
        if signal == "BUY"  and nifty_trend == "bearish": return None
        if signal == "SELL" and nifty_trend == "bullish": return None

        # News sentiment filter
        _, sentiment_label = get_news_sentiment(symbol)
        if signal == "BUY"  and sentiment_label == "Negative": return None
        if signal == "SELL" and sentiment_label == "Positive": return None

        # ATR-based stop loss & target
        if signal == "BUY":
            stop_loss = round(close - 1.5 * atr, 2)
            target    = round(close + (close - stop_loss) * RISK_REWARD, 2)
        else:
            stop_loss = round(close + 1.5 * atr, 2)
            target    = round(close - (stop_loss - close) * RISK_REWARD, 2)

        return {
            "symbol":    symbol,
            "signal":    signal,
            "entry":     round(close, 2),
            "target":    target,
            "stop_loss": stop_loss,
            "score":     score,
            "rsi":       round(rsi, 1),
            "adx":       round(adx, 1),
            "sentiment": sentiment_label,
            "candle":    c_name or "--",
            "reasons":   " | ".join(reasons),
        }

    except Exception:
        return None


# ─────────────────────────────────────────────
# CONSOLE TABLE
# ─────────────────────────────────────────────
def print_results(results: list, scan_time: str):
    if not results:
        console.print(f"[dim]{scan_time} — No high-confidence signals this scan.[/dim]")
        return

    table = Table(
        title=f"Swing Trade Signals  |  {scan_time}",
        box=box.ROUNDED, show_lines=True,
        header_style="bold white on dark_blue",
    )
    table.add_column("Symbol",    style="bold", width=14)
    table.add_column("Signal",    justify="center", width=8)
    table.add_column("Entry Rs",  justify="right",  width=10)
    table.add_column("Target Rs", justify="right",  width=10)
    table.add_column("SL Rs",     justify="right",  width=10)
    table.add_column("Score",     justify="center", width=7)
    table.add_column("RSI",       justify="right",  width=6)
    table.add_column("ADX",       justify="right",  width=6)
    table.add_column("Candle",    width=18)
    table.add_column("News",      justify="center", width=10)
    table.add_column("Reasons",   width=38)

    for r in results:
        sig_c  = "bold green" if r["signal"] == "BUY" else "bold red"
        tgt_c  = "green"      if r["signal"] == "BUY" else "red"
        sl_c   = "red"        if r["signal"] == "BUY" else "green"
        news_c = {"Positive": "green", "Negative": "red"}.get(r["sentiment"], "yellow")

        table.add_row(
            r["symbol"],
            Text(r["signal"], style=sig_c),
            f"{r['entry']:,.2f}",
            Text(f"{r['target']:,.2f}", style=tgt_c),
            Text(f"{r['stop_loss']:,.2f}", style=sl_c),
            f"{r['score']}/13",
            str(r["rsi"]),
            str(r["adx"]),
            r["candle"],
            Text(r["sentiment"], style=news_c),
            Text(r["reasons"], style="dim"),
        )
    console.print(table)


# ─────────────────────────────────────────────
# MARKET HOURS
# ─────────────────────────────────────────────
def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t <= MARKET_CLOSE


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
def main():
    console.rule("[bold cyan]>> Swing Trading Scanner - Auto Mode <<[/bold cyan]")
    console.print(
        "[dim]Nifty 100 | 15-min candles | RSI . MACD . EMA . BB . ADX . S/R . Candles . Volume . News[/dim]\n"
    )

    alerted_today:  dict = {}
    last_alert_date = date.today()

    while True:
        today = date.today()
        if today != last_alert_date:
            alerted_today   = {}
            last_alert_date = today

        if not is_market_open():
            now = datetime.now(IST).strftime("%d %b %Y %H:%M IST")
            console.print(f"[dim]{now} — Market closed. Checking again in 5 mins...[/dim]", end="\r")
            time.sleep(300)
            continue

        scan_time   = datetime.now(IST).strftime("%d %b %Y %H:%M IST")
        console.print(f"\n[bold]* Scan started at {scan_time}[/bold]")

        nifty_trend = get_nifty_trend()
        console.print(f"[dim]Nifty 50 trend: {nifty_trend.upper()}[/dim]")

        results = []
        total   = len(NIFTY100_SYMBOLS)

        for i, symbol in enumerate(NIFTY100_SYMBOLS, 1):
            console.print(f"[dim]  Analyzing [{i}/{total}] {symbol}...[/dim]", end="\r")
            data = analyze_stock(symbol, nifty_trend)

            if data:
                key = f"{symbol}_{data['signal']}"
                if key not in alerted_today:
                    alerted_today[key] = scan_time
                    results.append(data)
                    console.print(
                        f"[green]  >> {symbol} {data['signal']} | Score:{data['score']}/13[/green]"
                    )

        console.print(" " * 70, end="\r")
        print_results(results, scan_time)
        console.print(f"[dim]Next scan in {SCAN_INTERVAL_SECONDS // 60} minutes...[/dim]")
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()