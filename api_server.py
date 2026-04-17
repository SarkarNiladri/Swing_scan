# -*- coding: utf-8 -*-
"""
SwingScan API Server
=====================
FastAPI backend that exposes your swing trading scanner as REST + SSE endpoints.

Endpoints:
  GET  /status          — market open/closed, Nifty trend
  GET  /scan/stream     — SSE stream: live scan progress + signals
  GET  /scan/result     — last completed scan results (JSON)
  POST /scan/stop       — abort an in-progress scan

Setup:
  pip install fastapi uvicorn yfinance pandas numpy pytz feedparser

Run:
  uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload

Then open scanner_ui.html in your browser (or serve it via any static server).
CORS is open for localhost dev — tighten it before deploying.
"""

import asyncio
import json
import threading
import warnings
import os
warnings.filterwarnings("ignore")

from datetime import datetime, date
from typing import AsyncGenerator

import pytz
import yfinance as yf
import httpx
import pandas as pd
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# ── Try importing TA-Lib; fall back to pure-Python indicators ──────────────
try:
    import talib
    TALIB = True
except ImportError:
    TALIB = False

# ── Try importing feedparser for news sentiment ────────────────────────────
try:
    import feedparser
    FEEDPARSER = True
except ImportError:
    FEEDPARSER = False


# ── Telegram Notification Config ──────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN',   '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
TELEGRAM_API_URL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'


def send_telegram(signal: dict):
    """Send a signal notification to Telegram. Runs in a background thread."""
    try:
        icon    = '🟢' if signal['signal'] == 'BUY' else '🔴'
        entry   = signal['entry']
        target  = signal['target']
        sl      = signal['stop_loss']
        pct_tgt = round((target - entry) / entry * 100, 2)
        pct_sl  = round((sl - entry) / entry * 100, 2)

        lines = [
            f"{icon} *{signal['signal']} SIGNAL — {signal['symbol']}*",
            "",
            f"Entry  : Rs {entry:,.2f}",
            f"Target : Rs {target:,.2f}  ({pct_tgt:+.2f}%)",
            f"SL     : Rs {sl:,.2f}  ({pct_sl:+.2f}%)",
            f"Score  : {signal['score']}/13",
            f"RSI    : {signal['rsi']}",
            f"ADX    : {signal['adx']}",
            f"Candle : {signal.get('candle', '-')}",
            f"News   : {signal.get('sentiment', '-')}",
            "",
            signal.get('reasons', ''),
        ]
        msg = "\n".join(lines)

        httpx.post(
            TELEGRAM_API_URL,
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'},
            timeout=10,
        )
    except Exception as e:
        print(f'Telegram error: {e}')

# ── Try importing Symbols list ────────────────────────────────────────────
try:
    import Symbols
    NIFTY100_SYMBOLS = Symbols.symbols_100  # adjust attr name if needed
except Exception:
    # Minimal fallback list so the server still starts
    NIFTY100_SYMBOLS = [
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC",
        "SBIN","BHARTIARTL","KOTAKBANK","BAJFINANCE","LT","AXISBANK","ASIANPAINT",
        "MARUTI","WIPRO","TITAN","ULTRACEMCO","SUNPHARMA","NESTLEIND",
    ]

IST            = pytz.timezone("Asia/Kolkata")
MARKET_OPEN    = (9, 15)
MARKET_CLOSE   = (15, 30)

# ── Strategy config (keep in sync with your scanner) ──────────────────────
RSI_PERIOD    = 14
RSI_OVERSOLD  = 40
BB_PERIOD     = 20
BB_STD        = 2
EMA_SHORT     = 20
EMA_LONG      = 50
ADX_PERIOD    = 14
ADX_THRESHOLD = 25
ATR_PERIOD    = 14
SR_ZONE_PCT   = 0.015
RISK_REWARD   = 2.0
MIN_SCORE     = 9
ATR_MULT      = 2.0

# ── Background scheduler config ───────────────────────────────────────────
AUTO_SCAN_INTERVAL = 15 * 60   # seconds between scans

app = FastAPI(title="SwingScan API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared scan state ──────────────────────────────────────────────────────
scan_state = {
    "running":      False,
    "stop_flag":    False,
    "results":      [],
    "scanned":      0,
    "total":        0,
    "last_scan":    None,
    "notified_today": {},
    "notified_date":  None,
}

# ── Trade tracker — persists in memory, polled by UI ──────────────────────
# Each trade: {id, symbol, signal, entry, target, stop_loss, score, time,
#              outcome: OPEN|WIN|LOSS, resolved_at, pnl_pct}
trade_tracker = {
    "trades":      [],          # list of trade dicts
    "week_start":  None,        # ISO date of current week start (Monday)
}


# ════════════════════════════════════════════════════════════════════════════
# PURE-PYTHON INDICATOR FALLBACKS  (used when TA-Lib is not installed)
# ════════════════════════════════════════════════════════════════════════════

def _rsi(series: pd.Series, period=14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _macd(series: pd.Series, fast=12, slow=26, sig=9):
    m  = _ema(series, fast) - _ema(series, slow)
    sl = _ema(m, sig)
    return m, sl, m - sl

def _bbands(series: pd.Series, period=20, std=2):
    sma  = series.rolling(period).mean()
    sd   = series.rolling(period).std()
    return sma + std * sd, sma, sma - std * sd

def _atr(df: pd.DataFrame, period=14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _adx(df: pd.DataFrame, period=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    pdm = h.diff().clip(lower=0); ndm = (-l.diff()).clip(lower=0)
    pdm[pdm < ndm] = 0; ndm[ndm < pdm] = 0
    tr  = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    pdi = 100 * pdm.rolling(period).mean() / atr.replace(0, np.nan)
    ndi = 100 * ndm.rolling(period).mean() / atr.replace(0, np.nan)
    dx  = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.rolling(period).mean(), pdi, ndi


# ════════════════════════════════════════════════════════════════════════════
# INDICATOR HELPERS  (TA-Lib if available, else pure-Python)
# ════════════════════════════════════════════════════════════════════════════

def calc_rsi(close_arr, close_s, period):
    if TALIB:
        return pd.Series(talib.RSI(close_arr, timeperiod=period), index=close_s.index)
    return _rsi(close_s, period)

def calc_ema(close_arr, close_s, period):
    if TALIB:
        return pd.Series(talib.EMA(close_arr, timeperiod=period), index=close_s.index)
    return _ema(close_s, period)

def calc_macd(close_arr, close_s):
    if TALIB:
        m, sl, h = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
        return (pd.Series(m, index=close_s.index),
                pd.Series(sl, index=close_s.index),
                pd.Series(h, index=close_s.index))
    return _macd(close_s)

def calc_bbands(close_arr, close_s, period, std):
    if TALIB:
        u, m, l = talib.BBANDS(close_arr, timeperiod=period, nbdevup=std, nbdevdn=std)
        return (pd.Series(u, index=close_s.index),
                pd.Series(m, index=close_s.index),
                pd.Series(l, index=close_s.index))
    return _bbands(close_s, period, std)

def calc_atr(df):
    if TALIB:
        return pd.Series(
            talib.ATR(df["High"].values.astype(float),
                      df["Low"].values.astype(float),
                      df["Close"].values.astype(float),
                      timeperiod=ATR_PERIOD),
            index=df.index)
    return _atr(df, ATR_PERIOD)

def calc_adx(df):
    if TALIB:
        h = df["High"].values.astype(float)
        l = df["Low"].values.astype(float)
        c = df["Close"].values.astype(float)
        adx = talib.ADX(h, l, c, timeperiod=ADX_PERIOD)
        return pd.Series(adx, index=df.index), None, None
    adx, pdi, ndi = _adx(df, ADX_PERIOD)
    return adx, pdi, ndi


# ════════════════════════════════════════════════════════════════════════════
# SUPPORT / RESISTANCE
# ════════════════════════════════════════════════════════════════════════════

def find_sr_levels(df: pd.DataFrame, close: float):
    levels = []
    highs  = df["High"].rolling(5, center=True).max()
    lows   = df["Low"].rolling(5, center=True).min()
    levels.extend(df["High"][df["High"] == highs].tail(50).tolist())
    levels.extend(df["Low"][df["Low"] == lows].tail(50).tolist())
    levels.append(float(df["High"].tail(252 * 26).max()))
    levels.append(float(df["Low"].tail(252 * 26).min()))
    for base in [50, 100]:
        for offset in [-1, 0, 1]:
            levels.append((round(close / base) + offset) * base)
    zone         = close * SR_ZONE_PCT
    near_support = any(abs(close - l) <= zone and l <= close for l in levels if l > 0)
    near_resist  = any(abs(close - l) <= zone and l >= close for l in levels if l > 0)
    return near_support, near_resist


# ════════════════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERNS
# ════════════════════════════════════════════════════════════════════════════

def detect_candle_pattern(df: pd.DataFrame):
    if len(df) < 3:
        return False, False, ""
    c1 = df.iloc[-3]; c2 = df.iloc[-2]; c3 = df.iloc[-1]
    o3,h3,l3,c3c = float(c3["Open"]),float(c3["High"]),float(c3["Low"]),float(c3["Close"])
    o2,h2,l2,c2c = float(c2["Open"]),float(c2["High"]),float(c2["Low"]),float(c2["Close"])
    o1,h1,l1,c1c = float(c1["Open"]),float(c1["High"]),float(c1["Low"]),float(c1["Close"])
    body3       = abs(c3c - o3)
    range3      = h3 - l3 if h3 != l3 else 0.001
    upper_wick3 = h3 - max(c3c, o3)
    lower_wick3 = min(c3c, o3) - l3
    hammer       = lower_wick3 >= 2*body3 and upper_wick3 <= 0.1*range3 and body3 > 0
    bull_engulf  = c2c < o2 and c3c > o3 and o3 <= c2c and c3c >= o2
    morning_star = (c1c < o1 and
                    abs(c2c-o2) < 0.3*(h2-l2 if h2!=l2 else 0.001) and
                    c3c > o3 and c3c > (o1+c1c)/2)
    shooting_star = upper_wick3 >= 2*body3 and lower_wick3 <= 0.1*range3 and body3 > 0
    bear_engulf   = c2c > o2 and c3c < o3 and o3 >= c2c and c3c <= o2
    evening_star  = (c1c > o1 and
                     abs(c2c-o2) < 0.3*(h2-l2 if h2!=l2 else 0.001) and
                     c3c < o3 and c3c < (o1+c1c)/2)
    bullish = hammer or bull_engulf or morning_star
    bearish = shooting_star or bear_engulf or evening_star
    if hammer:          name = "Hammer"
    elif bull_engulf:   name = "Bullish Engulfing"
    elif morning_star:  name = "Morning Star"
    elif shooting_star: name = "Shooting Star"
    elif bear_engulf:   name = "Bearish Engulfing"
    elif evening_star:  name = "Evening Star"
    else:               name = "—"
    return bullish, bearish, name


# ════════════════════════════════════════════════════════════════════════════
# NEWS SENTIMENT
# ════════════════════════════════════════════════════════════════════════════

def get_news_sentiment(symbol: str):
    if not FEEDPARSER:
        return 0.0, "Neutral"
    positive = ["surge","rally","gain","profit","growth","beat","upgrade","outperform",
                "buy","strong","record","boost","expansion","deal","order","positive",
                "rise","bullish","momentum","breakout","revenue","acquisition"]
    negative = ["fall","drop","loss","decline","downgrade","miss","weak","sell","cut",
                "reduce","concern","risk","fraud","penalty","fine","negative","bearish",
                "crash","debt","warning","lawsuit","probe","slump","plunge"]
    try:
        url  = f"https://news.google.com/rss/search?q={symbol}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        entries = feed.entries[:10]
        if not entries:
            return 0.0, "Neutral"
        score = 0
        for entry in entries:
            text = (entry.get("title","") + " " + entry.get("summary","")).lower()
            for w in positive:
                if w in text: score += 1
            for w in negative:
                if w in text: score -= 1
        normalized = score / max(len(entries), 1)
        if normalized > 0.3:    label = "Positive"
        elif normalized < -0.3: label = "Negative"
        else:                   label = "Neutral"
        return round(normalized, 2), label
    except Exception:
        return 0.0, "Neutral"


# ════════════════════════════════════════════════════════════════════════════
# NIFTY 50 TREND
# ════════════════════════════════════════════════════════════════════════════

def get_nifty_trend() -> str:
    try:
        df = yf.download("^NSEI", period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        close_s   = df["Close"].astype(float)
        close_arr = close_s.values
        df["EMA20"] = calc_ema(close_arr, close_s, 20)
        df["EMA50"] = calc_ema(close_arr, close_s, 50)
        df.dropna(inplace=True)
        last  = df.iloc[-1]
        e20, e50, c = float(last["EMA20"]), float(last["EMA50"]), float(last["Close"])
        if c > e20 > e50:   return "bullish"
        elif c < e20 < e50: return "bearish"
        else:               return "neutral"
    except Exception:
        return "neutral"


# ════════════════════════════════════════════════════════════════════════════
# PER-STOCK DAILY TREND
# ════════════════════════════════════════════════════════════════════════════

def get_stock_trend(symbol: str) -> str:
    try:
        df = yf.download(symbol + ".NS", period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        close_s   = df["Close"].astype(float)
        close_arr = close_s.values
        df["EMA20"] = calc_ema(close_arr, close_s, 20)
        df["EMA50"] = calc_ema(close_arr, close_s, 50)
        df.dropna(inplace=True)
        last  = df.iloc[-1]
        e20, e50, c = float(last["EMA20"]), float(last["EMA50"]), float(last["Close"])
        if c > e20 > e50:   return "bullish"
        elif c < e20 < e50: return "bearish"
        else:               return "neutral"
    except Exception:
        return "neutral"


# ════════════════════════════════════════════════════════════════════════════
# CORE STOCK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def analyze_stock(symbol: str, nifty_trend: str) -> dict | None:
    """
    Improved signal logic:
    1. Strategy unified to Trend + Pullback (no mean-reversion conflict)
    2. Volume uses tiered scoring (+1/+2/+3) instead of hard filter
    3. Context (market trend, stock trend, sentiment) uses score penalty not hard rejection
    4. BB used for volatility squeeze detection not direct entry
    5. RSI 40-60 continuation zone adds to trend score
    6. Overextension check: avoid trades far from EMA
    """
    try:
        df = yf.download(symbol + ".NS", period="60d", interval="15m",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < EMA_LONG + 20:
            return None
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)

        close_s   = df["Close"].astype(float)
        close_arr = close_s.values

        df["RSI"]       = calc_rsi(close_arr, close_s, RSI_PERIOD)
        _, _, macdh     = calc_macd(close_arr, close_s)
        df["MACDh"]     = macdh
        df["MACDh_prev"]= macdh.shift(1)
        bbu, bb_mid, bbl = calc_bbands(close_arr, close_s, BB_PERIOD, BB_STD)
        df["BB_Upper"]  = bbu
        df["BB_Lower"]  = bbl
        df["BB_Mid"]    = bb_mid
        df["BB_Width"]  = (bbu - bbl) / bb_mid   # volatility squeeze measure
        df["EMA_Short"] = calc_ema(close_arr, close_s, EMA_SHORT)
        df["EMA_Long"]  = calc_ema(close_arr, close_s, EMA_LONG)
        adx_s, _, _     = calc_adx(df)
        df["ADX"]       = adx_s
        df["ATR"]       = calc_atr(df)
        df.dropna(inplace=True)

        if len(df) < 5:
            return None

        last = df.iloc[-1]
        close     = float(last["Close"])
        volume    = float(last["Volume"])
        rsi       = float(last.get("RSI",       50))
        macd_hist = float(last.get("MACDh",      0))
        prev_macd = float(last.get("MACDh_prev", 0))
        bb_lower  = float(last.get("BB_Lower",  close))
        bb_upper  = float(last.get("BB_Upper",  close))
        bb_width  = float(last.get("BB_Width",  0.02))
        ema_short = float(last.get("EMA_Short", close))
        ema_long  = float(last.get("EMA_Long",  close))
        adx       = float(last.get("ADX",           0))
        atr       = float(last.get("ATR", close*0.01))

        avg_vol   = df["Volume"].tail(20*26).mean()
        vol_ratio = volume / avg_vol if avg_vol > 0 else 0

        buy_score = 0; sell_score = 0; reasons = []

        # ════════════════════════════════════════════════════
        # CORE SIGNALS (max ~9 points) — must all align
        # ════════════════════════════════════════════════════

        # ── CORE 1: TREND via EMA + ADX (max 3 pts) ──────────────────────
        # EMA is the primary trend filter — highest weight
        if ema_short > ema_long:
            buy_score += 3; reasons.append("EMA bullish")
        else:
            sell_score += 3; reasons.append("EMA bearish")

        # ADX must confirm trend is actually moving
        if adx > ADX_THRESHOLD:
            if ema_short > ema_long: buy_score  += 2
            else:                    sell_score += 2
            reasons.append(f"ADX({adx:.0f})")
        else:
            # Weak trend — penalise
            buy_score -= 1; sell_score -= 1

        # ── CORE 2: PULLBACK via RSI (max 2 pts) ─────────────────────────
        # Only reward RSI if it aligns with the trend direction
        if ema_short > ema_long:
            if 35 <= rsi <= 55:
                buy_score += 2; reasons.append(f"RSI({rsi:.0f}) pullback")
            elif rsi < 35:
                buy_score += 1; reasons.append(f"RSI({rsi:.0f}) oversold")
            elif rsi > 70:
                buy_score -= 1   # overbought in uptrend — bad entry
        else:
            if 45 <= rsi <= 65:
                sell_score += 2; reasons.append(f"RSI({rsi:.0f}) pullback")
            elif rsi > 65:
                sell_score += 1; reasons.append(f"RSI({rsi:.0f}) overbought")
            elif rsi < 30:
                sell_score -= 1  # oversold in downtrend — bad short entry

        # ── CORE 3: MACD CONFIRMATION (max 2 pts) ────────────────────────
        # Only fresh crossovers get full +2; existing momentum gets +1
        if macd_hist > 0 and prev_macd <= 0:
            buy_score += 2; reasons.append("MACD ↑ cross")
        elif macd_hist < 0 and prev_macd >= 0:
            sell_score += 2; reasons.append("MACD ↓ cross")
        elif macd_hist > 0:  buy_score  += 1
        elif macd_hist < 0:  sell_score += 1

        # ── CORE 4: VOLUME (max 2 pts, tiered) ───────────────────────────
        if vol_ratio >= 2.0:
            buy_score += 2; sell_score += 2; reasons.append(f"Vol {vol_ratio:.1f}x surge")
        elif vol_ratio >= 1.5:
            buy_score += 1; sell_score += 1; reasons.append(f"Vol {vol_ratio:.1f}x")
        elif vol_ratio < 1.0:
            buy_score -= 1; sell_score -= 1  # below avg volume — penalty only

        # ════════════════════════════════════════════════════
        # OPTIONAL SIGNALS (max +2 combined) — supporting evidence
        # ════════════════════════════════════════════════════
        optional_score = 0

        # S/R proximity (+1 only, reduced from +2)
        near_sup, near_res = find_sr_levels(df, close)
        if near_sup and ema_short > ema_long:
            optional_score += 1; reasons.append("Near support")
        if near_res and ema_short <= ema_long:
            optional_score += 1; reasons.append("Near resistance")

        # Candle pattern (+1 only, reduced from +2)
        bull_c, bear_c, c_name = detect_candle_pattern(df)
        if bull_c and ema_short > ema_long:
            optional_score += 1; reasons.append(c_name)
        elif bear_c and ema_short <= ema_long:
            optional_score += 1; reasons.append(c_name)

        # BB squeeze bonus (+1 max — signals potential breakout)
        avg_bb_width = df["BB_Width"].tail(50).mean()
        if bb_width < avg_bb_width * 0.75:
            optional_score += 1; reasons.append("BB squeeze")

        # Cap optional contribution at +2 to prevent overfitting
        optional_score = min(optional_score, 2)
        buy_score  += optional_score
        sell_score += optional_score

        # ── OVEREXTENSION CHECK ───────────────────────────────────────────
        ema_dist_pct = abs(close - ema_short) / ema_short * 100
        if ema_dist_pct > 3.0:
            buy_score -= 1; sell_score -= 1
            reasons.append(f"Extended {ema_dist_pct:.1f}% from EMA")

        # ── Determine raw signal ───────────────────────────────────────────
        if buy_score >= MIN_SCORE:
            signal, score = "BUY", buy_score
        elif sell_score >= MIN_SCORE:
            signal, score = "SELL", sell_score
        else:
            return None

        # ── 9. CONTEXT — score penalty instead of hard rejection ──────────
        penalty_reasons = []

        # Market trend mismatch → -2 penalty
        if signal == "BUY"  and nifty_trend == "bearish":
            score -= 2; penalty_reasons.append("Nifty bearish(-2)")
        elif signal == "SELL" and nifty_trend == "bullish":
            score -= 2; penalty_reasons.append("Nifty bullish(-2)")

        # Stock daily trend mismatch → -2 penalty
        stock_trend = get_stock_trend(symbol)
        if signal == "BUY"  and stock_trend == "bearish":
            score -= 2; penalty_reasons.append("Stock trend bearish(-2)")
        elif signal == "SELL" and stock_trend == "bullish":
            score -= 2; penalty_reasons.append("Stock trend bullish(-2)")

        # News sentiment → -1 penalty (softer than trend)
        _, sentiment_label = get_news_sentiment(symbol)
        if signal == "BUY"  and sentiment_label == "Negative":
            score -= 1; penalty_reasons.append("News negative(-1)")
        elif signal == "SELL" and sentiment_label == "Positive":
            score -= 1; penalty_reasons.append("News positive(-1)")

        if penalty_reasons:
            reasons.append("Penalties: " + ", ".join(penalty_reasons))

        # After penalties, re-check score threshold
        if score < MIN_SCORE:
            return None

        # ── 10. STOP LOSS & TARGET ────────────────────────────────────────
        if signal == "BUY":
            stop_loss = round(close - ATR_MULT * atr, 2)
            target    = round(close + (close - stop_loss) * RISK_REWARD, 2)
        else:
            stop_loss = round(close + ATR_MULT * atr, 2)
            target    = round(close - (stop_loss - close) * RISK_REWARD, 2)

        # Final RR sanity check
        risk   = abs(close - stop_loss)
        reward = abs(target - close)
        if risk == 0 or reward / risk < RISK_REWARD * 0.9:
            return None

        return {
            "symbol":     symbol,
            "signal":     signal,
            "entry":      round(close, 2),
            "target":     target,
            "stop_loss":  stop_loss,
            "score":      score,
            "rsi":        round(rsi, 1),
            "adx":        round(adx, 1),
            "vol_ratio":  round(vol_ratio, 1),
            "sentiment":  sentiment_label,
            "candle":     c_name or "—",
            "reasons":    " | ".join(reasons),
            "time":       datetime.now(IST).strftime("%H:%M:%S IST"),
        }
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# MARKET STATUS HELPER
# ════════════════════════════════════════════════════════════════════════════

def market_is_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return MARKET_OPEN <= t <= MARKET_CLOSE




# ════════════════════════════════════════════════════════════════════════════
# TRADE TRACKER HELPERS
# ════════════════════════════════════════════════════════════════════════════

def get_week_start() -> str:
    """Returns the ISO date string of this week's Monday in IST."""
    today = datetime.now(IST).date()
    monday = today - pd.Timedelta(days=today.weekday())
    return monday.isoformat()


def reset_tracker_if_new_week():
    """Clears all trades at the start of each new week (Monday IST)."""
    week_start = get_week_start()
    if trade_tracker["week_start"] != week_start:
        trade_tracker["trades"]     = []
        trade_tracker["week_start"] = week_start
        print(f"[tracker] Weekly reset — new week started {week_start}")


def add_trade(signal: dict):
    """Add a new OPEN trade to the tracker. Ignores duplicates."""
    reset_tracker_if_new_week()
    trade_id = (
        f"{signal['symbol']}_{signal['signal']}_"
        f"{signal['entry']}_{signal['target']}_{signal['stop_loss']}"
    )
    # Skip if already tracked
    if any(t["id"] == trade_id for t in trade_tracker["trades"]):
        return
    trade_tracker["trades"].append({
        "id":        trade_id,
        "symbol":    signal["symbol"],
        "signal":    signal["signal"],
        "entry":     signal["entry"],
        "target":    signal["target"],
        "stop_loss": signal["stop_loss"],
        "score":     signal["score"],
        "time":      signal.get("time", ""),
        "outcome":   "OPEN",
        "resolved_at": None,
        "pnl_pct":   None,
    })
    print(f"[tracker] Added {signal['signal']} trade: {signal['symbol']}")


def check_open_trades():
    """
    For every OPEN trade, fetch latest candles and check if
    target or SL has been hit. Updates outcome in-place.
    Sends Telegram notification when resolved.
    """
    reset_tracker_if_new_week()
    open_trades = [t for t in trade_tracker["trades"] if t["outcome"] == "OPEN"]
    if not open_trades:
        return

    print(f"[tracker] Checking {len(open_trades)} open trade(s)...")

    for trade in open_trades:
        try:
            ticker = trade["symbol"] + ".NS"
            # Fetch last 2 days of 15-min candles
            df = yf.download(ticker, period="2d", interval="15m",
                             progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                continue
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.dropna(inplace=True)

            entry     = float(trade["entry"])
            target    = float(trade["target"])
            stop_loss = float(trade["stop_loss"])
            signal    = trade["signal"]

            # Only look at candles after the trade was added
            # (use last 26 candles ≈ 1 trading day as window)
            recent = df.tail(26)

            outcome = None
            for _, row in recent.iterrows():
                high  = float(row["High"])
                low   = float(row["Low"])
                close = float(row["Close"])

                if signal == "BUY":
                    if low <= stop_loss:
                        outcome = "LOSS"
                        pnl_pct = round((stop_loss - entry) / entry * 100, 2)
                        break
                    if high >= target:
                        outcome = "WIN"
                        pnl_pct = round((target - entry) / entry * 100, 2)
                        break
                else:  # SELL
                    if high >= stop_loss:
                        outcome = "LOSS"
                        pnl_pct = round((entry - stop_loss) / entry * 100, 2)
                        break
                    if low <= target:
                        outcome = "WIN"
                        pnl_pct = round((entry - target) / entry * 100, 2)
                        break

            if outcome:
                trade["outcome"]     = outcome
                trade["pnl_pct"]     = pnl_pct
                trade["resolved_at"] = datetime.now(IST).strftime("%H:%M:%S IST")
                print(f"[tracker] {trade['symbol']} → {outcome} ({pnl_pct:+.2f}%)")

                # Telegram notification for outcome
                icon = "✅" if outcome == "WIN" else "❌"
                lines = [
                    f"{icon} *{outcome} — {trade['symbol']} {signal}*",
                    "",
                    f"Entry  : Rs {entry:,.2f}",
                    f"Target : Rs {target:,.2f}",
                    f"SL     : Rs {stop_loss:,.2f}",
                    f"P&L    : {pnl_pct:+.2f}%",
                    f"Time   : {trade['resolved_at']}",
                ]
                msg = "\n".join(lines)
                try:
                    httpx.post(
                        TELEGRAM_API_URL,
                        json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"},
                        timeout=10,
                    )
                except Exception as e:
                    print(f"[tracker] Telegram error: {e}")

        except Exception as e:
            print(f"[tracker] Error checking {trade['symbol']}: {e}")

# ════════════════════════════════════════════════════════════════════════════
# BACKGROUND SCHEDULER — runs scan every 15 min during market hours
# Independent of browser/UI — fires Telegram alerts automatically
# ════════════════════════════════════════════════════════════════════════════

def run_background_scan():
    """
    Runs a full Nifty 100 scan in a background thread.
    Sends Telegram notifications for new signals.
    Respects the duplicate fingerprint filter.
    """
    if scan_state["running"]:
        print("[scheduler] Scan already running — skipping this cycle")
        return

    print(f"[scheduler] Starting background scan at {datetime.now(IST).strftime('%H:%M:%S IST')}")
    scan_state["running"]   = True
    scan_state["stop_flag"] = False
    scan_state["results"]   = []
    scan_state["scanned"]   = 0
    scan_state["total"]     = len(NIFTY100_SYMBOLS)

    try:
        nifty_trend  = get_nifty_trend()
        signal_count = 0

        for idx, symbol in enumerate(NIFTY100_SYMBOLS, 1):
            if scan_state["stop_flag"]:
                break

            scan_state["scanned"] = idx
            result = analyze_stock(symbol, nifty_trend)

            if result:
                signal_count += 1
                scan_state["results"].append(result)

                # Duplicate notification check
                today = datetime.now(IST).date().isoformat()
                if scan_state["notified_date"] != today:
                    scan_state["notified_today"] = {}
                    scan_state["notified_date"]  = today

                fingerprint = (
                    f"{result['symbol']}|{result['signal']}|"
                    f"{result['entry']}|{result['target']}|{result['stop_loss']}"
                )
                if fingerprint not in scan_state["notified_today"]:
                    scan_state["notified_today"][fingerprint] = today
                    send_telegram(result)
                    add_trade(result)   # register in trade tracker
                    print(f"[scheduler] {result['signal']} signal: {result['symbol']} score={result['score']}")

        scan_state["last_scan"] = datetime.now(IST).isoformat()
        print(f"[scheduler] Scan complete — {signal_count} signal(s) found")

    except Exception as e:
        print(f"[scheduler] Error during scan: {e}")
    finally:
        scan_state["running"] = False


def scheduler_loop():
    """
    Infinite loop that runs in a daemon thread.
    Every minute:
      - Checks if it is time to run a full scan (every 15 min, market open)
      - Checks all open trades for target/SL hits (every minute, market open)
      - Resets tracker on Monday
    """
    last_scan_time = 0

    while True:
        try:
            import time as _time
            now_ts = datetime.now(IST).timestamp()
            reset_tracker_if_new_week()

            if market_is_open():
                # Full scan every 15 min
                if (now_ts - last_scan_time) >= AUTO_SCAN_INTERVAL:
                    last_scan_time = now_ts
                    run_background_scan()
                # Check open trades every minute during market hours
                threading.Thread(target=check_open_trades, daemon=True).start()

        except Exception as e:
            print(f"[scheduler] Loop error: {e}")

        import time as _time
        _time.sleep(60)


# Start background scheduler thread on server startup
_scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True, name="SwingScan-Scheduler")
_scheduler_thread.start()
print("[scheduler] Background scheduler started — will scan every 15 min during market hours")


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.get("/status")
async def status():
    """Quick health check + market state + Nifty trend."""
    ist_now = datetime.now(IST).strftime("%d %b %Y %H:%M:%S IST")
    loop = asyncio.get_event_loop()
    trend = await loop.run_in_executor(None, get_nifty_trend)
    return {
        "market_open": market_is_open(),
        "ist_time":    ist_now,
        "nifty_trend": trend,
        "talib":       TALIB,
        "symbols":     len(NIFTY100_SYMBOLS),
        "scan_running": scan_state["running"],
        "min_score":   MIN_SCORE,
        "risk_reward": RISK_REWARD,
    }


@app.get("/scan/stream")
async def scan_stream(symbols: str = ""):
    """
    SSE endpoint — streams scan progress and signals in real time.
    Query param: ?symbols=RELIANCE,TCS   (blank = full Nifty 100)
    
    Event types sent:
      data: {"type":"start",  "total": N, "nifty_trend": "bullish"}
      data: {"type":"progress","symbol":"X","index":i,"total":N}
      data: {"type":"signal",  ...signal fields...}
      data: {"type":"done",   "total_signals":N,"scanned":N}
      data: {"type":"error",  "msg":"..."}
      data: {"type":"stopped"}
    """
    symbol_list = (
        [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if symbols else NIFTY100_SYMBOLS
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        if scan_state["running"]:
            yield f"data: {json.dumps({'type':'error','msg':'Scan already in progress'})}\n\n"
            return

        scan_state["running"]   = True
        scan_state["stop_flag"] = False
        scan_state["results"]   = []
        scan_state["scanned"]   = 0
        scan_state["total"]     = len(symbol_list)

        loop = asyncio.get_event_loop()

        try:
            # Get Nifty trend first
            nifty_trend = await loop.run_in_executor(None, get_nifty_trend)
            yield f"data: {json.dumps({'type':'start','total':len(symbol_list),'nifty_trend':nifty_trend})}\n\n"

            signal_count = 0
            for idx, symbol in enumerate(symbol_list, 1):
                if scan_state["stop_flag"]:
                    yield f"data: {json.dumps({'type':'stopped'})}\n\n"
                    return

                # Progress event
                yield f"data: {json.dumps({'type':'progress','symbol':symbol,'index':idx,'total':len(symbol_list)})}\n\n"
                await asyncio.sleep(0)   # yield control so the event is flushed

                # Run blocking IO in thread pool
                result = await loop.run_in_executor(None, analyze_stock, symbol, nifty_trend)

                scan_state["scanned"] = idx

                if result:
                    signal_count += 1
                    scan_state["results"].append(result)
                    payload = dict(result)
                    payload["type"] = "signal"
                    yield f"data: {json.dumps(payload)}\n\n"
                    # Duplicate check — build a fingerprint from key fields
                    today = datetime.now(IST).date().isoformat()
                    if scan_state["notified_date"] != today:
                        scan_state["notified_today"] = {}
                        scan_state["notified_date"]  = today
                    fingerprint = (
                        f"{result['symbol']}|{result['signal']}|"
                        f"{result['entry']}|{result['target']}|{result['stop_loss']}"
                    )
                    if fingerprint not in scan_state["notified_today"]:
                        scan_state["notified_today"][fingerprint] = today
                        threading.Thread(target=send_telegram, args=(result,), daemon=True).start()
                        add_trade(result)   # register in trade tracker

            scan_state["last_scan"] = datetime.now(IST).isoformat()
            yield f"data: {json.dumps({'type':'done','total_signals':signal_count,'scanned':len(symbol_list)})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg':str(e)})}\n\n"
        finally:
            scan_state["running"] = False

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/scan/result")
async def scan_result():
    """Returns the results of the most recently completed scan."""
    return JSONResponse({
        "results":   scan_state["results"],
        "scanned":   scan_state["scanned"],
        "total":     scan_state["total"],
        "last_scan": scan_state["last_scan"],
        "running":   scan_state["running"],
    })


@app.post("/scan/stop")
async def scan_stop():
    """Signals the running scan to stop after the current stock."""
    if not scan_state["running"]:
        return {"stopped": False, "msg": "No scan running"}
    scan_state["stop_flag"] = True
    return {"stopped": True, "msg": "Stop signal sent"}


@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    return {"ping": "pong", "time": datetime.now(IST).isoformat()}


@app.get("/tracker/trades")
async def tracker_trades():
    """Returns all trades for the current week with their outcomes."""
    reset_tracker_if_new_week()
    return JSONResponse({
        "trades":     trade_tracker["trades"],
        "week_start": trade_tracker["week_start"],
        "total":      len(trade_tracker["trades"]),
        "wins":       sum(1 for t in trade_tracker["trades"] if t["outcome"] == "WIN"),
        "losses":     sum(1 for t in trade_tracker["trades"] if t["outcome"] == "LOSS"),
        "open":       sum(1 for t in trade_tracker["trades"] if t["outcome"] == "OPEN"),
    })


@app.post("/tracker/reset")
async def tracker_reset():
    """Manually reset the trade tracker."""
    trade_tracker["trades"]     = []
    trade_tracker["week_start"] = get_week_start()
    return {"reset": True, "week_start": trade_tracker["week_start"]}


@app.get("/")
async def serve_ui():
    html_path = Path(__file__).parent / "scanner_ui.html"
    if html_path.exists():
        return FileResponse(html_path, media_type="text/html")
    return JSONResponse({"msg": "scanner_ui.html not found"}, status_code=404)


# ENTRY POINT
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port)