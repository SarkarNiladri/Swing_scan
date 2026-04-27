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

# ── Try importing KiteConnect ──────────────────────────────────────────────
try:
    from kiteconnect import KiteConnect
    KITE_AVAILABLE = True
except ImportError:
    KITE_AVAILABLE = False
    print("[kite] kiteconnect not installed — pip install kiteconnect")

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, HTMLResponse
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

# ── Zerodha Kite Connect Config ────────────────────────────────────────────
KITE_API_KEY    = os.environ.get('KITE_API_KEY',    '')
KITE_API_SECRET = os.environ.get('KITE_API_SECRET', '')

# Kite session state — access token valid for one trading day
kite_state = {
    "access_token": None,
    "token_date":   None,   # date when token was generated
    "kite":         None,   # KiteConnect instance
    "login_url":    None,   # one-time login URL shown to user
}


def kite_get() -> "KiteConnect | None":
    """
    Returns an authenticated KiteConnect instance if token is valid today.
    Returns None if not logged in yet.
    """
    if not KITE_AVAILABLE:
        return None
    today = datetime.now(IST).date().isoformat()
    if kite_state["access_token"] and kite_state["token_date"] == today:
        return kite_state["kite"]
    return None


def kite_set_token(access_token: str):
    """Store access token and create KiteConnect session."""
    if not KITE_AVAILABLE:
        return False
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(access_token)
        # Verify token works
        profile = kite.profile()
        kite_state["access_token"] = access_token
        kite_state["token_date"]   = datetime.now(IST).date().isoformat()
        kite_state["kite"]         = kite
        print(f"[kite] Logged in as {profile.get('user_name','?')} — token valid today")
        return True
    except Exception as e:
        print(f"[kite] Token error: {e}")
        return False


def kite_get_login_url() -> str:
    """Generate and cache the Kite login URL."""
    if not KITE_AVAILABLE:
        return ""
    kite = KiteConnect(api_key=KITE_API_KEY)
    url  = kite.login_url()
    kite_state["login_url"] = url
    return url


def kite_get_quote(symbol: str) -> dict | None:
    """
    Fetch live quote for a symbol using Kite.
    Returns dict with last_price, high, low, close or None on failure.
    """
    kite = kite_get()
    if not kite:
        return None
    try:
        instrument = f"NSE:{symbol}"
        quote = kite.quote([instrument])
        data  = quote.get(instrument, {})
        ohlc  = data.get("ohlc", {})
        return {
            "last_price": data.get("last_price", 0),
            "high":       ohlc.get("high", 0),
            "low":        ohlc.get("low",  0),
            "close":      ohlc.get("close",0),
            "volume":     data.get("volume", 0),
        }
    except Exception as e:
        print(f"[kite] Quote error {symbol}: {e}")
        return None


def kite_get_candles(symbol: str, days: int = 2) -> pd.DataFrame | None:
    """
    Fetch 15-min historical candles via Kite (much more accurate than yfinance).
    Falls back to yfinance if Kite not available.
    """
    kite = kite_get()
    if not kite:
        return None
    try:
        from datetime import timedelta
        instrument_token = None
        # Get instrument token for the symbol
        instruments = kite.instruments("NSE")
        for inst in instruments:
            if inst["tradingsymbol"] == symbol and inst["instrument_type"] == "EQ":
                instrument_token = inst["instrument_token"]
                break
        if not instrument_token:
            return None
        to_date   = datetime.now(IST)
        from_date = to_date - timedelta(days=days)
        records   = kite.historical_data(
            instrument_token, from_date, to_date, "15minute"
        )
        if not records:
            return None
        df = pd.DataFrame(records)
        df.rename(columns={"date":"Date","open":"Open","high":"High",
                            "low":"Low","close":"Close","volume":"Volume"}, inplace=True)
        df.set_index("Date", inplace=True)
        return df
    except Exception as e:
        print(f"[kite] Candles error {symbol}: {e}")
        return None


def should_notify_telegram(signal: dict) -> tuple:
    """
    Check if a signal passes all Telegram filter thresholds.
    Returns (True, "") if it passes, or (False, reason) if filtered out.
    Filters are set via Render environment variables:
      TG_MIN_SCORE, TG_MIN_ADX, TG_MIN_RR, TG_NEWS_FILTER, TG_DATA_SOURCE
    """
    score  = signal.get("score", 0)
    adx    = signal.get("adx",   0)
    entry  = float(signal.get("entry",     0))
    target = float(signal.get("target",    0))
    sl     = float(signal.get("stop_loss", 0))
    news   = signal.get("sentiment",   "Neutral")
    source = signal.get("data_source", "")

    # Score filter
    if score < TG_MIN_SCORE:
        return False, f"Score {score} < min {TG_MIN_SCORE}"

    # ADX filter
    if adx < TG_MIN_ADX:
        return False, f"ADX {adx:.1f} < min {TG_MIN_ADX}"

    # R/R filter
    if TG_MIN_RR > 0 and entry > 0:
        risk   = abs(entry - sl)
        reward = abs(target - entry)
        rr     = reward / risk if risk > 0 else 0
        if rr < TG_MIN_RR:
            return False, f"R/R {rr:.1f} < min {TG_MIN_RR}"

    # News filter
    if TG_NEWS_FILTER == "Positive" and news != "Positive":
        return False, f"News {news} not Positive"
    if TG_NEWS_FILTER == "Neutral" and news == "Negative":
        return False, f"News {news} is Negative"

    # Data source filter
    if TG_DATA_SOURCE == "kite" and source != "kite":
        return False, f"Data source {source} not kite"

    return True, ""


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

# ════════════════════════════════════════════════════════════════════════════
# AI SIGNAL SCORER
# ════════════════════════════════════════════════════════════════════════════
_ai_model     = None
_ai_features  = None
_ai_meta      = {}

def load_ai_model():
    """Load the trained ML model if available. Silently skips if not found."""
    global _ai_model, _ai_features, _ai_meta
    import os, pickle, json
    model_path = os.path.join(os.path.dirname(__file__), "signal_model.pkl")
    meta_path  = os.path.join(os.path.dirname(__file__), "signal_model_meta.json")
    try:
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        _ai_model    = payload["model"]
        _ai_features = payload["features"]
        _ai_meta     = {k: v for k, v in payload.items() if k != "model"}
        if meta_path and os.path.exists(meta_path):
            with open(meta_path) as f:
                _ai_meta = json.load(f)
        print(f"[AI] Model loaded — AUC={_ai_meta.get('auc',0):.3f} | "
              f"Trained on {_ai_meta.get('n_trades',0)} trades")
    except FileNotFoundError:
        print("[AI] No model file found — AI scoring disabled")
    except Exception as e:
        print(f"[AI] Model load error: {e}")

def ai_confidence(score: int, adx: float, rsi: float, hour: int,
                  is_sell: int, sl_pct: float, tgt_pct: float) -> float | None:
    """
    Returns AI win probability (0.0-1.0) or None if model not loaded.
    Features must match exactly what train_model.py produces.
    """
    if _ai_model is None:
        return None
    try:
        row = {
            "score":      score,
            "adx":        adx,
            "rsi":        rsi,
            "hour":       hour,
            "is_sell":    is_sell,
            "is_hour_12": int(hour == 12),
            "is_hour_11": int(hour == 11),
            "is_hour_13": int(hour == 13),
            "is_hour_10": int(hour == 10),
            "adx_strong": int(adx >= 40),
            "adx_very":   int(adx >= 55),
            "score_high": int(score >= 11),
            "rsi_zone":   0 if rsi < 35 else (1 if rsi < 50 else (2 if rsi < 65 else 3)),
            "sl_pct":     sl_pct,
            "tgt_pct":    tgt_pct,
        }
        X    = [[row[f] for f in _ai_features]]
        prob = _ai_model.predict_proba(X)[0][1]
        return round(float(prob), 4)
    except Exception:
        return None

load_ai_model()
MARKET_OPEN    = (9, 15)
MARKET_CLOSE   = (15, 30)

# ── Strategy config (keep in sync with your scanner) ──────────────────────
RSI_PERIOD      = 14
RSI_OVERSOLD    = 40
BB_PERIOD       = 20
BB_STD          = 2
EMA_SHORT       = 20
EMA_LONG        = 50
ADX_PERIOD      = 14
ADX_THRESHOLD   = 25
ATR_PERIOD      = 14
DAILY_ATR_PERIOD= 14      # ATR on daily candles for SL calculation
SR_ZONE_PCT     = 0.015
RISK_REWARD     = 2.0
MIN_SCORE            = 9      # Validated by backtest
ATR_MULT             = 1.5    # Applied to blended daily ATR
NIFTY_ADX_GATE       = 20     # Skip scan if Nifty ADX < this
MIN_CANDLE_BODY      = 0.40   # Relaxed — 40% body ratio (validated)
VOL_LOOKBACK         = 3      # Consecutive candles for volume check
COUNTER_TREND_PENALTY= 2      # Extra score needed to trade against Nifty trend
SCORE9_ADX_MIN       = 30     # Score-9 signals need ADX≥30 (higher conviction)

# Time-of-day filter (validated: 11:00-13:30 IST has best win rate)
TRADE_HOUR_START = 11
TRADE_HOUR_END   = 13
TRADE_MIN_END    = 30

# ── Background scheduler config ───────────────────────────────────────────
AUTO_SCAN_INTERVAL = 15 * 60   # seconds between scans

# ── Telegram signal filter config ─────────────────────────────────────────
# Set these in Render environment variables to control which signals
# get sent to Telegram. Changes take effect without redeploying.
TG_MIN_SCORE    = int(os.environ.get("TG_MIN_SCORE",    "9"))
TG_MIN_ADX      = float(os.environ.get("TG_MIN_ADX",    "25"))
TG_MIN_RR       = float(os.environ.get("TG_MIN_RR",     "2.0"))
TG_NEWS_FILTER  = os.environ.get("TG_NEWS_FILTER",  "")    # "" = Any, "Positive", "Neutral"
TG_DATA_SOURCE  = os.environ.get("TG_DATA_SOURCE",  "")    # "" = Any, "kite"

app = FastAPI(title="SwingScan API", version="1.0")

# ── CORS — only allow requests from your own Render domain ─────────────────
ALLOWED_ORIGINS = [
    "https://swing-scan.onrender.com",
    "http://localhost:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "null",   # file:// origin when opening HTML locally
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── API Password Auth ──────────────────────────────────────────────────────
# Set API_PASSWORD in Render environment variables.
# If not set, auth is disabled (for local dev convenience).
API_PASSWORD = os.environ.get("API_PASSWORD", "")


def verify_auth(request: Request):
    """
    Dependency — checks X-API-Key header or ?api_key= query param.
    Skips check if API_PASSWORD is not configured (local dev mode).
    Public routes (/, /ping, /kite/login, /kite/callback) skip this check.
    """
    if not API_PASSWORD:
        return True   # auth disabled locally
    # Accept from header or query param
    key = (request.headers.get("X-API-Key", "")
           or request.query_params.get("api_key", ""))
    if key != API_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized — invalid API key")
    return True

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

# ── Trade tracker — PostgreSQL backed, survives deploys & restarts ──────────
# In-memory cache is populated from DB on startup and kept in sync.
trade_tracker = {
    "trades":      [],
    "week_start":  None,
}


def get_week_start() -> str:
    """Returns the ISO date string of this week's Monday in IST."""
    today = datetime.now(IST).date()
    monday = today - pd.Timedelta(days=today.weekday())
    return monday.isoformat()


# ── PostgreSQL connection ─────────────────────────────────────────────────
import psycopg2, psycopg2.extras

_DB_URL = os.environ.get("DATABASE_URL", "")

def _db_conn():
    """Return a psycopg2 connection. Raises if DATABASE_URL not set."""
    if not _DB_URL:
        raise RuntimeError("DATABASE_URL env var not set")
    return psycopg2.connect(_DB_URL, sslmode="require")

def init_db():
    """Create trades table if it doesn't exist."""
    try:
        with _db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id            TEXT PRIMARY KEY,
                        symbol        TEXT,
                        signal        TEXT,
                        entry         REAL,
                        target        REAL,
                        stop_loss     REAL,
                        score         INTEGER,
                        adx           REAL,
                        rsi           REAL,
                        hour          INTEGER,
                        sl_pct        REAL,
                        tgt_pct       REAL,
                        ai_confidence REAL,
                        source        TEXT,
                        time          TEXT,
                        date          TEXT,
                        outcome       TEXT DEFAULT 'OPEN',
                        resolved_at   TEXT,
                        pnl_pct       REAL,
                        exit          TEXT,
                        week_start    TEXT
                    )
                """)
            conn.commit()
        print("[db] Table ready")
    except Exception as e:
        print(f"[db] init_db error: {e}")

def save_trade_db(trade: dict):
    """Upsert a single trade to PostgreSQL."""
    if not _DB_URL:
        return
    try:
        with _db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trades
                        (id, symbol, signal, entry, target, stop_loss,
                         score, adx, rsi, hour, sl_pct, tgt_pct,
                         ai_confidence, source, time, date,
                         outcome, resolved_at, pnl_pct, exit, week_start)
                    VALUES
                        (%(id)s, %(symbol)s, %(signal)s, %(entry)s, %(target)s,
                         %(stop_loss)s, %(score)s, %(adx)s, %(rsi)s, %(hour)s,
                         %(sl_pct)s, %(tgt_pct)s, %(ai_confidence)s, %(source)s,
                         %(time)s, %(date)s, %(outcome)s, %(resolved_at)s,
                         %(pnl_pct)s, %(exit)s, %(week_start)s)
                    ON CONFLICT (id) DO UPDATE SET
                        outcome      = EXCLUDED.outcome,
                        resolved_at  = EXCLUDED.resolved_at,
                        pnl_pct      = EXCLUDED.pnl_pct,
                        exit         = EXCLUDED.exit
                """, {**trade, "week_start": trade_tracker["week_start"]})
            conn.commit()
    except Exception as e:
        print(f"[db] save_trade error: {e}")

def delete_week_db(week_start: str):
    """Delete all trades for a given week (weekly reset)."""
    if not _DB_URL:
        return
    try:
        with _db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM trades WHERE week_start = %s", (week_start,))
            conn.commit()
    except Exception as e:
        print(f"[db] delete_week error: {e}")

def load_tracker():
    """Load current week's trades from PostgreSQL into memory on startup."""
    week_start = get_week_start()
    trade_tracker["week_start"] = week_start

    if not _DB_URL:
        print("[db] DATABASE_URL not set — running without persistence")
        return

    try:
        init_db()
        with _db_conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM trades WHERE week_start = %s ORDER BY time",
                    (week_start,)
                )
                rows = cur.fetchall()
        trade_tracker["trades"] = [dict(r) for r in rows]
        print(f"[db] Loaded {len(rows)} trades for week {week_start}")
    except Exception as e:
        print(f"[db] load_tracker error: {e} — running without persistence")
        print(f"[db] Check DATABASE_URL — use Internal Database URL from Render PostgreSQL dashboard")

def save_tracker():
    """No-op — individual saves handled by save_trade_db(). Kept for compatibility."""
    pass

load_tracker()


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

def get_nifty_trend() -> tuple:
    """
    Get Nifty 50 trend + ADX using Kite data if available, else yfinance.
    Returns (trend_str, adx_value).
    ADX gate: if ADX < NIFTY_ADX_GATE the market is choppy — skip scanning.
    """
    try:
        df = fetch_nifty_daily()
        if df is None or len(df) < 55:
            return "neutral", 0
        close_s   = df["Close"].astype(float)
        close_arr = close_s.values
        df["EMA20"]    = calc_ema(close_arr, close_s, 20)
        df["EMA50"]    = calc_ema(close_arr, close_s, 50)
        adx_s, _, _    = calc_adx(df)
        df["ADX"]      = adx_s
        df.dropna(inplace=True)
        last  = df.iloc[-1]
        e20   = float(last["EMA20"])
        e50   = float(last["EMA50"])
        c     = float(last["Close"])
        adx   = float(last["ADX"])
        if c > e20 > e50:   trend = "bullish"
        elif c < e20 < e50: trend = "bearish"
        else:               trend = "neutral"
        return trend, round(adx, 1)
    except Exception:
        return "neutral", 0


# ════════════════════════════════════════════════════════════════════════════
# PER-STOCK DAILY TREND
# ════════════════════════════════════════════════════════════════════════════

def get_stock_daily_data(symbol: str) -> tuple:
    """
    Fetch daily data for a stock and return:
    (trend_str, daily_atr, daily_rsi, daily_ema20_above_ema50)
    Used for:
      - Higher timeframe trend confirmation
      - Daily ATR for realistic SL calculation
      - Daily RSI to confirm setup quality
    """
    try:
        df = fetch_candles(symbol, interval="day", days=120)
        if df is None or len(df) < 55:
            return "neutral", 0, 50, False
        close_s   = df["Close"].astype(float)
        close_arr = close_s.values
        df["EMA20"]  = calc_ema(close_arr, close_s, 20)
        df["EMA50"]  = calc_ema(close_arr, close_s, 50)
        df["RSI"]    = calc_rsi(close_arr, close_s, 14)
        df["ATR"]    = calc_atr(df, DAILY_ATR_PERIOD)
        df.dropna(inplace=True)
        last  = df.iloc[-1]
        e20   = float(last["EMA20"])
        e50   = float(last["EMA50"])
        c     = float(last["Close"])
        d_atr = float(last["ATR"])
        d_rsi = float(last["RSI"])
        ema_bull = e20 > e50

        if c > e20 > e50:   trend = "bullish"
        elif c < e20 < e50: trend = "bearish"
        else:               trend = "neutral"

        # Higher timeframe confirmation:
        # For BUY: price above daily EMA20 AND daily RSI > 45
        # For SELL: price below daily EMA20 AND daily RSI < 55
        return trend, round(d_atr, 2), round(d_rsi, 1), ema_bull
    except Exception:
        return "neutral", 0, 50, False


def get_stock_trend(symbol: str) -> str:
    """Simple wrapper for backward compatibility."""
    trend, _, _, _ = get_stock_daily_data(symbol)
    return trend


# ════════════════════════════════════════════════════════════════════════════
# CENTRAL DATA FETCHER — Kite first, yfinance fallback
# ════════════════════════════════════════════════════════════════════════════

# Cache instrument tokens to avoid repeated API calls
_instrument_cache: dict = {}


def get_instrument_token(symbol: str) -> int | None:
    """Get NSE instrument token for a symbol, cached in memory."""
    if symbol in _instrument_cache:
        return _instrument_cache[symbol]
    kite = kite_get()
    if not kite:
        return None
    try:
        instruments = kite.instruments("NSE")
        for inst in instruments:
            if inst["tradingsymbol"] == symbol and inst["instrument_type"] == "EQ":
                _instrument_cache[symbol] = inst["instrument_token"]
                return inst["instrument_token"]
    except Exception as e:
        print(f"[kite] Instrument lookup error {symbol}: {e}")
    return None


def fetch_candles(symbol: str, interval: str = "15minute",
                  days: int = 60, source_log: list | None = None) -> pd.DataFrame | None:
    """
    Fetch OHLCV candles for a symbol.
    Priority:
      1. Kite Connect  — real broker data, matches Zerodha charts exactly
      2. yfinance      — fallback when Kite not logged in (delayed/adjusted)

    Args:
        symbol    : NSE symbol e.g. 'RELIANCE'
        interval  : '15minute' | 'day'
        days      : how many days of history to fetch
        source_log: optional list to append data source string
    """
    from datetime import timedelta

    kite = kite_get()

    # ── Try Kite first ─────────────────────────────────────────────────────
    if kite:
        try:
            token = get_instrument_token(symbol)
            if token:
                to_date   = datetime.now(IST)
                from_date = to_date - timedelta(days=days)
                records   = kite.historical_data(token, from_date, to_date, interval)
                if records:
                    df = pd.DataFrame(records)
                    df.rename(columns={
                        "date": "Date", "open": "Open", "high": "High",
                        "low":  "Low",  "close": "Close", "volume": "Volume"
                    }, inplace=True)
                    df.set_index("Date", inplace=True)
                    df.dropna(inplace=True)
                    if len(df) > 10:
                        if source_log is not None:
                            source_log.append("kite")
                        return df
        except Exception as e:
            print(f"[kite] fetch_candles error {symbol}: {e}")

    # ── Fall back to yfinance ──────────────────────────────────────────────
    try:
        yf_interval = "15m" if interval == "15minute" else "1d"
        yf_period   = f"{days}d" if days <= 59 else "3mo"
        df = yf.download(symbol + ".NS", period=yf_period,
                         interval=yf_interval, progress=False, auto_adjust=True)
        if df is not None and len(df) > 10:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df.dropna(inplace=True)
            if source_log is not None:
                source_log.append("yfinance")
            return df
    except Exception as e:
        print(f"[yfinance] fetch_candles error {symbol}: {e}")

    return None


def fetch_nifty_daily() -> pd.DataFrame | None:
    """
    Fetch Nifty 50 daily candles.
    Kite uses instrument token 256265 for NIFTY 50 index.
    Falls back to yfinance ^NSEI.
    """
    from datetime import timedelta
    kite = kite_get()

    if kite:
        try:
            to_date   = datetime.now(IST)
            from_date = to_date - timedelta(days=120)
            # Nifty 50 index token on NSE
            records = kite.historical_data(256265, from_date, to_date, "day")
            if records:
                df = pd.DataFrame(records)
                df.rename(columns={
                    "date": "Date", "open": "Open", "high": "High",
                    "low":  "Low",  "close": "Close", "volume": "Volume"
                }, inplace=True)
                df.set_index("Date", inplace=True)
                df.dropna(inplace=True)
                if len(df) > 10:
                    return df
        except Exception as e:
            print(f"[kite] Nifty fetch error: {e}")

    # yfinance fallback
    try:
        df = yf.download("^NSEI", period="3mo", interval="1d",
                         progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        return df if len(df) > 10 else None
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
# CORE STOCK ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def analyze_stock(symbol: str, nifty_trend: str, manual: bool = False) -> dict | None:
    """
    High-quality signal logic v3:
    1. EMA + ADX both MANDATORY — no signal without both
    2. Daily ATR for SL — prevents wick-based false SL hits
    3. Strong candle body filter — signal candle must be decisive
    4. Consistent volume trend — 3-candle volume check
    5. Higher timeframe daily confirmation
    6. Nifty ADX gate applied upstream (in scanner loop)
    """
    source_log = []
    try:
        # ── Fetch 15-min candles ──────────────────────────────────────────
        df = fetch_candles(symbol, interval="15minute", days=60, source_log=source_log)
        if df is None or len(df) < EMA_LONG + 20:
            return None
        df = df.copy()

        close_s   = df["Close"].astype(float)
        close_arr = close_s.values

        df["RSI"]        = calc_rsi(close_arr, close_s, RSI_PERIOD)
        _, _, macdh      = calc_macd(close_arr, close_s)
        df["MACDh"]      = macdh
        df["MACDh_prev"] = macdh.shift(1)
        bbu, bb_mid, bbl = calc_bbands(close_arr, close_s, BB_PERIOD, BB_STD)
        df["BB_Upper"]   = bbu
        df["BB_Lower"]   = bbl
        df["BB_Mid"]     = bb_mid
        df["BB_Width"]   = (bbu - bbl) / bb_mid
        df["EMA_Short"]  = calc_ema(close_arr, close_s, EMA_SHORT)
        df["EMA_Long"]   = calc_ema(close_arr, close_s, EMA_LONG)
        adx_s, _, _      = calc_adx(df)
        df["ADX"]        = adx_s
        df["ATR"]        = calc_atr(df)
        df.dropna(inplace=True)

        if len(df) < 5:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]

        close     = float(last["Close"])
        open_     = float(last["Open"])
        high_     = float(last["High"])
        low_      = float(last["Low"])
        volume    = float(last["Volume"])
        rsi       = float(last.get("RSI",        50))
        macd_hist = float(last.get("MACDh",       0))
        prev_macd = float(last.get("MACDh_prev",  0))
        bb_lower  = float(last.get("BB_Lower",  close))
        bb_upper  = float(last.get("BB_Upper",  close))
        bb_width  = float(last.get("BB_Width",  0.02))
        ema_short = float(last.get("EMA_Short", close))
        ema_long  = float(last.get("EMA_Long",  close))
        adx       = float(last.get("ADX",           0))
        atr_15m   = float(last.get("ATR", close*0.005))

        avg_vol   = df["Volume"].tail(20*26).mean()
        vol_ratio = volume / avg_vol if avg_vol > 0 else 0

        # ════════════════════════════════════════════════════════════
        # GATE 1 — EMA + ADX MANDATORY
        # Both must be present — no signal without these two
        # ════════════════════════════════════════════════════════════
        ema_bullish = ema_short > ema_long
        ema_bearish = ema_short <= ema_long
        adx_strong  = adx >= ADX_THRESHOLD

        # Hard reject if ADX is weak — choppy stock
        if not adx_strong:
            return None

        # ════════════════════════════════════════════════════════════
        # GATE 1B — TIME-OF-DAY FILTER (validated: 11:00-13:30 IST)
        # Skipped for manual scans — only enforced on auto scheduler
        # ════════════════════════════════════════════════════════════
        if not manual:
            now_ist  = datetime.now(IST)
            c_hour   = now_ist.hour
            c_minute = now_ist.minute
            after_start = (c_hour > TRADE_HOUR_START or
                           (c_hour == TRADE_HOUR_START and c_minute >= 0))
            before_end  = (c_hour < TRADE_HOUR_END or
                           (c_hour == TRADE_HOUR_END and c_minute <= TRADE_MIN_END))
            if not (after_start and before_end):
                return None

        # ════════════════════════════════════════════════════════════
        # GATE 1C — MACD DIRECTION ALIGNMENT
        # MACD histogram must align with EMA trend direction
        # Captures both crossovers and trend continuation entries
        # ════════════════════════════════════════════════════════════
        macd_bullish    = macd_hist > 0
        macd_bearish    = macd_hist < 0
        if ema_bullish and not macd_bullish:
            return None   # trend up but MACD negative — skip
        if ema_bearish and not macd_bearish:
            return None   # trend down but MACD positive — skip

        # ════════════════════════════════════════════════════════════
        # GATE 2 — STRONG CANDLE BODY FILTER
        # Signal candle body must be at least MIN_CANDLE_BODY of range
        # Doji / spinning top candles = indecision = skip
        # ════════════════════════════════════════════════════════════
        candle_range = high_ - low_
        candle_body  = abs(close - open_)
        if candle_range > 0:
            body_ratio = candle_body / candle_range
            if body_ratio < MIN_CANDLE_BODY:
                return None   # indecision candle — skip

        # ════════════════════════════════════════════════════════════
        # GATE 3 — CONSISTENT VOLUME TREND (last 3 candles)
        # Volume must be above average in at least 2 of last 3 candles
        # Avoids single-candle volume spikes that immediately reverse
        # ════════════════════════════════════════════════════════════
        recent_vols = df["Volume"].tail(VOL_LOOKBACK + 1).values
        avg_vol_short = df["Volume"].tail(26).mean()   # 1-day average
        vol_above_avg = sum(1 for v in recent_vols[:-1] if v > avg_vol_short)
        if vol_above_avg < 2:
            return None   # volume not consistently elevated

        # ════════════════════════════════════════════════════════════
        # SCORING — EMA+ADX already confirmed above
        # ════════════════════════════════════════════════════════════
        buy_score = 0; sell_score = 0; reasons = []

        # CORE 1: EMA trend (+3) — already validated
        if ema_bullish:
            buy_score += 3; reasons.append("EMA bullish")
        else:
            sell_score += 3; reasons.append("EMA bearish")

        # CORE 2: ADX strength (+2) — already validated
        if ema_bullish: buy_score  += 2
        else:           sell_score += 2
        reasons.append(f"ADX({adx:.0f})")

        # CORE 3: RSI pullback zone (+2)
        if ema_bullish:
            if 35 <= rsi <= 55:
                buy_score += 2; reasons.append(f"RSI({rsi:.0f}) pullback")
            elif rsi < 35:
                buy_score += 1; reasons.append(f"RSI({rsi:.0f}) oversold")
            elif rsi > 72:
                buy_score -= 1
        else:
            if 45 <= rsi <= 65:
                sell_score += 2; reasons.append(f"RSI({rsi:.0f}) pullback")
            elif rsi > 65:
                sell_score += 1; reasons.append(f"RSI({rsi:.0f}) overbought")
            elif rsi < 28:
                sell_score -= 1

        # CORE 4: MACD confirmation (+2 crossover, +1 momentum)
        # Reason only added for the direction that scored — no contradictions
        if macd_hist > 0 and prev_macd <= 0:
            buy_score += 2
            if ema_bullish: reasons.append("MACD ↑ cross")
        elif macd_hist < 0 and prev_macd >= 0:
            sell_score += 2
            if ema_bearish: reasons.append("MACD ↓ cross")
        elif macd_hist > 0:
            buy_score += 1
            if ema_bullish: reasons.append("MACD bullish")
        elif macd_hist < 0:
            sell_score += 1
            if ema_bearish: reasons.append("MACD bearish")

        # CORE 5: Volume quality (+2) — directional only
        avg_vol_short = df["Volume"].tail(26).mean()
        vol_penalty   = vol_above_avg < 1   # all recent candles below avg
        prev_close    = float(df.iloc[-2]["Close"]) if len(df) >= 2 else close

        # Hard reject: high volume AGAINST direction = distribution/short-covering
        if ema_bullish and volume > avg_vol_short * 1.5 and close < prev_close:
            return None
        if ema_bearish and volume > avg_vol_short * 1.5 and close > prev_close:
            return None

        if ema_bullish:
            if vol_ratio >= 2.0:   buy_score += 2; reasons.append(f"Vol {vol_ratio:.1f}x surge")
            elif vol_ratio >= 1.5: buy_score += 1; reasons.append(f"Vol {vol_ratio:.1f}x")
            elif vol_penalty:      buy_score -= 1
        else:
            if vol_ratio >= 2.0:   sell_score += 2; reasons.append(f"Vol {vol_ratio:.1f}x surge")
            elif vol_ratio >= 1.5: sell_score += 1; reasons.append(f"Vol {vol_ratio:.1f}x")
            elif vol_penalty:      sell_score -= 1

        # OPTIONAL: S/R, candle pattern, BB squeeze (capped at +2)
        optional_score = 0
        near_sup, near_res = find_sr_levels(df, close)
        if near_sup and ema_bullish:
            optional_score += 1; reasons.append("Near support")
        if near_res and ema_bearish:
            optional_score += 1; reasons.append("Near resistance")

        bull_c, bear_c, c_name = detect_candle_pattern(df)
        if bull_c and ema_bullish:
            optional_score += 1; reasons.append(c_name)
        elif bear_c and ema_bearish:
            optional_score += 1; reasons.append(c_name)

        avg_bb_width = df["BB_Width"].tail(50).mean()
        if bb_width < avg_bb_width * 0.75:
            optional_score += 1; reasons.append("BB squeeze")

        optional_score = min(optional_score, 2)
        # Apply optional score only to EMA-confirmed direction
        if ema_bullish: buy_score  += optional_score
        else:           sell_score += optional_score

        # Overextension penalty — directional only
        ema_dist_pct = abs(close - ema_short) / ema_short * 100
        if ema_dist_pct > 3.0:
            if ema_bullish: buy_score  -= 1
            else:           sell_score -= 1
            reasons.append(f"Extended {ema_dist_pct:.1f}% from EMA")

        # ── Determine raw signal ──────────────────────────────────────────
        if buy_score >= MIN_SCORE:
            signal, score = "BUY", buy_score
        elif sell_score >= MIN_SCORE:
            signal, score = "SELL", sell_score
        else:
            return None

        # ── SCORE 9 + ADX QUALITY FILTER ─────────────────────────────────
        # Score-9 signals validated at 33% WR — only keep if ADX ≥ 30
        # Score 10+ passes regardless (validated 40-47% WR)
        if score == MIN_SCORE and adx < SCORE9_ADX_MIN:
            return None

        # ════════════════════════════════════════════════════════════
        # DAILY TIMEFRAME CONFIRMATION + daily ATR for SL
        # ════════════════════════════════════════════════════════════
        stock_trend, daily_atr, daily_rsi, daily_ema_bull = get_stock_daily_data(symbol)

        penalty_reasons = []

        # Daily trend mismatch → -2 penalty
        if signal == "BUY"  and stock_trend == "bearish":
            score -= 2; penalty_reasons.append("Daily bearish(-2)")
        elif signal == "SELL" and stock_trend == "bullish":
            score -= 2; penalty_reasons.append("Daily bullish(-2)")

        # Higher timeframe confirmation:
        # BUY: daily RSI should be > 45 (not in deep downtrend)
        # SELL: daily RSI should be < 55
        if signal == "BUY" and daily_rsi < 45:
            score -= 1; penalty_reasons.append(f"Daily RSI weak({daily_rsi:.0f})")
        elif signal == "SELL" and daily_rsi > 55:
            score -= 1; penalty_reasons.append(f"Daily RSI strong({daily_rsi:.0f})")

        # Nifty directional bias — counter-trend signals need higher conviction
        # Bullish market: SELL needs MIN_SCORE + COUNTER_TREND_PENALTY
        # Bearish market: BUY  needs MIN_SCORE + COUNTER_TREND_PENALTY
        if nifty_trend == "bullish":
            buy_threshold  = MIN_SCORE
            sell_threshold = MIN_SCORE + COUNTER_TREND_PENALTY
        elif nifty_trend == "bearish":
            buy_threshold  = MIN_SCORE + COUNTER_TREND_PENALTY
            sell_threshold = MIN_SCORE
        else:
            buy_threshold  = MIN_SCORE
            sell_threshold = MIN_SCORE

        if signal == "BUY"  and score < buy_threshold:
            penalty_reasons.append(f"Nifty bearish — BUY needs {buy_threshold}+ (has {score})")
            score = 0   # force rejection below
        elif signal == "SELL" and score < sell_threshold:
            penalty_reasons.append(f"Nifty bullish — SELL needs {sell_threshold}+ (has {score})")
            score = 0

        # News sentiment penalty
        _, sentiment_label = get_news_sentiment(symbol)
        if signal == "BUY"  and sentiment_label == "Negative":
            score -= 1; penalty_reasons.append("News negative(-1)")
        elif signal == "SELL" and sentiment_label == "Positive":
            score -= 1; penalty_reasons.append("News positive(-1)")

        if penalty_reasons:
            reasons.append("Penalties: " + ", ".join(penalty_reasons))

        if score < MIN_SCORE:
            return None

        # ════════════════════════════════════════════════════════════
        # SL USING DAILY ATR (not 15-min ATR)
        # Daily ATR is typically 5-10x the 15-min ATR — much more
        # realistic and prevents wick-based false SL triggers
        # ════════════════════════════════════════════════════════════
        # Blended ATR: 60% daily + 40% (15-min×4) — validated in backtest
        # Pure daily ATR → too wide (timeouts); pure 15-min → too tight (wick hits)
        if daily_atr > 0:
            effective_atr = 0.6 * daily_atr + 0.4 * (atr_15m * 4)
        else:
            effective_atr = atr_15m * 3
        min_sl_dist   = close * 0.005   # min 0.5% of price
        max_sl_dist   = close * 0.025   # max 2.5% of price (cap runaway SLs)
        sl_dist       = min(max(ATR_MULT * effective_atr, min_sl_dist), max_sl_dist)

        if signal == "BUY":
            stop_loss = round(close - sl_dist, 2)
            target    = round(close + sl_dist * RISK_REWARD, 2)
        else:
            stop_loss = round(close + sl_dist, 2)
            target    = round(close - sl_dist * RISK_REWARD, 2)

        # Final RR sanity check
        risk   = abs(close - stop_loss)
        reward = abs(target - close)
        if risk == 0 or reward / risk < RISK_REWARD * 0.9:
            return None

        data_source = source_log[0] if source_log else "unknown"

        # ── AI CONFIDENCE SCORE ───────────────────────────────────────────
        sl_pct  = abs(close - stop_loss) / close * 100
        tgt_pct = abs(target - close)    / close * 100
        now_h   = datetime.now(IST).hour
        ai_conf = ai_confidence(
            score   = score,
            adx     = adx,
            rsi     = rsi,
            hour    = now_h,
            is_sell = 1 if signal == "SELL" else 0,
            sl_pct  = sl_pct,
            tgt_pct = tgt_pct,
        )

        return {
            "symbol":       symbol,
            "signal":       signal,
            "entry":        round(close, 2),
            "target":       target,
            "stop_loss":    stop_loss,
            "score":        score,
            "rsi":          round(rsi, 1),
            "adx":          round(adx, 1),
            "vol_ratio":    round(vol_ratio, 1),
            "daily_atr":    round(effective_atr, 2),
            "daily_rsi":    daily_rsi,
            "sentiment":    sentiment_label,
            "candle":       (c_name if (signal=="BUY" and bull_c) or (signal=="SELL" and bear_c) else "—"),
            "reasons":      " | ".join(reasons),
            "time":         datetime.now(IST).strftime("%H:%M:%S IST"),
            "data_source":  data_source,
            "current_price": round(close, 2),
            "ai_confidence": ai_conf,          # 0.0-1.0 or null if model not loaded
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



def reset_tracker_if_new_week():
    """Clears all trades at the start of each new week (Monday IST)."""
    week_start = get_week_start()
    if trade_tracker["week_start"] != week_start:
        old_week = trade_tracker["week_start"]
        trade_tracker["trades"]     = []
        trade_tracker["week_start"] = week_start
        print(f"[tracker] Weekly reset — new week started {week_start}")
        if old_week:
            delete_week_db(old_week)


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
    entry    = float(signal["entry"])
    sl       = float(signal["stop_loss"])
    tgt      = float(signal["target"])
    sl_pct   = round(abs(entry - sl)  / entry * 100, 3) if entry > 0 else 0
    tgt_pct  = round(abs(tgt  - entry) / entry * 100, 3) if entry > 0 else 0
    # Parse IST hour from signal time string e.g. "10:30:00 IST"
    try:
        sig_hour = int(signal.get("time","12:00:00").split(":")[0])
    except Exception:
        sig_hour = datetime.now(IST).hour

    trade_tracker["trades"].append({
        "id":           trade_id,
        "symbol":       signal["symbol"],
        "signal":       signal["signal"],
        "entry":        signal["entry"],
        "target":       signal["target"],
        "stop_loss":    signal["stop_loss"],
        "score":        signal["score"],
        "adx":          signal.get("adx",    0),
        "rsi":          signal.get("rsi",    50),
        "hour":         sig_hour,
        "sl_pct":       sl_pct,
        "tgt_pct":      tgt_pct,
        "ai_confidence":signal.get("ai_confidence", None),
        "source":       signal.get("data_source", "kite"),
        "time":         signal.get("time", ""),
        "date":         datetime.now(IST).strftime("%d %b %Y"),
        "outcome":      "OPEN",
        "resolved_at":  None,
        "pnl_pct":      None,
        "exit":         None,
    })
    print(f"[tracker] Added {signal['signal']} trade: {signal['symbol']}")
    save_trade_db(trade_tracker["trades"][-1])


def resolve_trade_outcome(trade: dict, df: pd.DataFrame) -> tuple:
    """
    Given a candle DataFrame, check if target or SL was hit using
    CLOSING PRICE CONFIRMATION — requires candle to CLOSE beyond the
    level (not just wick), eliminating false triggers from spikes.

    Returns (outcome, pnl_pct) or (None, None) if neither hit yet.
    """
    entry     = float(trade["entry"])
    target    = float(trade["target"])
    stop_loss = float(trade["stop_loss"])
    signal    = trade["signal"]

    # Walk candle by candle — use CLOSE not High/Low for confirmation
    # This is the key fix: a candle must CLOSE beyond the level
    for _, row in df.iterrows():
        close = float(row["Close"])
        high  = float(row["High"])
        low   = float(row["Low"])

        if signal == "BUY":
            # SL: candle closes below stop_loss (confirmed, not a wick)
            if close <= stop_loss:
                return "LOSS", round((stop_loss - entry) / entry * 100, 2)
            # Target: candle high touches or exceeds target (target is profit, ok to use high)
            if high >= target:
                return "WIN",  round((target - entry) / entry * 100, 2)
        else:  # SELL
            # SL: candle closes above stop_loss
            if close >= stop_loss:
                return "LOSS", round((entry - stop_loss) / entry * 100, 2)
            # Target: candle low touches or goes below target
            if low <= target:
                return "WIN",  round((entry - target) / entry * 100, 2)

    return None, None


def notify_outcome(trade: dict, outcome: str, pnl_pct: float):
    """Send Telegram message when a trade outcome is resolved."""
    icon  = "✅" if outcome == "WIN" else "❌"
    entry = float(trade["entry"])
    lines = [
        f"{icon} *{outcome} — {trade['symbol']} {trade['signal']}*",
        "",
        f"Entry  : Rs {entry:,.2f}",
        f"Target : Rs {float(trade['target']):,.2f}",
        f"SL     : Rs {float(trade['stop_loss']):,.2f}",
        f"P&L    : {pnl_pct:+.2f}%",
        f"Time   : {trade.get('resolved_at','')}",
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


def check_open_trades():
    """
    For every OPEN trade:
    1. Try Kite Connect first — accurate candles + closing price confirmation
    2. Fall back to yfinance if Kite not logged in
    Uses CLOSE-based SL confirmation to avoid wick false triggers.
    """
    reset_tracker_if_new_week()
    open_trades = [t for t in trade_tracker["trades"] if t["outcome"] == "OPEN"]
    if not open_trades:
        return

    kite_active = kite_get() is not None
    print(f"[tracker] Checking {len(open_trades)} open trade(s) — Kite: {'YES' if kite_active else 'NO (yfinance fallback)'}")

    for trade in open_trades:
        try:
            df = None

            # ── Use fetch_candles (Kite first, yfinance fallback) ────────────
            df = fetch_candles(trade["symbol"], interval="15minute", days=2)

            # ── Fall back to fetch_candles (which itself falls back to yfinance) ──
            if df is None or len(df) == 0:
                df = fetch_candles(trade["symbol"], interval="15minute", days=2)

            if df is None or len(df) == 0:
                continue

            # Only check last 26 candles (1 trading day)
            df = df.tail(26)
            outcome, pnl_pct = resolve_trade_outcome(trade, df)

            if outcome:
                trade["outcome"]     = outcome
                trade["pnl_pct"]     = pnl_pct
                trade["resolved_at"] = datetime.now(IST).strftime("%H:%M:%S IST")
                save_trade_db(trade)
                source = "Kite" if kite_active else "yfinance"
                print(f"[tracker] {trade['symbol']} → {outcome} ({pnl_pct:+.2f}%) via {source}")
                threading.Thread(target=notify_outcome, args=(trade, outcome, pnl_pct), daemon=True).start()

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
        nifty_trend, nifty_adx = get_nifty_trend()
        signal_count = 0

        # ── Nifty ADX gate — skip entire scan if market is choppy ─────────
        if nifty_adx < NIFTY_ADX_GATE:
            print(f"[scheduler] Nifty ADX={nifty_adx:.1f} < {NIFTY_ADX_GATE} — market choppy, scan skipped")
            return

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
                    add_trade(result)   # always register in trade tracker
                    ok, reason = should_notify_telegram(result)
                    if ok:
                        send_telegram(result)
                        print(f"[scheduler] {result['signal']} signal: {result['symbol']} score={result['score']} → Telegram sent")
                    else:
                        print(f"[scheduler] {result['signal']} signal: {result['symbol']} score={result['score']} → Telegram skipped ({reason})")

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
async def status(auth: bool = Depends(verify_auth)):
    """Quick health check + market state + Nifty trend."""
    ist_now = datetime.now(IST).strftime("%d %b %Y %H:%M:%S IST")
    loop = asyncio.get_event_loop()
    trend, nifty_adx = await loop.run_in_executor(None, get_nifty_trend)
    return {
        "market_open":    market_is_open(),
        "ist_time":       ist_now,
        "nifty_trend":    trend,
        "nifty_adx":      nifty_adx,
        "nifty_adx_gate": NIFTY_ADX_GATE,
        "talib":          TALIB,
        "symbols":        len(NIFTY100_SYMBOLS),
        "scan_running":   scan_state["running"],
        "min_score":      MIN_SCORE,
        "risk_reward":    RISK_REWARD,
        "kite_active":    kite_get() is not None,
        "tg_filters": {
            "min_score":   TG_MIN_SCORE,
            "min_adx":     TG_MIN_ADX,
            "min_rr":      TG_MIN_RR,
            "news_filter": TG_NEWS_FILTER or "Any",
            "data_source": TG_DATA_SOURCE or "Any",
        },
    }


@app.get("/scan/stream")
async def scan_stream(symbols: str = "", auth: bool = Depends(verify_auth)):
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
            nifty_trend, nifty_adx = await loop.run_in_executor(None, get_nifty_trend)
            yield f"data: {json.dumps({'type':'start','total':len(symbol_list),'nifty_trend':nifty_trend,'nifty_adx':nifty_adx})}\n\n"

            # Nifty ADX gate — INFO only for manual scan, not blocking
            if nifty_adx < NIFTY_ADX_GATE:
                yield f"data: {json.dumps({'type':'warning','reason':f'Nifty ADX={nifty_adx:.1f} < {NIFTY_ADX_GATE} — market choppy, signals may be weak'})}\n\n"

            signal_count = 0
            for idx, symbol in enumerate(symbol_list, 1):
                if scan_state["stop_flag"]:
                    yield f"data: {json.dumps({'type':'stopped'})}\n\n"
                    return

                # Progress event
                yield f"data: {json.dumps({'type':'progress','symbol':symbol,'index':idx,'total':len(symbol_list)})}\n\n"
                await asyncio.sleep(0)   # yield control so the event is flushed

                # Run blocking IO in thread pool — manual=True bypasses time filter
                result = await loop.run_in_executor(None, analyze_stock, symbol, nifty_trend, True)

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
                        add_trade(result)   # always register in trade tracker
                        ok, reason = should_notify_telegram(result)
                        if ok:
                            threading.Thread(target=send_telegram, args=(result,), daemon=True).start()

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
async def scan_result(auth: bool = Depends(verify_auth)):
    """Returns the results of the most recently completed scan."""
    return JSONResponse({
        "results":   scan_state["results"],
        "scanned":   scan_state["scanned"],
        "total":     scan_state["total"],
        "last_scan": scan_state["last_scan"],
        "running":   scan_state["running"],
    })


@app.post("/scan/stop")
async def scan_stop(auth: bool = Depends(verify_auth)):
    """Signals the running scan to stop after the current stock."""
    if not scan_state["running"]:
        return {"stopped": False, "msg": "No scan running"}
    scan_state["stop_flag"] = True
    return {"stopped": True, "msg": "Stop signal sent"}


@app.api_route("/ping", methods=["GET", "HEAD"])
async def ping():
    return {"ping": "pong", "time": datetime.now(IST).isoformat()}


# ════════════════════════════════════════════════════════════════════════════
# KITE CONNECT AUTH ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.get("/kite/login")
async def kite_login():
    """Serves a login page that redirects to Zerodha."""
    if not KITE_AVAILABLE:
        return HTMLResponse("<h2>kiteconnect not installed on server</h2>", status_code=500)
    url = kite_get_login_url()
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>SwingScan — Zerodha Login</title>
  <style>
    body{{margin:0;font-family:monospace;background:#0a0d12;color:#e2e8f0;display:flex;
         align-items:center;justify-content:center;min-height:100vh;}}
    .box{{background:#111620;border:1px solid #1e2a3d;border-radius:8px;padding:40px;
          max-width:440px;width:90%;text-align:center;}}
    h2{{color:#00e5a0;margin:0 0 10px;font-size:22px;}}
    p{{color:#64748b;font-size:13px;line-height:1.7;margin:0 0 24px;}}
    a.btn{{display:inline-block;background:#00e5a0;color:#000;font-weight:700;
           padding:12px 32px;border-radius:6px;text-decoration:none;font-size:14px;
           transition:opacity 0.2s;}}
    a.btn:hover{{opacity:0.85;}}
    .note{{margin-top:20px;font-size:11px;color:#334155;}}
  </style>
</head>
<body>
  <div class="box">
    <h2>Zerodha Login</h2>
    <p>Click below to login with your Zerodha credentials.<br/>
       After login you will be automatically redirected back here.</p>
    <a class="btn" href="{url}">Login with Zerodha ↗</a>
    <div class="note">Your credentials are never stored. Only today's session token is kept in memory.</div>
  </div>
</body>
</html>"""
    return HTMLResponse(html)


@app.get("/kite/callback")
async def kite_callback(token: str = "", request_token: str = "", action: str = "", status: str = ""):
    """
    Zerodha redirects here after login with ?request_token=XXX&action=login&status=success
    Automatically exchanges token and shows a success/error HTML page.
    No copy-pasting needed — fully automatic.
    """
    def html_page(title, message, color, auto_close=False):
        close_script = """
        <p style='color:#64748b;font-size:12px;margin-top:16px;'>This tab will close in 3 seconds...</p>
        <script>setTimeout(()=>window.close(),3000);</script>""" if auto_close else ""
        return HTMLResponse(f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>SwingScan — Kite Auth</title>
  <style>
    body{{margin:0;font-family:monospace;background:#0a0d12;color:#e2e8f0;display:flex;
         align-items:center;justify-content:center;min-height:100vh;}}
    .box{{background:#111620;border:1px solid #1e2a3d;border-radius:8px;padding:40px;
          max-width:440px;width:90%;text-align:center;}}
    h2{{color:{color};margin:0 0 14px;font-size:22px;}}
    p{{color:#94a3b8;font-size:13px;line-height:1.7;margin:0;}}
    .back{{display:inline-block;margin-top:20px;background:{color};color:#000;
           font-weight:700;padding:10px 24px;border-radius:6px;text-decoration:none;font-size:13px;}}
  </style>
</head>
<body>
  <div class="box">
    <h2>{title}</h2>
    <p>{message}</p>
    {close_script}
    <a class="back" href="/">Back to Dashboard</a>
  </div>
</body>
</html>""")

    if not KITE_AVAILABLE:
        return html_page("Error", "kiteconnect not installed on server.", "#ff4757")

    # Zerodha sends status=success or status=error
    if status == "error" or action == "error":
        return html_page("Login Cancelled", "Zerodha login was cancelled or failed.<br/>Please try again from the dashboard.", "#ff4757")

    req_token = token or request_token
    if not req_token:
        return html_page("Missing Token",
            "No request_token found in URL.<br/>Please login again from the dashboard.", "#f4b942")

    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        session      = kite.generate_session(req_token, api_secret=KITE_API_SECRET)
        access_token = session["access_token"]
        success      = kite_set_token(access_token)
        if success:
            profile = kite_state["kite"].profile()
            name    = profile.get("user_name", "?")
            return html_page(
                "Login Successful ✓",
                f"Welcome <strong style='color:#00e5a0'>{name}</strong>!<br/>"
                f"Kite is now active for today.<br/>"
                f"Accurate candle data and closing-price SL confirmation are now enabled.",
                "#00e5a0",
                auto_close=True,
            )
        else:
            return html_page("Login Failed", "Token exchange failed. Please try again.", "#ff4757")
    except Exception as e:
        return html_page("Error", f"Authentication error:<br/>{str(e)}", "#ff4757")


@app.get("/kite/status")
async def kite_status(auth: bool = Depends(verify_auth)):
    """Check if Kite is authenticated for today."""
    kite = kite_get()
    today = datetime.now(IST).date().isoformat()
    if kite:
        try:
            profile = kite.profile()
            return JSONResponse({
                "logged_in":  True,
                "user":       profile.get("user_name", "?"),
                "token_date": kite_state["token_date"],
                "valid":      kite_state["token_date"] == today,
            })
        except Exception:
            pass
    login_url = kite_get_login_url() if KITE_AVAILABLE else ""
    return JSONResponse({
        "logged_in":  False,
        "login_url":  login_url,
        "message":    "Open login_url to authenticate. Then call /kite/callback?token=REQUEST_TOKEN",
    })


@app.get("/kite/token")
async def kite_token(auth: bool = Depends(verify_auth)):
    """
    Returns today's Kite access token for running backtest.py locally.
    Protected by API password.
    """
    today = datetime.now(IST).date().isoformat()
    if kite_state["access_token"] and kite_state["token_date"] == today:
        return JSONResponse({
            "access_token": kite_state["access_token"],
            "token_date":   kite_state["token_date"],
            "note":         "Valid for today only. Use: python backtest.py --kite-token TOKEN",
        })
    return JSONResponse({"error": "No valid token — login via /kite/login first"}, status_code=401)


@app.get("/quotes")
async def get_quotes(symbols: str = "", auth: bool = Depends(verify_auth)):
    """
    Fetch current live prices for a list of symbols.
    Used by UI to show live price vs entry/target/SL.
    ?symbols=RELIANCE,TCS,INFY
    Kite if logged in, else yfinance fallback.
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not symbol_list:
        return JSONResponse({"quotes": {}})

    quotes = {}
    kite   = kite_get()

    if kite:
        # Batch fetch from Kite — single API call for all symbols
        try:
            instruments = [f"NSE:{s}" for s in symbol_list]
            data        = kite.quote(instruments)
            for sym in symbol_list:
                key  = f"NSE:{sym}"
                item = data.get(key, {})
                if item:
                    quotes[sym] = {
                        "price":  item.get("last_price", 0),
                        "change": round(item.get("net_change", 0), 2),
                        "source": "kite",
                    }
        except Exception as e:
            print(f"[quotes] Kite batch error: {e}")

    # yfinance fallback for any symbols not fetched via Kite
    missing = [s for s in symbol_list if s not in quotes]
    if missing:
        loop = asyncio.get_event_loop()
        def fetch_yf_prices():
            result = {}
            try:
                tickers = " ".join(s + ".NS" for s in missing)
                data    = yf.download(tickers, period="1d", interval="1m",
                                      progress=False, auto_adjust=True)
                if data is not None and len(data) > 0:
                    if len(missing) == 1:
                        price = float(data["Close"].iloc[-1])
                        result[missing[0]] = {"price": round(price, 2), "change": 0, "source": "yfinance"}
                    else:
                        for sym in missing:
                            try:
                                price = float(data["Close"][sym + ".NS"].iloc[-1])
                                result[sym] = {"price": round(price, 2), "change": 0, "source": "yfinance"}
                            except Exception:
                                pass
            except Exception:
                pass
            return result
        yf_prices = await loop.run_in_executor(None, fetch_yf_prices)
        quotes.update(yf_prices)

    return JSONResponse({"quotes": quotes})


@app.get("/tracker/trades")
async def tracker_trades(auth: bool = Depends(verify_auth)):
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
async def tracker_reset(auth: bool = Depends(verify_auth)):
    """Manually reset the trade tracker."""
    old_week = trade_tracker["week_start"]
    trade_tracker["trades"]     = []
    trade_tracker["week_start"] = get_week_start()
    if old_week:
        delete_week_db(old_week)
    return {"reset": True, "week_start": trade_tracker["week_start"]}


@app.get("/tracker/export")
async def tracker_export(auth: bool = Depends(verify_auth)):
    """
    Export all CLOSED trades (WIN/LOSS) as a CSV file.
    Format matches backtest CSV — ready for train_model.py retraining.
    Only exports resolved trades (not OPEN ones — no outcome yet).
    """
    import io, csv as csv_mod
    from fastapi.responses import StreamingResponse

    closed = [t for t in trade_tracker["trades"]
              if t["outcome"] in ("WIN", "LOSS")]

    if not closed:
        return JSONResponse({"error": "No closed trades to export"}, status_code=404)

    # Columns matching backtest CSV exactly
    fieldnames = [
        "symbol", "time", "date", "hour", "signal", "score",
        "entry", "target", "stop_loss", "outcome", "pnl_pct",
        "candles", "exit", "rsi", "adx", "source",
        "sl_pct", "tgt_pct", "ai_confidence",
    ]

    output = io.StringIO()
    writer = csv_mod.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for t in closed:
        row = {f: t.get(f, "") for f in fieldnames}
        # candles not tracked in live mode — set to 0
        if not row.get("candles"):
            row["candles"] = 0
        # exit type
        if not row.get("exit"):
            row["exit"] = "TARGET" if t["outcome"] == "WIN" else "SL"
        writer.writerow(row)

    output.seek(0)
    fname = f"live_trades_{datetime.now(IST).strftime('%Y%m%d_%H%M')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"}
    )


@app.get("/prices")
async def get_prices(symbols: str = "", auth: bool = Depends(verify_auth)):
    """
    Fetch current live prices for a comma-separated list of symbols.
    Uses Kite quote if logged in, else yfinance last close.
    Returns: {SYMBOL: {price, change_pct}, ...}
    """
    if not symbols:
        return JSONResponse({})

    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    result = {}
    loop   = asyncio.get_event_loop()

    async def fetch_one(sym):
        def _fetch():
            kite = kite_get()
            if kite:
                try:
                    q    = kite.quote([f"NSE:{sym}"])
                    data = q.get(f"NSE:{sym}", {})
                    lp   = data.get("last_price", 0)
                    prev = data.get("ohlc", {}).get("close", lp) or lp
                    chg  = round((lp - prev) / prev * 100, 2) if prev else 0
                    return {"price": lp, "change_pct": chg, "source": "kite"}
                except Exception:
                    pass
            # yfinance fallback — use last close
            try:
                tk = yf.Ticker(sym + ".NS")
                info = tk.fast_info
                lp   = float(info.last_price or 0)
                prev = float(info.previous_close or lp)
                chg  = round((lp - prev) / prev * 100, 2) if prev else 0
                return {"price": lp, "change_pct": chg, "source": "yfinance"}
            except Exception:
                return {"price": 0, "change_pct": 0, "source": "error"}

        return sym, await loop.run_in_executor(None, _fetch)

    # Fetch all symbols concurrently
    tasks   = [fetch_one(s) for s in symbol_list]
    results = await asyncio.gather(*tasks)
    return JSONResponse({sym: data for sym, data in results})


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