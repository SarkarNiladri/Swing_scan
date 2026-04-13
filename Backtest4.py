"""
Swing Trading Strategy - Backtester (Fixed)
=============================================
Fixes applied:
  1. in_trade reset bug fixed - while loop with proper i skip after each trade
  2. Nifty 50 trend filter added - matches live scanner logic
  3. Candlestick direction check added - proper hammer/shooting star
  4. TIMEOUT trades excluded from win rate calculation
  5. Market hours filter added - only 9:15 AM - 3:30 PM IST candles

Usage:
  Backtest all Nifty 100 stocks : python backtest.py
  Backtest a single stock       : python backtest.py RELIANCE
  Backtest a few stocks         : python backtest.py RELIANCE TCS INFY

Requirements:
    pip install yfinance pandas numpy rich pytz
"""
import Symbols
import warnings
warnings.filterwarnings("ignore")
import Symbols
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()
IST     = pytz.timezone("Asia/Kolkata")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RSI_PERIOD    = 14
RSI_OVERSOLD  = 40
BB_PERIOD     = 20
BB_STD        = 2
EMA_SHORT     = 20
EMA_LONG      = 50
ADX_PERIOD    = 14
ADX_THRESHOLD = 25
ATR_PERIOD    = 14
VOLUME_MULT   = 1.7
SR_ZONE_PCT   = 0.015
RISK_REWARD   = 2.0
MIN_SCORE     = 9    # Raised from 8 → only best setups
ATR_MULT      = 2.0   # Raised from 1.5 → wider stop, less noise

BT_PERIOD     = "60d"
BT_INTERVAL   = "15m"
MAX_LOOKAHEAD = 26 * 5   # 5 trading days of 15m bars

# ─────────────────────────────────────────────
# NIFTY 100 SYMBOLS
# ─────────────────────────────────────────────
NIFTY100_SYMBOLS  = Symbols.symbols_100

# ─────────────────────────────────────────────
# TECHNICAL INDICATORS (no external ta library)
# ─────────────────────────────────────────────
def calculate_rsi(data, period=14):
    delta = data.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()


def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast  = calculate_ema(data, fast)
    ema_slow  = calculate_ema(data, slow)
    macd      = ema_fast - ema_slow
    signal_ln = calculate_ema(macd, signal)
    return macd, signal_ln, macd - signal_ln


def calculate_bbands(data, period=20, num_std=2):
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    return sma + std * num_std, sma, sma - std * num_std


def calculate_atr(df, period=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_adx(df, period=14):
    plus_dm  = df["High"].diff().clip(lower=0)
    minus_dm = (-df["Low"].diff()).clip(lower=0)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"]  - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr      = tr.rolling(period).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.rolling(period).mean()  / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr
    di_sum   = (plus_di + minus_di).replace(0, np.nan)
    dx       = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.rolling(period).mean(), plus_di, minus_di


# ─────────────────────────────────────────────
# FIX 2 - NIFTY 50 TREND MAP
# Built once before the backtest loop starts
# ─────────────────────────────────────────────
def build_nifty_trend_map() -> dict:
    """
    Returns a dict: { 'YYYY-MM-DD' -> 'bullish' | 'bearish' | 'neutral' }
    Uses EMA20 > EMA50 logic on daily Nifty 50 data.
    """
    console.print("[dim]  Fetching Nifty 50 trend data...[/dim]")
    try:
        df = yf.download("^NSEI", period="6mo", interval="1d",
                         progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        df["EMA20"] = calculate_ema(df["Close"], 20)
        df["EMA50"] = calculate_ema(df["Close"], 50)
        df.dropna(inplace=True)

        trend_map = {}
        for dt, row in df.iterrows():
            close = float(row["Close"])
            e20   = float(row["EMA20"])
            e50   = float(row["EMA50"])
            if close > e20 > e50:   trend = "bullish"
            elif close < e20 < e50: trend = "bearish"
            else:                   trend = "neutral"
            date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
            trend_map[date_str] = trend

        return trend_map
    except Exception as e:
        console.print(f"[yellow]  Nifty trend fetch failed ({e}), using neutral.[/yellow]")
        return {}


def get_nifty_trend_for_date(trend_map: dict, timestamp) -> str:
    try:
        date_str = timestamp.strftime("%Y-%m-%d") if hasattr(timestamp, "strftime") else str(timestamp)[:10]
        if date_str in trend_map:
            return trend_map[date_str]
        # Walk back up to 5 days to find nearest trading day
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        for delta in range(1, 6):
            prev = (d - timedelta(days=delta)).strftime("%Y-%m-%d")
            if prev in trend_map:
                return trend_map[prev]
    except Exception:
        pass
    return "neutral"


# ─────────────────────────────────────────────
# IMPROVEMENT 2 — PER-STOCK DAILY TREND MAP
# Built once per symbol before signal scanning
# ─────────────────────────────────────────────
def build_stock_trend_map(symbol: str) -> dict:
    """
    Returns a dict: { 'YYYY-MM-DD' -> 'bullish' | 'bearish' | 'neutral' }
    for the individual stock using daily EMA20 vs EMA50.
    Filters out BUY signals on stocks in daily downtrend and vice versa.
    """
    try:
        ticker = symbol + ".NS"
        df = yf.download(ticker, period="6mo", interval="1d",
                         progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)
        df["EMA20"] = calculate_ema(df["Close"], 20)
        df["EMA50"] = calculate_ema(df["Close"], 50)
        df.dropna(inplace=True)

        trend_map = {}
        for dt, row in df.iterrows():
            close = float(row["Close"])
            e20   = float(row["EMA20"])
            e50   = float(row["EMA50"])
            if close > e20 > e50:   trend = "bullish"
            elif close < e20 < e50: trend = "bearish"
            else:                   trend = "neutral"
            date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
            trend_map[date_str] = trend
        return trend_map
    except Exception:
        return {}
# ─────────────────────────────────────────────
def find_sr_levels(df: pd.DataFrame, close: float) -> tuple:
    levels = []
    highs  = df["High"].rolling(5, center=True).max()
    lows   = df["Low"].rolling(5, center=True).min()
    levels.extend(df["High"][df["High"] == highs].tail(50).tolist())
    levels.extend(df["Low"][df["Low"] == lows].tail(50).tolist())
    levels.append(float(df["High"].max()))
    levels.append(float(df["Low"].min()))
    for base in [50, 100]:
        for offset in [-1, 0, 1]:
            levels.append((round(close / base) + offset) * base)
    zone            = close * SR_ZONE_PCT
    near_support    = any(abs(close - l) <= zone and l <= close for l in levels if l > 0)
    near_resistance = any(abs(close - l) <= zone and l >= close for l in levels if l > 0)
    return near_support, near_resistance


# ─────────────────────────────────────────────
# FIX 3 - CANDLESTICK PATTERNS (direction-aware)
# ─────────────────────────────────────────────
def detect_candle_pattern(df: pd.DataFrame) -> tuple:
    if len(df) < 3:
        return False, False, ""

    c1 = df.iloc[-3]; c2 = df.iloc[-2]; c3 = df.iloc[-1]
    o3, h3, l3, c3c = float(c3["Open"]), float(c3["High"]), float(c3["Low"]), float(c3["Close"])
    o2, h2, l2, c2c = float(c2["Open"]), float(c2["High"]), float(c2["Low"]), float(c2["Close"])
    o1, h1, l1, c1c = float(c1["Open"]), float(c1["High"]), float(c1["Low"]), float(c1["Close"])

    body3       = abs(c3c - o3)
    range3      = h3 - l3 if h3 != l3 else 0.001
    upper_wick3 = h3 - max(c3c, o3)
    lower_wick3 = min(c3c, o3) - l3

    # FIX: added c3c >= o3 (bullish candle required for hammer)
    hammer = (
        lower_wick3 >= 2 * body3 and
        upper_wick3 <= 0.1 * range3 and
        body3 > 0 and
        c3c >= o3
    )
    bull_engulf  = c2c < o2 and c3c > o3 and o3 <= c2c and c3c >= o2
    morning_star = (
        c1c < o1 and
        abs(c2c - o2) < 0.3 * (h2 - l2 if h2 != l2 else 0.001) and
        c3c > o3 and c3c > (o1 + c1c) / 2
    )

    # FIX: added c3c <= o3 (bearish candle required for shooting star)
    shooting_star = (
        upper_wick3 >= 2 * body3 and
        lower_wick3 <= 0.1 * range3 and
        body3 > 0 and
        c3c <= o3
    )
    bear_engulf  = c2c > o2 and c3c < o3 and o3 >= c2c and c3c <= o2
    evening_star = (
        c1c > o1 and
        abs(c2c - o2) < 0.3 * (h2 - l2 if h2 != l2 else 0.001) and
        c3c < o3 and c3c < (o1 + c1c) / 2
    )

    bullish = hammer or bull_engulf or morning_star
    bearish = shooting_star or bear_engulf or evening_star

    if hammer:          name = "Hammer"
    elif bull_engulf:   name = "Bull Engulf"
    elif morning_star:  name = "Morning Star"
    elif shooting_star: name = "Shoot Star"
    elif bear_engulf:   name = "Bear Engulf"
    elif evening_star:  name = "Evening Star"
    else:               name = ""

    return bullish, bearish, name


# ─────────────────────────────────────────────
# SIGNAL LOGIC
# ─────────────────────────────────────────────
def compute_signal(df_slice: pd.DataFrame) -> dict | None:
    if len(df_slice) < 5:
        return None

    last = df_slice.iloc[-1]
    prev = df_slice.iloc[-2]

    close     = float(last["Close"])
    volume    = float(last["Volume"])
    rsi       = float(last.get("RSI",       50))
    macd_hist = float(last.get("MACDh",      0))
    prev_macd = float(prev.get("MACDh",      0))
    bb_lower  = float(last.get("BB_Lower",  close))
    bb_upper  = float(last.get("BB_Upper",  close))
    ema_short = float(last.get("EMA_Short", close))
    ema_long  = float(last.get("EMA_Long",  close))
    adx       = float(last.get("ADX",          0))
    atr       = float(last.get("ATR",  close * 0.01))

    avg_vol = df_slice["Volume"].tail(20 * 26).mean()
    if volume <= avg_vol * VOLUME_MULT:
        return None

    buy_score = 0; sell_score = 0; reasons = []

    if rsi < RSI_OVERSOLD:
        buy_score += 2; reasons.append(f"RSI({rsi:.0f})")
    elif rsi > (100 - RSI_OVERSOLD):
        sell_score += 2; reasons.append(f"RSI({rsi:.0f})")

    if macd_hist > 0 and prev_macd <= 0:
        buy_score += 2; reasons.append("MACD up")
    elif macd_hist < 0 and prev_macd >= 0:
        sell_score += 2; reasons.append("MACD dn")
    elif macd_hist > 0:  buy_score  += 1
    elif macd_hist < 0:  sell_score += 1

    if close <= bb_lower:
        buy_score += 2; reasons.append("BB lower")
    elif close >= bb_upper:
        sell_score += 2; reasons.append("BB upper")

    if ema_short > ema_long:
        buy_score += 1; reasons.append("EMA up")
    else:
        sell_score += 1; reasons.append("EMA dn")

    if adx > ADX_THRESHOLD:
        if ema_short > ema_long: buy_score  += 2
        else:                    sell_score += 2
        reasons.append(f"ADX({adx:.0f})")

    near_sup, near_res = find_sr_levels(df_slice, close)
    if near_sup:  buy_score  += 2; reasons.append("S/R sup")
    if near_res:  sell_score += 2; reasons.append("S/R res")

    bull_c, bear_c, c_name = detect_candle_pattern(df_slice)
    if bull_c:   buy_score  += 2; reasons.append(c_name)
    elif bear_c: sell_score += 2; reasons.append(c_name)

    if buy_score >= MIN_SCORE:
        signal, score = "BUY", buy_score
    elif sell_score >= MIN_SCORE:
        signal, score = "SELL", sell_score
    else:
        return None

    if signal == "BUY":
        stop_loss = round(close - ATR_MULT * atr, 2)
        target    = round(close + (close - stop_loss) * RISK_REWARD, 2)
    else:
        stop_loss = round(close + ATR_MULT * atr, 2)
        target    = round(close - (stop_loss - close) * RISK_REWARD, 2)

    return {
        "signal":    signal,
        "score":     score,
        "entry":     round(close, 2),
        "target":    target,
        "stop_loss": stop_loss,
        "atr":       round(atr, 2),
        "reasons":   " | ".join(reasons),
    }


# ─────────────────────────────────────────────
# OUTCOME CHECKER
# ─────────────────────────────────────────────
def check_outcome(future_df: pd.DataFrame, signal: str,
                  entry: float, target: float, stop_loss: float) -> dict:
    for _, row in future_df.iterrows():
        high = float(row["High"])
        low  = float(row["Low"])
        if signal == "BUY":
            if low <= stop_loss:
                return {"outcome": "LOSS", "pnl_pct": round((stop_loss - entry) / entry * 100, 2)}
            if high >= target:
                return {"outcome": "WIN",  "pnl_pct": round((target - entry) / entry * 100, 2)}
        else:
            if high >= stop_loss:
                return {"outcome": "LOSS", "pnl_pct": round((entry - stop_loss) / entry * 100, 2)}
            if low <= target:
                return {"outcome": "WIN",  "pnl_pct": round((entry - target) / entry * 100, 2)}

    last_close = float(future_df.iloc[-1]["Close"])
    pnl_pct = round((last_close - entry) / entry * 100, 2) if signal == "BUY" \
              else round((entry - last_close) / entry * 100, 2)
    return {"outcome": "TIMEOUT", "pnl_pct": pnl_pct}


# ─────────────────────────────────────────────
# BACKTEST ONE STOCK
# ─────────────────────────────────────────────
def backtest_symbol(symbol: str, trend_map: dict) -> list:
    ticker = symbol + ".NS"
    trades = []

    try:
        df = yf.download(ticker, period=BT_PERIOD, interval=BT_INTERVAL,
                         progress=False, auto_adjust=True)
        if df is None or len(df) < EMA_LONG + MAX_LOOKAHEAD + 20:
            return []

        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)

        # FIX 5 - Filter to NSE market hours only
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize("UTC").tz_convert(IST)
        else:
            df.index = df.index.tz_convert(IST)
        df = df.between_time("09:15", "15:30")

        if len(df) < EMA_LONG + MAX_LOOKAHEAD + 20:
            return []

        # Compute indicators on full dataset once
        df["RSI"]       = calculate_rsi(df["Close"], RSI_PERIOD)
        _, _, macdh     = calculate_macd(df["Close"])
        df["MACDh"]     = macdh
        bbu, _, bbl     = calculate_bbands(df["Close"], BB_PERIOD, BB_STD)
        df["BB_Upper"]  = bbu
        df["BB_Lower"]  = bbl
        df["EMA_Short"] = calculate_ema(df["Close"], EMA_SHORT)
        df["EMA_Long"]  = calculate_ema(df["Close"], EMA_LONG)
        df["ADX"], _, _ = calculate_adx(df, ADX_PERIOD)
        df["ATR"]       = calculate_atr(df, ATR_PERIOD)
        df.dropna(inplace=True)

        # Improvement 2 — build per-stock daily trend map once per symbol
        stock_trend_map = build_stock_trend_map(symbol)

        warmup = EMA_LONG + 10
        total  = len(df)

        # FIX 1 - while loop so we can skip i forward after each trade
        i = warmup
        while i < total - MAX_LOOKAHEAD:
            df_slice = df.iloc[:i + 1]
            sig      = compute_signal(df_slice)

            if sig is None:
                i += 1
                continue

            candle_time = df.index[i]

            # FIX 2 - Nifty trend filter
            nifty_trend = get_nifty_trend_for_date(trend_map, candle_time)
            if sig["signal"] == "BUY"  and nifty_trend == "bearish":
                i += 1
                continue
            if sig["signal"] == "SELL" and nifty_trend == "bullish":
                i += 1
                continue

            # Improvement 2 — Per-stock daily trend filter
            stock_trend = get_nifty_trend_for_date(stock_trend_map, candle_time)
            if sig["signal"] == "BUY"  and stock_trend == "bearish":
                i += 1
                continue
            if sig["signal"] == "SELL" and stock_trend == "bullish":
                i += 1
                continue

            future_df = df.iloc[i + 1: i + 1 + MAX_LOOKAHEAD]
            if len(future_df) == 0:
                i += 1
                continue

            result = check_outcome(
                future_df,
                sig["signal"],
                sig["entry"],
                sig["target"],
                sig["stop_loss"],
            )

            trade_time_str = candle_time.strftime("%d %b %Y %H:%M") \
                             if hasattr(candle_time, "strftime") else str(candle_time)

            trades.append({
                "symbol":    symbol,
                "time":      trade_time_str,
                "signal":    sig["signal"],
                "score":     sig["score"],
                "entry":     sig["entry"],
                "target":    sig["target"],
                "stop_loss": sig["stop_loss"],
                "outcome":   result["outcome"],
                "pnl_pct":   result["pnl_pct"],
                "reasons":   sig["reasons"],
            })

            # FIX 1 - skip forward after trade to avoid overlapping signals
            i += MAX_LOOKAHEAD

    except Exception as e:
        console.print(f"[red]  Error backtesting {symbol}: {e}[/red]")

    return trades


# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────
def compute_stats(trades: list) -> dict:
    if not trades:
        return {}

    total    = len(trades)
    wins     = [t for t in trades if t["outcome"] == "WIN"]
    losses   = [t for t in trades if t["outcome"] == "LOSS"]
    timeouts = [t for t in trades if t["outcome"] == "TIMEOUT"]

    # FIX 4 - Win rate only counts decided trades (WIN + LOSS), not TIMEOUT
    decided  = [t for t in trades if t["outcome"] != "TIMEOUT"]
    win_rate = round(len(wins) / max(len(decided), 1) * 100, 1)

    avg_win  = round(sum(t["pnl_pct"] for t in wins)   / max(len(wins),   1), 2)
    avg_loss = round(sum(t["pnl_pct"] for t in losses) / max(len(losses), 1), 2)
    net_pnl  = round(sum(t["pnl_pct"] for t in trades), 2)
    best     = max(trades, key=lambda x: x["pnl_pct"])
    worst    = min(trades, key=lambda x: x["pnl_pct"])

    return {
        "total":    total,
        "wins":     len(wins),
        "losses":   len(losses),
        "timeouts": len(timeouts),
        "decided":  len(decided),
        "win_rate": win_rate,
        "avg_win":  avg_win,
        "avg_loss": avg_loss,
        "net_pnl":  net_pnl,
        "best":     best,
        "worst":    worst,
    }


# ─────────────────────────────────────────────
# PRINT TRADE LOG
# ─────────────────────────────────────────────
def print_trade_log(trades: list):
    if not trades:
        return
    table = Table(
        title="Trade Log",
        box=box.SIMPLE_HEAVY, show_lines=False,
        header_style="bold white on grey23",
    )
    table.add_column("Symbol",    style="bold", width=12)
    table.add_column("Time",      width=18)
    table.add_column("Signal",    justify="center", width=7)
    table.add_column("Entry Rs",  justify="right",  width=10)
    table.add_column("Target Rs", justify="right",  width=10)
    table.add_column("SL Rs",     justify="right",  width=10)
    table.add_column("Outcome",   justify="center", width=9)
    table.add_column("P&L %",     justify="right",  width=8)
    table.add_column("Score",     justify="center", width=7)

    for t in trades:
        sig_c = "green" if t["signal"] == "BUY" else "red"
        if t["outcome"] == "WIN":
            out_c, out_sym = "bold green", "[W] WIN"
        elif t["outcome"] == "LOSS":
            out_c, out_sym = "bold red",   "[L] LOSS"
        else:
            out_c, out_sym = "yellow",     "[T] TIME"
        pnl_c = "green" if t["pnl_pct"] >= 0 else "red"
        table.add_row(
            t["symbol"],
            t["time"],
            Text(t["signal"], style=sig_c),
            f"{t['entry']:,.2f}",
            f"{t['target']:,.2f}",
            f"{t['stop_loss']:,.2f}",
            Text(out_sym, style=out_c),
            Text(f"{t['pnl_pct']:+.2f}%", style=pnl_c),
            f"{t['score']}/13",
        )
    console.print(table)


# ─────────────────────────────────────────────
# PRINT SUMMARY REPORT
# ─────────────────────────────────────────────
def print_summary(stats: dict, symbol_label: str):
    if not stats:
        console.print("[red]No trades found in backtest period.[/red]")
        return
    net_c = "bold green" if stats["net_pnl"] >= 0 else "bold red"
    console.print()
    console.rule(f"[bold cyan]>> Backtest Summary - {symbol_label}[/bold cyan]")
    console.print(f"  Period        : Last 60 days | 15-min | Market hours only (9:15-15:30 IST)")
    console.print(f"  Total Trades  : {stats['total']}")
    console.print(f"  Wins          : [green]{stats['wins']}[/green]")
    console.print(f"  Losses        : [red]{stats['losses']}[/red]")
    console.print(f"  Timeouts      : [yellow]{stats['timeouts']}[/yellow]  (excluded from win rate)")
    console.print(f"  Decided       : {stats['decided']}  (wins + losses only)")
    console.print(f"  Win Rate      : {'[green]' if stats['win_rate'] >= 50 else '[red]'}{stats['win_rate']}%[/]  (of decided trades)")
    console.print(f"  Avg Win       : [green]+{stats['avg_win']}%[/green]")
    console.print(f"  Avg Loss      : [red]{stats['avg_loss']}%[/red]")
    console.print(f"  Net P&L       : [{net_c}]{stats['net_pnl']:+.2f}%[/]  (timeouts counted at last close)")
    console.print()
    console.print(f"  Best Trade    : [green]{stats['best']['symbol']} {stats['best']['signal']} "
                  f"on {stats['best']['time']} → +{stats['best']['pnl_pct']}%[/green]")
    console.print(f"  Worst Trade   : [red]{stats['worst']['symbol']} {stats['worst']['signal']} "
                  f"on {stats['worst']['time']} → {stats['worst']['pnl_pct']}%[/red]")
    console.print()


# ─────────────────────────────────────────────
# PRINT PER-STOCK TABLE
# ─────────────────────────────────────────────
def print_per_stock_table(all_trades: list):
    from collections import defaultdict
    by_symbol = defaultdict(list)
    for t in all_trades:
        by_symbol[t["symbol"]].append(t)

    table = Table(
        title="Per-Stock Performance",
        box=box.ROUNDED, show_lines=True,
        header_style="bold white on dark_blue",
    )
    table.add_column("Symbol",   style="bold", width=14)
    table.add_column("Trades",   justify="center", width=8)
    table.add_column("Wins",     justify="center", width=6)
    table.add_column("Losses",   justify="center", width=8)
    table.add_column("Timeouts", justify="center", width=10)
    table.add_column("Win Rate", justify="center", width=10)
    table.add_column("Avg Win",  justify="right",  width=10)
    table.add_column("Avg Loss", justify="right",  width=10)
    table.add_column("Net P&L",  justify="right",  width=10)

    symbol_stats = []
    for sym, trades in by_symbol.items():
        s = compute_stats(trades)
        s["symbol"] = sym
        symbol_stats.append(s)
    symbol_stats.sort(key=lambda x: x["net_pnl"], reverse=True)

    for s in symbol_stats:
        wr_c  = "green" if s["win_rate"] >= 50 else "red"
        pnl_c = "green" if s["net_pnl"]  >= 0  else "red"
        table.add_row(
            s["symbol"],
            str(s["total"]),
            Text(str(s["wins"]),          style="green"),
            Text(str(s["losses"]),        style="red"),
            Text(str(s["timeouts"]),      style="yellow"),
            Text(f"{s['win_rate']}%",     style=wr_c),
            Text(f"+{s['avg_win']}%",     style="green"),
            Text(f"{s['avg_loss']}%",     style="red"),
            Text(f"{s['net_pnl']:+.2f}%", style=pnl_c),
        )
    console.print(table)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args         = sys.argv[1:]
    symbols      = [s.upper() for s in args] if args else NIFTY100_SYMBOLS
    symbol_label = ", ".join(symbols) if args else "Nifty 100"

    console.rule("[bold cyan]>> Swing Trading Backtester (Fixed)[/bold cyan]")
    console.print(
        f"[dim]Symbols: {symbol_label} | Period: 60d | Interval: 15m | "
        f"Min Score: {MIN_SCORE}/13 | R/R: {RISK_REWARD}:1[/dim]"
    )
    console.print(
        "[dim]Fixes active: trade-reset · nifty-filter · stock-trend-filter · "
        "candle-direction · timeout-excluded · market-hours[/dim]\n"
    )

    # Build Nifty trend map once upfront
    trend_map  = build_nifty_trend_map()
    all_trades = []
    total_syms = len(symbols)

    for idx, symbol in enumerate(symbols, 1):
        console.print(f"[dim]  Backtesting [{idx}/{total_syms}] {symbol}...[/dim]", end="\r")
        trades = backtest_symbol(symbol, trend_map)
        all_trades.extend(trades)

    console.print(" " * 60, end="\r")

    if not all_trades:
        console.print("[red]No trades generated. Try lowering MIN_SCORE in config.[/red]")
        return

    print_trade_log(all_trades)

    if len(symbols) > 1:
        print_per_stock_table(all_trades)

    overall_stats = compute_stats(all_trades)
    print_summary(overall_stats, symbol_label)

    console.print(
        "[bold yellow]** Disclaimer:[/bold yellow] "
        "Past performance does not guarantee future results. "
        "Backtest results are simulated and do not account for "
        "slippage, brokerage, or live market conditions.\n"
    )


if __name__ == "__main__":
    main()