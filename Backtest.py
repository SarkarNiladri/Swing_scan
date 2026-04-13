# -*- coding: utf-8 -*-
"""
Swing Trading Strategy - Backtester
=====================================
How it works:
  1. Downloads 1 year of 15-min historical data for each stock
  2. Replays candle by candle (walk-forward simulation)
  3. At each candle, applies the same signal logic as the live scanner
  4. When a signal fires - records entry, target, stop loss
  5. Scans future candles to see what was hit first: target or stop loss
  6. Compiles full performance report per stock and overall

Usage:
  Backtest all Nifty 100 stocks:
      python backtest.py

  Backtest a single stock:
      python backtest.py RELIANCE

  Backtest a few stocks:
      python backtest.py RELIANCE TCS INFY HDFCBANK

Requirements:
    pip install yfinance pandas numpy rich pytz
"""

import warnings
warnings.filterwarnings("ignore")
import Symbols
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

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
IST     = pytz.timezone("Asia/Kolkata")

# ─────────────────────────────────────────────
# CONFIGURATION  (keep in sync with scanner)
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
SR_ZONE_PCT   = 0.010
RISK_REWARD   = 2.0
MIN_SCORE     = 8

# Backtest window: yfinance allows up to 60 days for 15m data
BT_PERIOD     = "60d"
BT_INTERVAL   = "15m"

# Max candles to look ahead for target/SL hit (26 candles ≈ 1 trading day)
MAX_LOOKAHEAD = 26 * 5   # up to 5 trading days

# ─────────────────────────────────────────────
# NIFTY 100 SYMBOLS
# ─────────────────────────────────────────────
NIFTY100_SYMBOLS = Symbols.symbols


# ─────────────────────────────────────────────
# S/R DETECTION
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
# CANDLESTICK PATTERNS
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
    hammer        = lower_wick3 >= 2 * body3 and upper_wick3 <= 0.1 * range3 and body3 > 0
    bull_engulf   = c2c < o2 and c3c > o3 and o3 <= c2c and c3c >= o2
    morning_star  = (c1c < o1 and abs(c2c - o2) < 0.3 * (h2 - l2 if h2 != l2 else 0.001)
                     and c3c > o3 and c3c > (o1 + c1c) / 2)
    shooting_star = upper_wick3 >= 2 * body3 and lower_wick3 <= 0.1 * range3 and body3 > 0
    bear_engulf   = c2c > o2 and c3c < o3 and o3 >= c2c and c3c <= o2
    evening_star  = (c1c > o1 and abs(c2c - o2) < 0.3 * (h2 - l2 if h2 != l2 else 0.001)
                     and c3c < o3 and c3c < (o1 + c1c) / 2)
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
# SIGNAL LOGIC  (applied on a slice of df)
# ─────────────────────────────────────────────
def compute_signal(df_slice: pd.DataFrame) -> dict | None:
    """
    Given a slice of OHLCV data with indicators already computed,
    evaluate the signal at the LAST candle of the slice.
    Returns signal dict or None.
    """
    if len(df_slice) < 5:
        return None

    last = df_slice.iloc[-1]
    prev = df_slice.iloc[-2]

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

    

    # Volume filter
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
        stop_loss = round(close - 1.5 * atr, 2)
        target    = round(close + (close - stop_loss) * RISK_REWARD, 2)
    else:
        stop_loss = round(close + 1.5 * atr, 2)
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
    """
    Walk forward through future candles.
    Returns outcome: WIN / LOSS / TIMEOUT and P&L %.
    """
    for _, row in future_df.iterrows():
        high = float(row["High"])
        low  = float(row["Low"])

        if signal == "BUY":
            if low <= stop_loss:
                pnl_pct = round((stop_loss - entry) / entry * 100, 2)
                return {"outcome": "LOSS", "pnl_pct": pnl_pct}
            if high >= target:
                pnl_pct = round((target - entry) / entry * 100, 2)
                return {"outcome": "WIN", "pnl_pct": pnl_pct}
        else:  # SELL
            if high >= stop_loss:
                pnl_pct = round((entry - stop_loss) / entry * 100, 2)
                return {"outcome": "LOSS", "pnl_pct": pnl_pct}
            if low <= target:
                pnl_pct = round((entry - target) / entry * 100, 2)
                return {"outcome": "WIN", "pnl_pct": pnl_pct}

    # Neither hit within lookahead window
    last_close = float(future_df.iloc[-1]["Close"])
    if signal == "BUY":
        pnl_pct = round((last_close - entry) / entry * 100, 2)
    else:
        pnl_pct = round((entry - last_close) / entry * 100, 2)
    return {"outcome": "TIMEOUT", "pnl_pct": pnl_pct}


# ─────────────────────────────────────────────
# BACKTEST ONE STOCK
# ─────────────────────────────────────────────
def backtest_symbol(symbol: str) -> list:
    ticker = symbol + ".NS"
    trades = []

    try:
        df = yf.download(ticker, period=BT_PERIOD, interval=BT_INTERVAL,
                         progress=False, auto_adjust=True)
        if df is None or len(df) < EMA_LONG + MAX_LOOKAHEAD + 20:
            return []

        df.columns  = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.dropna(inplace=True)

        # Compute all indicators on full dataset once
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

        # Minimum warmup candles needed before we start checking signals
        warmup   = EMA_LONG + 10
        total    = len(df)
        in_trade = False   # Only one trade at a time per stock

        for i in range(warmup, total - MAX_LOOKAHEAD):
            if in_trade:
                continue

            df_slice = df.iloc[:i + 1]
            sig      = compute_signal(df_slice)

            if sig is None:
                continue

            # Look ahead to check outcome
            future_df = df.iloc[i + 1: i + 1 + MAX_LOOKAHEAD]
            if len(future_df) == 0:
                continue

            result = check_outcome(
                future_df,
                sig["signal"],
                sig["entry"],
                sig["target"],
                sig["stop_loss"],
            )

            trade_time = df.index[i]
            if hasattr(trade_time, "strftime"):
                trade_time_str = trade_time.strftime("%d %b %Y %H:%M")
            else:
                trade_time_str = str(trade_time)

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

            # After a signal, skip ahead to avoid overlapping trades
            in_trade = True
            # Reset after lookahead window
            # (simple approach: allow new signals after gap)

    except Exception as e:
        console.print(f"[red]  Error backtesting {symbol}: {e}[/red]")

    return trades


# ─────────────────────────────────────────────
# SUMMARY STATS
# ─────────────────────────────────────────────
def compute_stats(trades: list) -> dict:
    if not trades:
        return {}

    total     = len(trades)
    wins      = [t for t in trades if t["outcome"] == "WIN"]
    losses    = [t for t in trades if t["outcome"] == "LOSS"]
    timeouts  = [t for t in trades if t["outcome"] == "TIMEOUT"]

    win_rate  = round(len(wins) / total * 100, 1)
    avg_win   = round(sum(t["pnl_pct"] for t in wins)   / max(len(wins),  1), 2)
    avg_loss  = round(sum(t["pnl_pct"] for t in losses) / max(len(losses),1), 2)
    net_pnl   = round(sum(t["pnl_pct"] for t in trades), 2)
    best      = max(trades, key=lambda x: x["pnl_pct"])
    worst     = min(trades, key=lambda x: x["pnl_pct"])

    return {
        "total":     total,
        "wins":      len(wins),
        "losses":    len(losses),
        "timeouts":  len(timeouts),
        "win_rate":  win_rate,
        "avg_win":   avg_win,
        "avg_loss":  avg_loss,
        "net_pnl":   net_pnl,
        "best":      best,
        "worst":     worst,
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
    table.add_column("Symbol",   style="bold", width=12)
    table.add_column("Time",     width=18)
    table.add_column("Signal",   justify="center", width=7)
    table.add_column("Entry Rs",  justify="right",  width=10)
    table.add_column("Target Rs", justify="right",  width=10)
    table.add_column("SL Rs",     justify="right",  width=10)
    table.add_column("Outcome",  justify="center", width=9)
    table.add_column("P&L %",    justify="right",  width=8)
    table.add_column("Score",    justify="center", width=7)

    for t in trades:
        sig_c = "green" if t["signal"] == "BUY" else "red"

        if t["outcome"] == "WIN":
            out_c, out_sym = "bold green", "[W] WIN"
        elif t["outcome"] == "LOSS":
            out_c, out_sym = "bold red", "[L] LOSS"
        else:
            out_c, out_sym = "yellow", "[T] TIME"

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
    console.print(f"  Period          : Last 60 days | 15-min candles")
    console.print(f"  Total Trades    : {stats['total']}")
    console.print(f"  Wins            : [green]{stats['wins']}[/green]")
    console.print(f"  Losses          : [red]{stats['losses']}[/red]")
    console.print(f"  Timeouts        : [yellow]{stats['timeouts']}[/yellow]  (neither target nor SL in 5 days)")
    console.print(f"  Win Rate        : {'[green]' if stats['win_rate'] >= 50 else '[red]'}{stats['win_rate']}%[/]")
    console.print(f"  Avg Win         : [green]+{stats['avg_win']}%[/green]")
    console.print(f"  Avg Loss        : [red]{stats['avg_loss']}%[/red]")
    console.print(f"  Net P&L         : [{net_c}]{stats['net_pnl']:+.2f}%[/]")
    console.print()
    console.print(f"  Best Trade      : [green]{stats['best']['symbol']} {stats['best']['signal']} "
                  f"on {stats['best']['time']} → +{stats['best']['pnl_pct']}%[/green]")
    console.print(f"  Worst Trade     : [red]{stats['worst']['symbol']} {stats['worst']['signal']} "
                  f"on {stats['worst']['time']} → {stats['worst']['pnl_pct']}%[/red]")
    console.print()


# ─────────────────────────────────────────────
# PRINT PER-STOCK SUMMARY TABLE (multi-stock)
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
    table.add_column("Symbol",    style="bold", width=14)
    table.add_column("Trades",    justify="center", width=8)
    table.add_column("Wins",      justify="center", width=6)
    table.add_column("Losses",    justify="center", width=8)
    table.add_column("Win Rate",  justify="center", width=10)
    table.add_column("Avg Win",   justify="right",  width=10)
    table.add_column("Avg Loss",  justify="right",  width=10)
    table.add_column("Net P&L",   justify="right",  width=10)

    # Sort by net P&L descending
    symbol_stats = []
    for sym, trades in by_symbol.items():
        s = compute_stats(trades)
        s["symbol"] = sym
        symbol_stats.append(s)
    symbol_stats.sort(key=lambda x: x["net_pnl"], reverse=True)

    for s in symbol_stats:
        wr_c   = "green" if s["win_rate"] >= 50 else "red"
        pnl_c  = "green" if s["net_pnl"] >= 0  else "red"
        table.add_row(
            s["symbol"],
            str(s["total"]),
            Text(str(s["wins"]),    style="green"),
            Text(str(s["losses"]),  style="red"),
            Text(f"{s['win_rate']}%", style=wr_c),
            Text(f"+{s['avg_win']}%",  style="green"),
            Text(f"{s['avg_loss']}%",  style="red"),
            Text(f"{s['net_pnl']:+.2f}%", style=pnl_c),
        )

    console.print(table)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Determine symbols to backtest
    args = sys.argv[1:]
    if args:
        symbols      = [s.upper() for s in args]
        symbol_label = ", ".join(symbols)
    else:
        symbols      = NIFTY100_SYMBOLS
        symbol_label = "Nifty 100"

    console.rule("[bold cyan]>> Swing Trading Backtester[/bold cyan]")
    console.print(
        f"[dim]Symbols: {symbol_label} | Period: 60 days | "
        f"Interval: 15m | Min Score: {MIN_SCORE}/13 | R/R: {RISK_REWARD}:1[/dim]\n"
    )

    all_trades = []
    total_syms = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        console.print(f"[dim]  Backtesting [{i}/{total_syms}] {symbol}...[/dim]", end="\r")
        trades = backtest_symbol(symbol)
        all_trades.extend(trades)

    console.print(" " * 60, end="\r")

    if not all_trades:
        console.print("[red]No trades generated. Try a longer period or lower MIN_SCORE.[/red]")
        return

    # Print trade log
    print_trade_log(all_trades)

    # Print per-stock table (only for multi-stock runs)
    if len(symbols) > 1:
        print_per_stock_table(all_trades)

    # Print overall summary
    overall_stats = compute_stats(all_trades)
    print_summary(overall_stats, symbol_label)

    console.print("[bold yellow]** Disclaimer:[/bold yellow] "
                  "Past performance does not guarantee future results. "
                  "Backtest results are simulated and do not account for slippage, "
                  "brokerage, or live market conditions.\n")


if __name__ == "__main__":
    main()