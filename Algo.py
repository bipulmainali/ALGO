"""
BM's Trading Algo — revised, research-oriented stock scanner.

Important:
- Signals are analytical outputs, not guarantees or personalized investment advice.
- Yahoo Finance data may be delayed, incomplete, adjusted, or unavailable.
- Backtests are historical simulations and do not establish future profitability.
- This app is long-only in its backtest. SELL means bearish/avoid context, not a short entry.
"""

from __future__ import annotations

import math
import re
import time
import warnings
from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

NY = ZoneInfo("America/New_York")
APP_TITLE = "BM's Trading Algo"
DEFAULT_WATCHLIST = (
    "SPY, QQQ, TSLA, AAPL, MSFT, NVDA, AMZN, META, GOOG, PLTR, "
    "AMD, SOXX, ADP, UNH, MU, NBIS, IREN"
)
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1wk"]
MATRIX_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="auto")

def _extract_matrix_cell(event: object) -> tuple[int, int] | None:
    """Normalize Streamlit dataframe selection coordinates across versions."""
    selection = (
        event.get("selection") if isinstance(event, dict)
        else getattr(event, "selection", None)
    )
    cells = (
        selection.get("cells", []) if isinstance(selection, dict)
        else getattr(selection, "cells", []) if selection is not None else []
    )
    if not cells:
        return None

    cell = cells[0]
    row_idx = col_idx = None
    if isinstance(cell, dict):
        row_idx = cell.get("row")
        col_idx = cell.get("column", cell.get("col"))
    elif isinstance(cell, (tuple, list)) and len(cell) >= 2:
        row_idx, col_idx = cell[0], cell[1]
    elif isinstance(cell, str):
        parts = [part.strip().strip("()[]") for part in re.split(r"[,;:]", cell.strip())]
        if len(parts) >= 2:
            row_idx, col_idx = parts[0], parts[1]
    else:
        row_idx = getattr(cell, "row", None)
        col_idx = getattr(cell, "column", getattr(cell, "col", None))

    try:
        row_idx = int(row_idx)
    except (TypeError, ValueError):
        return None
    if isinstance(col_idx, str) and not col_idx.strip().isdigit():
        return row_idx, col_idx
    try:
        return row_idx, int(col_idx)
    except (TypeError, ValueError):
        return None


def handle_matrix_selection() -> None:
    """Record a pending destination; apply it before sidebar widgets render."""
    cell = _extract_matrix_cell(st.session_state.get("matrix_table"))
    if cell is None:
        return

    row_idx, col_idx = cell
    row_tickers = st.session_state.get("_matrix_row_tickers", [])
    columns = st.session_state.get("_matrix_columns", [])
    if not (0 <= row_idx < len(row_tickers)):
        return

    if isinstance(col_idx, str):
        if col_idx not in columns:
            return
        col_idx = columns.index(col_idx)
    if not (0 <= col_idx < len(columns)):
        return

    timeframe = str(columns[col_idx])
    if timeframe not in MATRIX_TIMEFRAMES:
        return

    st.session_state["_pending_matrix_navigation"] = {
        "ticker": str(row_tickers[row_idx]),
        "timeframe": timeframe,
    }

st.session_state.setdefault("selected_tf", "1h")
st.session_state.setdefault("selected_analysis_ticker", "")

_pending_navigation = st.session_state.pop("_pending_matrix_navigation", None)
if isinstance(_pending_navigation, dict):
    _ticker = str(_pending_navigation.get("ticker", "")).strip().upper()
    _timeframe = str(_pending_navigation.get("timeframe", ""))
    if _ticker and _timeframe in MATRIX_TIMEFRAMES:
        st.session_state["selected_analysis_ticker"] = _ticker
        st.session_state["selected_tf"] = _timeframe

st.title("📈 BM's Trading Algo")
st.caption("Rule-based entry, exit, and wait signals with technical indicators — not a trade execution system.")

# ---------------------------------------------------------------------------
# Data retrieval and normalization
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def download_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            tickers=ticker, interval=interval, period=period, auto_adjust=True,
            progress=False, threads=False, group_by="column", timeout=20, prepost=False,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        chosen_level = None
        for level in range(raw.columns.nlevels):
            vals = {str(v).lower() for v in raw.columns.get_level_values(level)}
            if {"open", "high", "low", "close"}.issubset(vals):
                chosen_level = level
                break
        if chosen_level is None:
            return pd.DataFrame()
        raw.columns = raw.columns.get_level_values(chosen_level)

    raw.columns = [str(c).title() for c in raw.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not set(required).issubset(raw.columns):
        return pd.DataFrame()

    raw = raw[required].copy()
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    try:
        if raw.index.tz is not None:
            raw.index = raw.index.tz_convert(NY).tz_localize(None)
    except Exception:
        try:
            raw.index = raw.index.tz_localize(None)
        except Exception:
            pass

    for col in required:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    raw = raw[(raw["High"] >= raw["Low"]) & (raw["Volume"] >= 0)]
    return raw


@st.cache_data(ttl=300, show_spinner=False)
def download_vix(period: str = "2y") -> pd.Series:
    try:
        raw = yf.download(
            "^VIX", interval="1d", period=period, auto_adjust=True,
            progress=False, threads=False, timeout=20
        )
        if raw is None or raw.empty:
            return pd.Series(dtype=float)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if "Close" not in raw:
            return pd.Series(dtype=float)
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is not None:
            idx = idx.tz_convert(NY).tz_localize(None)
        s = pd.Series(pd.to_numeric(raw["Close"], errors="coerce").to_numpy(), index=idx)
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep="last")].dropna().sort_index()
        return s
    except Exception:
        return pd.Series(dtype=float)


def filter_unclosed_candles(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    if df.empty:
        return df
    now = datetime.now(NY).replace(tzinfo=None)
    out = df.copy()

    if tf in ("5m", "15m", "1h", "4h"):
        durations = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
        mins = durations[tf]
        last_ts = pd.Timestamp(out.index[-1])
        session_close = last_ts.normalize() + pd.Timedelta(hours=16)
        expected_end = min(last_ts + pd.Timedelta(minutes=mins), session_close)
        if last_ts.time() >= dt_time(16, 0):
            expected_end = last_ts + pd.Timedelta(minutes=mins)
        if now < expected_end:
            out = out.iloc[:-1]
    elif tf == "1d":
        last_date = pd.Timestamp(out.index[-1]).date()
        today = now.date()
        if last_date >= today and now.time() < dt_time(16, 10):
            out = out.iloc[:-1]
    elif tf == "1wk":
        last_ts = pd.Timestamp(out.index[-1])
        bar_week_monday = last_ts.normalize() - pd.Timedelta(days=last_ts.weekday())
        current_week_monday = pd.Timestamp(now.date()).normalize() - pd.Timedelta(days=now.weekday())
        friday_close = current_week_monday + pd.Timedelta(days=4, hours=16, minutes=10)
        if bar_week_monday >= current_week_monday and now < friday_close:
            out = out.iloc[:-1]
    return out


def resample_4h_regular_session(hourly: pd.DataFrame) -> pd.DataFrame:
    if hourly.empty:
        return hourly
    df = hourly.copy()
    idx = pd.DatetimeIndex(df.index)
    df["_date"] = idx.date
    minutes_from_open = (idx.hour * 60 + idx.minute) - (9 * 60 + 30)
    df["_bucket"] = np.floor_divide(np.maximum(minutes_from_open, 0), 240)
    regular = (minutes_from_open >= 0) & (minutes_from_open < 390)
    df = df.loc[regular]
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    grouped = df.groupby(["_date", "_bucket"], sort=True)
    out = grouped.agg(
        Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"),
        Close=("Close", "last"), Volume=("Volume", "sum"),
    )
    first_idx = df.groupby(["_date", "_bucket"], sort=True).apply(lambda x: x.index[0])
    out.index = pd.DatetimeIndex([pd.Timestamp(v) for v in first_idx.to_list()])
    out.index.name = "Datetime"
    return out.sort_index()


def get_price_data(ticker: str, tf: str) -> pd.DataFrame:
    settings = {
        "5m": ("5m", "60d"), "15m": ("15m", "60d"), "1h": ("1h", "730d"),
        "4h": ("1h", "730d"), "1d": ("1d", "5y"), "1wk": ("1wk", "10y"),
    }
    interval, period = settings[tf]
    data = download_ohlcv(ticker.upper(), interval, period)
    if data.empty:
        return data
    if tf == "4h":
        data = resample_4h_regular_session(data)
    data = filter_unclosed_candles(data, tf)
    if data.empty:
        return data

    vix = download_vix()
    if not vix.empty:
        vix_available = vix.copy()
        vix_available.index = vix_available.index + pd.Timedelta(days=1)

        def _day_keys(values) -> np.ndarray:
            idx = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce", utc=True))
            idx = idx.tz_convert(NY).tz_localize(None).normalize()
            return idx.to_numpy(dtype="datetime64[ns]").astype("int64")

        price_days = _day_keys(data.index)
        vix_days = _day_keys(vix_available.index)
        vix_values = pd.to_numeric(vix_available.to_numpy(), errors="coerce").astype(float)

        valid_vix = (vix_days != np.iinfo(np.int64).min) & np.isfinite(vix_values)
        vix_days = vix_days[valid_vix]
        vix_values = vix_values[valid_vix]
        order = np.argsort(vix_days, kind="stable")
        vix_days = vix_days[order]
        vix_values = vix_values[order]

        matched = np.full(len(price_days), np.nan, dtype=float)
        valid_price = price_days != np.iinfo(np.int64).min
        positions = np.searchsorted(vix_days, price_days[valid_price], side="right") - 1
        has_match = positions >= 0
        valid_indices = np.flatnonzero(valid_price)
        matched[valid_indices[has_match]] = vix_values[positions[has_match]]
        data["VIX"] = matched
    else:
        data["VIX"] = np.nan
    return data

# ---------------------------------------------------------------------------
# Indicator calculations 
# ---------------------------------------------------------------------------

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=max(1, int(length)), adjust=False, min_periods=max(1, int(length))).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(max(1, int(length)), min_periods=max(1, int(length))).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0)
    return result

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [(df["High"] - df["Low"]).abs(),
         (df["High"] - prev_close).abs(),
         (df["Low"] - prev_close).abs()], axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = atr(df, length)
    atr_smooth = tr.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean() / atr_smooth
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean() / atr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

def add_indicators(df: pd.DataFrame, tf: str, p: dict) -> pd.DataFrame:
    out = df.copy()
    out["ATR"] = atr(out, p["atr_len"])
    out["ATR_Pct"] = 100 * out["ATR"] / out["Close"].replace(0, np.nan)
    out["RSI"] = rsi(out["Close"], p["rsi_len"])
    out["SMA_Fast"] = sma(out["Close"], p["sma_fast"])
    out["SMA_Slow"] = sma(out["Close"], p["sma_slow"])
    out["EMA_Fast"] = ema(out["Close"], p["ema_fast"])
    out["EMA_Slow"] = ema(out["Close"], p["ema_slow"])
    out["Vol_SMA"] = sma(out["Volume"], p["rvol_len"])
    out["RVOL"] = out["Volume"] / out["Vol_SMA"].replace(0, np.nan)
    out["ADX"] = adx(out, p["adx_len"])
    
    # Bollinger Bands
    bb_mid = sma(out["Close"], p["bb_len"])
    bb_std = out["Close"].rolling(p["bb_len"], min_periods=p["bb_len"]).std(ddof=0)
    out["BB_Mid"] = bb_mid
    out["BB_Upper"] = bb_mid + p["bb_std"] * bb_std
    out["BB_Lower"] = bb_mid - p["bb_std"] * bb_std
    
    # Experimental: Keltner Channels & TTM Squeeze
    kc_mid = sma(out["Close"], 20)
    kc_upper = kc_mid + 1.5 * out["ATR"]
    kc_lower = kc_mid - 1.5 * out["ATR"]
    out["Squeeze_On"] = (out["BB_Upper"] < kc_upper) & (out["BB_Lower"] > kc_lower)
    
    # Experimental: RSI Bullish Divergence
    out["RSI_Div"] = (out["Close"] < out["Close"].shift(10)) & (out["RSI"] > out["RSI"].shift(10))
    
    # Experimental: Chandelier Trailing Stop (Long)
    highest_high = out["High"].rolling(22, min_periods=22).max()
    out["Chandelier_Exit"] = highest_high - 3 * out["ATR"]

    macd_fast = ema(out["Close"], 12)
    macd_slow = ema(out["Close"], 26)
    macd_line = macd_fast - macd_slow
    macd_signal = ema(macd_line, 9)
    out["MACD"] = macd_line
    out["MACD_Signal"] = macd_signal
    out["MACD_Hist"] = macd_line - macd_signal

    # QAQC Fix: VWAP is strictly an intraday session metric.
    if tf in ("5m", "15m"):
        typical = (out["High"] + out["Low"] + out["Close"]) / 3
        dates = pd.DatetimeIndex(out.index).normalize()
        cum_pv = (typical * out["Volume"]).groupby(dates).cumsum()
        cum_vol = out["Volume"].groupby(dates).cumsum().replace(0, np.nan)
        out["VWAP"] = cum_pv / cum_vol
    else:
        out["VWAP"] = np.nan

    out["Trend_Up"] = out["SMA_Fast"] > out["SMA_Slow"]
    out["Trend_Down"] = out["SMA_Fast"] < out["SMA_Slow"]
    return out.replace([np.inf, -np.inf], np.nan)

# ---------------------------------------------------------------------------
# Strategy, structural levels, and trade plans
# ---------------------------------------------------------------------------

class TradingScanner:
    def __init__(self, ticker: str, tf: str, params: dict, vix_max: float = 30):
        self.ticker = ticker.upper().strip()
        self.tf = tf
        self.p = params
        self.vix_max = vix_max
        self.is_index = self.ticker in {"SPY", "QQQ", "DIA", "IWM", "SPX", "NDX", "^GSPC", "^NDX"}

    def signal(self, row: pd.Series, tf: str | None = None) -> dict:
        tf = tf or self.tf
        required = ["Close", "ATR", "RSI", "ADX", "RVOL", "BB_Upper", "BB_Lower",
                    "SMA_Fast", "SMA_Slow", "MACD_Hist"]
        if tf in ("5m", "15m"):
            required.extend(["EMA_Fast", "EMA_Slow", "VWAP"])
            
        if any(c not in row.index or not np.isfinite(row[c]) for c in required):
            return {"action": "INSUFFICIENT DATA", "reason": "Required indicators are not available."}

        trending = row["ADX"] >= self.p["adx_thresh"]
        volume_ok = row["RVOL"] >= self.p["rvol_min"]
        
        # QAQC Fix: Missing VIX now explicitly flags a WAIT rather than passing.
        vix_val = row.get("VIX", np.nan)
        if not np.isfinite(vix_val):
            vix_ok = False
            vix_reason = "WAIT — MISSING VIX DATA"
        elif vix_val >= self.vix_max:
            vix_ok = False
            vix_reason = f"WAIT — HIGH VIX ({vix_val:.1f} >= {self.vix_max})"
        else:
            vix_ok = True
            vix_reason = ""

        not_chasing = row["Close"] <= row["BB_Upper"]
        not_panic_selling = row["Close"] >= row["BB_Lower"]

        if tf in ("5m", "15m"):
            up = row["EMA_Fast"] > row["EMA_Slow"] and row["Close"] > row["VWAP"]
            down = row["EMA_Fast"] < row["EMA_Slow"] and row["Close"] < row["VWAP"]
            long_momentum = (self.p["rsi_os"] <= row["RSI"] <= self.p["rsi_ob"])
            short_momentum = row["RSI"] >= self.p["short_rsi_min"]
            buy = up and long_momentum and trending and volume_ok and not_chasing and vix_ok
            sell = down and short_momentum and trending and volume_ok and not_panic_selling
        else:
            up = row["SMA_Fast"] > row["SMA_Slow"]
            down = row["SMA_Fast"] < row["SMA_Slow"]
            long_momentum = (self.p["swing_rsi_min"] <= row["RSI"] <= self.p["rsi_os"])
            short_momentum = row["RSI"] >= self.p["short_rsi_min"]
            # QAQC clarification: MACD Histogram is used as a Momentum Regime filter (>0), not a 1-bar crossover trigger.
            buy = up and long_momentum and row["MACD_Hist"] > 0 and trending and volume_ok and not_chasing and vix_ok
            sell = down and short_momentum and row["MACD_Hist"] < 0 and trending and not_panic_selling

        if buy:
            reason = "Trend, momentum, regime, volume, and price filters pass."
            return {"action": "BUY", "reason": reason}
            
        if sell:
            return {"action": "SELL / AVOID", "reason": "Bearish conditions; this is not a short-entry recommendation."}
        if not trending:
            return {"action": "CHOP / RANGE", "reason": f"ADX below {self.p['adx_thresh']}."}
        if not volume_ok:
            return {"action": "WAIT", "reason": f"RVOL below {self.p['rvol_min']:.2f}."}
        if not vix_ok:
            return {"action": vix_reason, "reason": vix_reason}
            
        return {"action": "WAIT", "reason": "No complete setup."}

    def support_resistance(self, df: pd.DataFrame, lookback: int = 150, order: int = 5):
        if df.empty or len(df) < (order * 2 + 5):
            return [], []
        close = df["Close"].tail(lookback).to_numpy(dtype=float)
        current = close[-1]
        supports, resistances = [], []
        for i in range(order, len(close) - order):
            window = close[i - order:i + order + 1]
            if close[i] == np.max(window) and np.sum(window == close[i]) == 1 and close[i] > current:
                resistances.append(float(close[i]))
            if close[i] == np.min(window) and np.sum(window == close[i]) == 1 and close[i] < current:
                supports.append(float(close[i]))
        def dedupe(values):
            result = []
            for value in sorted(values):
                if not result or abs(value - result[-1]) / max(abs(value), 1e-9) > 0.002:
                    result.append(value)
            return result
        supports = sorted(dedupe(supports), reverse=True)[:3]
        resistances = sorted(dedupe(resistances))[:3]
        return supports, resistances

    # QAQC Fix: Trade Plan stripped of account sizing inputs. Purely informational levels.
    def trade_plan(self, row: pd.Series, supports: list[float], is_intraday: bool, slippage_bps: float = 5.0) -> dict:
        price = float(row["Close"])
        atr_value = float(row["ATR"])
        sl_mult = self.p["intraday_sl_atr"] if is_intraday else self.p["swing_sl_atr"]
        tp_mult = self.p["intraday_tp_atr"] if is_intraday else self.p["swing_tp_atr"]
        
        stop = price - sl_mult * atr_value
        target = price + tp_mult * atr_value
        entry = price * (1 + slippage_bps / 10000)
        stop_distance = max(entry - stop, 0)
        
        pullback = next((s for s in supports if s < price and s > stop), np.nan)
        rr = (target - entry) / stop_distance if stop_distance > 0 else np.nan
        
        return {
            "price": price, "entry": entry, "stop": stop, "target": target,
            "tp2": price + (4.0 if not is_intraday else 2.0) * atr_value,
            "tp3": price + (6.0 if not is_intraday else 3.0) * atr_value,
            "pullback": pullback, "rr": rr,
        }

# ---------------------------------------------------------------------------
# Backtest Logic
# ---------------------------------------------------------------------------
def backtest_long_only(df: pd.DataFrame, scanner: TradingScanner, tf: str, hold_bars: int, slippage_bps: float) -> dict:
    trades = []
    if df.empty or len(df) < max(scanner.p["sma_slow"] + 10, 80):
        return {"trades": pd.DataFrame(), "setups": 0, "wins": 0, "win_rate": np.nan, "profit_factor": np.nan, "expectancy_r": np.nan, "max_drawdown_per_share": np.nan}

    i = max(scanner.p["sma_slow"], 50)
    while i < len(df) - 1:
        row = df.iloc[i]
        sig = scanner.signal(row, tf)
        if sig["action"] != "BUY":
            i += 1
            continue

        entry_bar = i + 1
        entry = float(df.iloc[entry_bar]["Open"]) * (1 + slippage_bps / 10000)
        atr_value = float(row["ATR"])
        if not np.isfinite(entry) or not np.isfinite(atr_value) or atr_value <= 0:
            i += 1
            continue
            
        is_intraday = tf in ("5m", "15m")
        sl_mult = scanner.p["intraday_sl_atr"] if is_intraday else scanner.p["swing_sl_atr"]
        tp_mult = scanner.p["intraday_tp_atr"] if is_intraday else scanner.p["swing_tp_atr"]
        stop = entry - sl_mult * atr_value
        target = entry + tp_mult * atr_value
        last_bar = min(len(df) - 1, entry_bar + hold_bars - 1)
        exit_price = None
        exit_i = last_bar

        for j in range(entry_bar, last_bar + 1):
            bar = df.iloc[j]
            if float(bar["Low"]) <= stop and float(bar["High"]) >= target:
                exit_price = stop * (1 - slippage_bps / 10000)
                exit_i = j
                break
            if float(bar["Low"]) <= stop:
                exit_price = stop * (1 - slippage_bps / 10000)
                exit_i = j
                break
            if float(bar["High"]) >= target:
                exit_price = target * (1 - slippage_bps / 10000)
                exit_i = j
                break

        if exit_price is None:
            exit_price = float(df.iloc[exit_i]["Close"]) * (1 - slippage_bps / 10000)
            
        net = (exit_price - entry)
        initial_risk = max(entry - stop, 1e-9)
        trades.append({
            "Net P/L per Share": net,
            "R Multiple": net / initial_risk,
            "Win": net > 0,
        })
        i = exit_i + 1

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {"trades": tdf, "setups": 0, "wins": 0, "win_rate": np.nan, "profit_factor": np.nan, "expectancy_r": np.nan, "max_drawdown_per_share": np.nan}
        
    pnl = tdf["Net P/L per Share"].astype(float)
    wins = int((pnl > 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else np.nan)
    equity = pnl.cumsum()
    peak = equity.cummax()
    max_dd = float((equity - peak).min())
    
    return {
        "trades": tdf, "setups": len(tdf), "wins": wins,
        "win_rate": 100 * wins / len(tdf),
        "profit_factor": pf, "expectancy_r": float(tdf["R Multiple"].mean()),
        "max_drawdown_per_share": max_dd, 
    }

# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def format_money(x):
    return "N/A" if x is None or not np.isfinite(x) else f"${x:,.2f}"

def safe_indicator_row(df: pd.DataFrame, tf: str | None = None) -> pd.Series | None:
    if df.empty:
        return None
    required = [
        "Close", "ATR", "RSI", "ADX", "RVOL", "SMA_Fast", "SMA_Slow",
        "BB_Upper", "BB_Lower", "MACD_Hist",
    ]
    if tf in ("5m", "15m"):
        required.extend(["EMA_Fast", "EMA_Slow", "VWAP"])
    if any(col not in df.columns for col in required):
        return None
    valid = df[required].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None
    return df.loc[valid.index[-1]].copy()

def make_params() -> dict:
    return {
        "ema_fast": 9, "ema_slow": 21, "sma_fast": 50, "sma_slow": 200,
        "rsi_len": 14, "rsi_os": 50.0, "rsi_ob": 65.0, "swing_rsi_min": 35.0,
        "short_rsi_min": 50.0, "adx_thresh": 20.0, "adx_len": 14, "rvol_len": 20,
        "rvol_min": 1.0, "bb_len": 20, "bb_std": 2.0, "atr_len": 14,
        "intraday_sl_atr": 1.5, "intraday_tp_atr": 1.0, "swing_sl_atr": 2.0, "swing_tp_atr": 2.0,
    }

def analyze_ticker(ticker: str, tf: str, p: dict) -> tuple[pd.DataFrame, TradingScanner, pd.Series | None, dict]:
    raw = get_price_data(ticker, tf)
    scanner = TradingScanner(ticker, tf, p)
    if raw.empty:
        return pd.DataFrame(), scanner, None, {"action": "NO DATA", "reason": "No market data returned."}
    indicators = add_indicators(raw, tf, p)
    row = safe_indicator_row(indicators, tf)
    if row is None:
        return indicators, scanner, None, {"action": "INSUFFICIENT HISTORY", "reason": "Not enough completed candles."}
    return indicators, scanner, row, scanner.signal(row, tf)

params = make_params()

st.caption(
    "Signal-focused mode: review ENTRY, EXIT / AVOID, or WAIT decisions and the indicators behind them. "
    "Account sizing, commissions, slippage, and backtest controls are hidden."
)

# ---------------------------------------------------------------------------
# Matrix UI
# ---------------------------------------------------------------------------

st.subheader("🌐 Multi-Timeframe Trend & Entry Matrix")
st.caption("Select any timeframe cell to load that ticker’s analysis below. The matrix stays visible and does not change when you adjust the analysis timeframe.")
tickers_text = st.text_area("Comma-separated ticker symbols", value=DEFAULT_WATCHLIST, key="matrix_tickers")
scan = st.button("Generate Matrix", type="primary")

if scan:
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip()))
    if not tickers:
        st.warning("Enter at least one ticker.")
    else:
        rows = []
        total = len(tickers) * len(MATRIX_TIMEFRAMES)
        done = 0
        progress = st.progress(0)
        status = st.empty()
        for ticker in tickers:
            row = {"Ticker": ticker, "Price": np.nan}
            trend_points = 0.0
            trend_weight = 0.0
            entry_count = 0
            valid_count = 0
            for tf in MATRIX_TIMEFRAMES:
                done += 1
                status.text(f"Evaluating {ticker} on {tf} ({done}/{total})")
                progress.progress(done / total)
                df, scanner, latest, sig = analyze_ticker(ticker, tf, params)
                if latest is None:
                    row[tf] = sig["action"]
                    continue
                if not np.isfinite(row["Price"]):
                    row["Price"] = round(float(latest["Close"]), 2)
                valid_count += 1
                
                weight = {"5m": 1, "15m": 1, "1h": 2, "4h": 3, "1d": 4}[tf]
                trend_up = bool(latest["SMA_Fast"] > latest["SMA_Slow"])
                trend_down = bool(latest["SMA_Fast"] < latest["SMA_Slow"])
                if trend_up:
                    trend_points += weight
                trend_weight += weight
                if sig["action"] == "BUY":
                    row[tf] = "BUY"
                    entry_count += 1
                elif sig["action"] == "SELL / AVOID":
                    row[tf] = "SELL / AVOID"
                elif "WAIT — MISSING VIX" in sig["action"]:
                    row[tf] = "MISSING VIX"
                elif "WAIT — HIGH VIX" in sig["action"]:
                    row[tf] = "HIGH VIX"
                elif sig["action"] == "CHOP / RANGE":
                    row[tf] = "CHOP"
                else:
                    row[tf] = f"{'Bull' if trend_up else 'Bear' if trend_down else 'Flat'} trend; {sig['action']}"
            
            row["Valid TFs"] = valid_count
            row["Trend Alignment"] = f"{100 * trend_points / trend_weight:.0f}%" if trend_weight else "N/A"
            row["BUY Confirmations"] = f"{entry_count}/{valid_count}" if valid_count else "N/A"
            if entry_count >= 3 and row["Trend Alignment"] != "N/A" and float(row["Trend Alignment"].strip("%")) >= 70:
                row["Summary"] = "High alignment — review setup"
            elif entry_count >= 1:
                row["Summary"] = "Some entry confirmation"
            else:
                row["Summary"] = "No confirmed BUY"
            rows.append(row)
        status.success("Matrix complete.")
        st.session_state["_matrix_result"] = pd.DataFrame(rows)

matrix = st.session_state.get("_matrix_result")

if isinstance(matrix, pd.DataFrame) and not matrix.empty:
    def matrix_cell_style(value: object) -> str:
        text_value = str(value).upper()
        if text_value == "BUY" or text_value.startswith("BULL TREND"):
            return "background-color: #214D3A; color: #E8FFF1; font-weight: 600;"
        if text_value == "SELL / AVOID" or text_value.startswith("BEAR TREND"):
            return "background-color: #632F35; color: #FFECEF; font-weight: 600;"
        if text_value == "CHOP" or text_value.startswith("FLAT TREND"):
            return "background-color: #424852; color: #F1F3F5;"
        if "INSUFFICIENT" in text_value or "NO DATA" in text_value or "MISSING VIX" in text_value:
            return "background-color: #30343B; color: #C4CAD2;"
        if "WAIT" in text_value or "HOLD" in text_value or "N/A" in text_value:
            return "background-color: #424852; color: #F1F3F5;"
        return "background-color: #424852; color: #F1F3F5;"

    st.session_state["_matrix_row_tickers"] = matrix["Ticker"].astype(str).tolist()
    st.session_state["_matrix_columns"] = [str(column) for column in matrix.columns]
    styled_matrix = matrix.style.map(matrix_cell_style, subset=MATRIX_TIMEFRAMES)
    st.caption("Select a cell under 5m, 15m, 1h, 4h, or 1d to open that ticker and timeframe in the analysis panel below.")
    st.dataframe(
        styled_matrix, use_container_width=True, hide_index=True,
        on_select=handle_matrix_selection, selection_mode="single-cell",
        key="matrix_table", height=min(680, 38 + 35 * (len(matrix) + 1)),
    )
    st.caption("Legend: green = BUY/bullish trend; red = SELL/avoid/bearish trend; slate = wait, chop, or mixed signals; dark gray = insufficient data.")

# ---------------------------------------------------------------------------
# Deep analysis panel
# ---------------------------------------------------------------------------

analysis_ticker = st.session_state.get("selected_analysis_ticker", "").strip().upper()

if analysis_ticker:
    st.markdown("---")
    st.subheader("🔎 Selected Ticker Deep-Dive")
    control_left, control_right = st.columns([1, 2])
    with control_left:
        ticker = st.text_input("Analysis ticker", key="selected_analysis_ticker").strip().upper()
    with control_right:
        selected_tf = st.select_slider("Analysis timeframe", options=TIMEFRAMES, key="selected_tf")

    if ticker:
        st.header(f"🔎 {ticker} — {selected_tf} Deep-Dive")
        
        with st.spinner(f"Analyzing {ticker} ({selected_tf})..."):
            df, scanner, latest, sig = analyze_ticker(ticker, selected_tf, params)
            
        if df.empty:
            st.error(f"No usable market data found for {ticker}.")
        elif latest is None:
            st.warning(sig["reason"])
        else:
            price = float(latest["Close"])
            is_intraday = selected_tf in ("5m", "15m", "1h", "4h")
            supports, resistances = scanner.support_resistance(df)
            vix_value = latest.get("VIX", np.nan)
            
            st.subheader("🤖 Trading Signal")
            action = sig["action"]
            if action == "BUY":
                st.success("🟢 ENTRY SIGNAL — BUY")
            elif action == "SELL / AVOID":
                st.error("🔴 EXIT / AVOID — bearish conditions")
            elif action.startswith("CHOP"):
                st.warning("⚪ WAIT — RANGE / CHOP")
            elif action.startswith("WAIT — MISSING VIX"):
                st.error("⚪ WAIT — MISSING VIX DATA")
            else:
                st.info(f"🟡 WAIT — {action}")

            st.write(f"**Why:** {sig['reason']}")

            trend_up = bool(latest["SMA_Fast"] > latest["SMA_Slow"])
            trend_down = bool(latest["SMA_Fast"] < latest["SMA_Slow"])
            trend_label = "Bullish" if trend_up else "Bearish" if trend_down else "Neutral"
            macd_value = latest.get("MACD_Hist", np.nan)
            vwap_value = latest.get("VWAP", np.nan)
            
            indicator_cols = st.columns(4)
            indicator_cols[0].metric("Last Close", format_money(price))
            indicator_cols[1].metric("SMA Trend", trend_label)
            indicator_cols[2].metric("RSI (14)", f"{latest['RSI']:.1f}")
            indicator_cols[3].metric("ADX (14)", f"{latest['ADX']:.1f}")

            indicator_cols2 = st.columns(6)
            indicator_cols2[0].metric("Relative Volume", f"{latest['RVOL']:.2f}x")
            indicator_cols2[1].metric("ATR", format_money(latest["ATR"]))
            indicator_cols2[2].metric("ATR Pct", f"{latest.get('ATR_Pct', 0):.2f}%")
            indicator_cols2[3].metric("MACD Hist", f"{macd_value:.3f}" if np.isfinite(macd_value) else "N/A")
            indicator_cols2[4].metric("VWAP", format_money(vwap_value) if np.isfinite(vwap_value) else "N/A")
            indicator_cols2[5].metric("Prior VIX Close", f"{vix_value:.1f}" if np.isfinite(vix_value) else "N/A")

            # -------------------------------------------------------------------
            # BACKTEST & TRADE PLAN INTEGRATION
            # -------------------------------------------------------------------
            st.markdown("---")
            st.subheader("🧪 Historical Strategy Backtest")
            bt_results = backtest_long_only(df, scanner, selected_tf, hold_bars=10, slippage_bps=5.0)
            bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
            bt_col1.metric("Historical Setups", bt_results["setups"])
            bt_col2.metric("Win Rate", f"{bt_results['win_rate']:.1f}%" if np.isfinite(bt_results['win_rate']) else "N/A")
            bt_col3.metric("Profit Factor", f"{bt_results['profit_factor']:.2f}" if np.isfinite(bt_results['profit_factor']) else "N/A")
            bt_col4.metric("Max DD ($/sh)", f"${abs(bt_results['max_drawdown_per_share']):.2f}" if np.isfinite(bt_results['max_drawdown_per_share']) else "N/A")
            
            st.markdown("---")
            st.subheader("🤖 Informational Trade Plan Zones")
            st.caption("ATR-based reference levels. Always independently verify sizing and risk budget.")
            tp = scanner.trade_plan(latest, supports, is_intraday=(selected_tf in ("5m", "15m")), slippage_bps=5.0)

            plan_col1, plan_col2 = st.columns(2)
            with plan_col1:
                st.write(f"**Reference Entry:** {format_money(tp['entry'])}")
                st.write(f"**Structural Pullback Level:** {format_money(tp['pullback'])}")
                st.write(f"**Hard Stop Loss:** {format_money(tp['stop'])}")
                st.write(f"**Experimental Trailing Stop:** {format_money(latest.get('Chandelier_Exit', np.nan))}")
            with plan_col2:
                st.write(f"**TP 1 (Base Target):** {format_money(tp['target'])}")
                st.write(f"**TP 2 (Extension):** {format_money(tp['tp2'])}")
                st.write(f"**TP 3 (Runner):** {format_money(tp['tp3'])}")
                st.write(f"**Risk/Reward (TP1):** {tp['rr']:.2f}x" if np.isfinite(tp['rr']) else "**Risk/Reward:** N/A")

            # -------------------------------------------------------------------

            st.markdown("---")
            st.subheader("🧱 Confirmed Historical Levels")
            level_count = max(len(supports), len(resistances), 1)
            support_text = [format_money(x) for x in supports]
            resistance_text = [format_money(x) for x in resistances]
            support_text += ["—"] * (level_count - len(support_text))
            resistance_text += ["—"] * (level_count - len(resistance_text))
            levels_df = pd.DataFrame({"Support": support_text, "Resistance": resistance_text})
            
            levels_style = (levels_df.style
                .set_properties(**{"font-size": "14px", "font-weight": "400", "color": "#F1F3F5", "background-color": "#20242B", "text-align": "left"})
                .set_table_styles([
                    {"selector": "th", "props": [("font-size", "14px"), ("font-weight", "600"), ("color", "#F1F3F5"), ("background-color", "#303640"), ("text-align", "left")]},
                    {"selector": "td", "props": [("padding", "8px 12px")]}
                ])
            )
            st.dataframe(levels_style, hide_index=True, use_container_width=True, height=min(220, 38 + 35 * (level_count + 1)))

            st.subheader(f"Price Action — {ticker} ({selected_tf})")
            visible = 250 if selected_tf in ("5m", "15m") else 180 if selected_tf in ("1h", "4h") else 120
            chart = df.tail(visible).copy()
            chart = chart[~chart.index.duplicated(keep="last")].sort_index()
            fig = go.Figure()

            fig.add_trace(go.Candlestick(x=chart.index, open=chart["Open"], high=chart["High"], low=chart["Low"], close=chart["Close"], name=ticker))

            intraday_view = selected_tf in ("5m", "15m", "1h", "4h")

            if intraday_view and len(chart.index) > 1:
                expected_minutes = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}[selected_tf]
                time_deltas = chart.index.to_series().diff()
                gap_rows = time_deltas > pd.Timedelta(minutes=expected_minutes * 1.5)
                for column in ("VWAP", "EMA_Fast", "EMA_Slow", "BB_Upper", "BB_Lower", "Chandelier_Exit"):
                    if column in chart.columns:
                        chart.loc[gap_rows, column] = np.nan

            if selected_tf in ("5m", "15m"):
                if "VWAP" in chart.columns:
                    fig.add_trace(go.Scatter(x=chart.index, y=chart["VWAP"], name="Session VWAP", mode="lines", connectgaps=False, line=dict(width=1.5, dash="dot", color="yellow")))
                if "EMA_Fast" in chart.columns:
                    fig.add_trace(go.Scatter(x=chart.index, y=chart["EMA_Fast"], name=f"{params['ema_fast']} EMA", mode="lines", connectgaps=False, line=dict(width=1.2, color="cyan")))
                if "EMA_Slow" in chart.columns:
                    fig.add_trace(go.Scatter(x=chart.index, y=chart["EMA_Slow"], name=f"{params['ema_slow']} EMA", mode="lines", connectgaps=False, line=dict(width=1.2, color="magenta")))
            else:
                for column, label, color in (("SMA_Fast", f"{params['sma_fast']} SMA", "orange"), ("SMA_Slow", f"{params['sma_slow']} SMA", "blue")):
                    if column in chart.columns:
                        fig.add_trace(go.Scatter(x=chart.index, y=chart[column], name=label, mode="lines", connectgaps=False, line=dict(width=1.4, color=color)))

            for column, label in (("BB_Upper", "Upper Bollinger"), ("BB_Lower", "Lower Bollinger")):
                if column in chart.columns:
                    fig.add_trace(go.Scatter(x=chart.index, y=chart[column], name=label, mode="lines", connectgaps=False, line=dict(width=1, dash="dash", color="rgba(255,255,255,0.3)")))

            if "Chandelier_Exit" in chart.columns:
                fig.add_trace(go.Scatter(x=chart.index, y=chart["Chandelier_Exit"], name="Chandelier Trailing Stop (Experimental)", mode="lines", connectgaps=False, line=dict(width=1.5, dash="dot", color="red")))

            for level in supports:
                fig.add_hline(y=level, line_dash="dot", line_color="green", opacity=0.5)
            for level in resistances:
                fig.add_hline(y=level, line_dash="dot", line_color="red", opacity=0.5)

            fig.update_layout(template="plotly_dark", height=680, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))

            if intraday_view:
                fig.update_xaxes(type="date", rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])

            st.plotly_chart(fig, use_container_width=True)
            st.caption("Prices are auto-adjusted Yahoo Finance history. Intraday charts compress weekends and closed hours.")

st.markdown("---")
st.caption("BM's Trading Algo • Educational/research tool • Not a broker, not an execution engine, and not investment advice.")
