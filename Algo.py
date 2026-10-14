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
            buy = up and long_momentum and row["MACD_Hist"] > 0 and trending and volume_ok and not_chasing
