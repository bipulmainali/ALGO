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

# Apply matrix-cell navigation before sidebar widgets are instantiated. Streamlit does
# not allow changing an existing widget's session state after that widget is rendered.
_pending_matrix_navigation = st.session_state.pop("_pending_matrix_navigation", None)
if _pending_matrix_navigation:
    st.session_state["operating_mode"] = "Single Ticker Deep-Dive"
    st.session_state["selected_tf"] = _pending_matrix_navigation["timeframe"]
    st.session_state["deep_dive_ticker"] = _pending_matrix_navigation["ticker"]
    st.session_state["_auto_run_deep_dive"] = True

st.title("📈 BM's Trading Algo")
st.caption("Rule-based screening, trade planning, and historical research — not a trade execution system.")


# ---------------------------------------------------------------------------
# Data retrieval and normalization
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def download_ohlcv(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Cached Yahoo Finance OHLCV download with normalized columns/index."""
    try:
        raw = yf.download(
            tickers=ticker,
            interval=interval,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
            timeout=20,
            prepost=False,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    # yfinance versions may return a MultiIndex even for one ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        # Select the level containing OHLCV labels, rather than assuming level 0.
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


def is_intraday_tf(tf: str) -> bool:
    return tf in ("5m", "15m", "1h", "4h")


def filter_unclosed_candles(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Conservatively remove the currently forming bar where it can be identified."""
    if df.empty:
        return df
    now = datetime.now(NY).replace(tzinfo=None)
    out = df.copy()

    if tf in ("5m", "15m", "1h", "4h"):
        durations = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
        mins = durations[tf]
        last_ts = pd.Timestamp(out.index[-1])
        # US regular session ends at 16:00 ET. Bars near the close may be shorter.
        session_close = last_ts.normalize() + pd.Timedelta(hours=16)
        expected_end = min(last_ts + pd.Timedelta(minutes=mins), session_close)
        # If the last timestamp is outside regular hours, use its nominal interval.
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
        # Yahoo weekly bars generally begin on Monday. Treat the current week as open
        # until Friday's regular session is over.
        bar_week_monday = last_ts.normalize() - pd.Timedelta(days=last_ts.weekday())
        current_week_monday = pd.Timestamp(now.date()).normalize() - pd.Timedelta(days=now.weekday())
        friday_close = current_week_monday + pd.Timedelta(days=4, hours=16, minutes=10)
        if bar_week_monday >= current_week_monday and now < friday_close:
            out = out.iloc[:-1]
    return out


def resample_4h_regular_session(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly regular-session bars into session-anchored 4-hour groups.
    Grouping is anchored to 09:30 ET and never combines bars from different dates.
    """
    if hourly.empty:
        return hourly
    df = hourly.copy()
    idx = pd.DatetimeIndex(df.index)
    df["_date"] = idx.date
    minutes_from_open = (idx.hour * 60 + idx.minute) - (9 * 60 + 30)
    df["_bucket"] = np.floor_divide(np.maximum(minutes_from_open, 0), 240)
    # Ignore timestamps outside regular session; avoids accidentally mixing pre/post-market.
    regular = (minutes_from_open >= 0) & (minutes_from_open < 390)
    df = df.loc[regular]
    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    grouped = df.groupby(["_date", "_bucket"], sort=True)
    out = grouped.agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    )
    # Give each 4h candle the timestamp of its first constituent bar.
    first_idx = df.groupby(["_date", "_bucket"], sort=True).apply(lambda x: x.index[0])
    out.index = pd.DatetimeIndex([pd.Timestamp(v) for v in first_idx.to_list()])
    out.index.name = "Datetime"
    return out.sort_index()


def get_price_data(ticker: str, tf: str) -> pd.DataFrame:
    settings = {
        "5m": ("5m", "60d"),
        "15m": ("15m", "60d"),
        "1h": ("1h", "730d"),
        "4h": ("1h", "730d"),
        "1d": ("1d", "5y"),
        "1wk": ("1wk", "10y"),
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

    # Use only the previous completed daily VIX close for a conservative, no-look-ahead
    # risk context. Never backfill missing VIX observations from the future.
    vix = download_vix()
    if not vix.empty:
        # Shift each daily VIX observation forward one calendar day, then perform an
        # as-of join. This means a Friday close becomes available on Saturday and is
        # used for Monday decisions; it is never used for Friday intraday decisions.
        vix_available = vix.copy()
        vix_available.index = vix_available.index + pd.Timedelta(days=1)

        # merge_asof requires identical key dtypes. Yahoo Finance can return a
        # timezone-aware index in some environments and a timezone-naive index in
        # others, so normalize BOTH join keys to timezone-naive datetime64[ns].
        def _naive_normalized_index(values) -> pd.DatetimeIndex:
            idx = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
            if idx.tz is not None:
                idx = idx.tz_convert(NY).tz_localize(None)
            return idx.normalize()

        left = pd.DataFrame({
            "_date": _naive_normalized_index(data.index),
            "_order": np.arange(len(data)),
        })
        right = pd.DataFrame({
            "_date": _naive_normalized_index(vix_available.index),
            "VIX": pd.to_numeric(vix_available.to_numpy(), errors="coerce"),
        }).dropna(subset=["_date"]).sort_values("_date")
        left = left.dropna(subset=["_date"]).sort_values("_date")
        merged = pd.merge_asof(left, right, on="_date", direction="backward")
        data["VIX"] = merged.sort_values("_order")["VIX"].reindex(
            np.arange(len(data))
        ).to_numpy()
    else:
        data["VIX"] = np.nan
    return data


# ---------------------------------------------------------------------------
# Indicator calculations (pandas/numpy only; no pandas-ta or SciPy dependency)
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
         (df["Low"] - prev_close).abs()],
        axis=1,
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
    bb_mid = sma(out["Close"], p["bb_len"])
    bb_std = out["Close"].rolling(p["bb_len"], min_periods=p["bb_len"]).std(ddof=0)
    out["BB_Mid"] = bb_mid
    out["BB_Upper"] = bb_mid + p["bb_std"] * bb_std
    out["BB_Lower"] = bb_mid - p["bb_std"] * bb_std
    macd_fast = ema(out["Close"], 12)
    macd_slow = ema(out["Close"], 26)
    macd_line = macd_fast - macd_slow
    macd_signal = ema(macd_line, 9)
    out["MACD"] = macd_line
    out["MACD_Signal"] = macd_signal
    out["MACD_Hist"] = macd_line - macd_signal

    # Intraday VWAP resets each calendar day and is based only on bars in that session.
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    if tf in ("5m", "15m", "1h", "4h"):
        dates = pd.DatetimeIndex(out.index).normalize()
        cum_pv = (typical * out["Volume"]).groupby(dates).cumsum()
        cum_vol = out["Volume"].groupby(dates).cumsum().replace(0, np.nan)
        out["VWAP"] = cum_pv / cum_vol
    else:
        out["VWAP"] = np.nan

    # Directional movement supports clear trend confirmation, not just ADX strength.
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
            required += ["EMA_Fast", "EMA_Slow", "VWAP"]
        if any(c not in row.index or not np.isfinite(row[c]) for c in required):
            return {"action": "INSUFFICIENT DATA", "reason": "Required indicators are not available."}

        trending = row["ADX"] >= self.p["adx_thresh"]
        volume_ok = row["RVOL"] >= self.p["rvol_min"]
        vix_ok = (not np.isfinite(row.get("VIX", np.nan))) or (row["VIX"] < self.vix_max)
        # The previous completed VIX close is used. Missing VIX is disclosed, not backfilled.
        not_chasing = row["Close"] <= row["BB_Upper"]
        not_panic_selling = row["Close"] >= row["BB_Lower"]

        if tf in ("5m", "15m"):
            up = row["EMA_Fast"] > row["EMA_Slow"] and row["Close"] > row["VWAP"]
            down = row["EMA_Fast"] < row["EMA_Slow"] and row["Close"] < row["VWAP"]
            # A pullback in an established trend; avoid buying very overbought momentum.
            long_momentum = self.p["rsi_os"] <= row["RSI"] <= self.p["rsi_ob"]
            short_momentum = row["RSI"] >= self.p["short_rsi_min"]
            buy = up and long_momentum and trending and volume_ok and not_chasing and vix_ok
            sell = down and short_momentum and trending and volume_ok and not_panic_selling
        else:
            up = row["SMA_Fast"] > row["SMA_Slow"]
            down = row["SMA_Fast"] < row["SMA_Slow"]
            # MACD confirms direction; RSI must indicate a pullback rather than exhaustion.
            long_momentum = self.p["swing_rsi_min"] <= row["RSI"] <= self.p["rsi_os"]
            short_momentum = row["RSI"] >= self.p["short_rsi_min"]
            buy = up and long_momentum and row["MACD_Hist"] > 0 and trending and volume_ok and not_chasing and vix_ok
            sell = down and short_momentum and row["MACD_Hist"] < 0 and trending and not_panic_selling

        if buy:
            return {"action": "BUY", "reason": "Trend, momentum, regime, volume, and price filters pass."}
        if sell:
            return {"action": "SELL / AVOID", "reason": "Bearish conditions; this is not a short-entry recommendation."}
        if not trending:
            return {"action": "CHOP / RANGE", "reason": f"ADX below {self.p['adx_thresh']}."}
        if not volume_ok:
            return {"action": "WAIT", "reason": f"RVOL below {self.p['rvol_min']:.2f}."}
        if not vix_ok:
            return {"action": "WAIT — HIGH VIX", "reason": f"Previous completed VIX close is at/above {self.vix_max}."}
        return {"action": "WAIT", "reason": "No complete setup."}

    def support_resistance(self, df: pd.DataFrame, lookback: int = 150, order: int = 5):
        """Return past confirmed swing levels, excluding unconfirmed right-edge pivots."""
        if df.empty or len(df) < (order * 2 + 5):
            return [], []
        close = df["Close"].tail(lookback).to_numpy(dtype=float)
        current = close[-1]
        supports, resistances = [], []
        # A pivot at i is confirmed only after `order` bars to its right exist.
        for i in range(order, len(close) - order):
            window = close[i - order:i + order + 1]
            if close[i] == np.max(window) and np.sum(window == close[i]) == 1 and close[i] > current:
                resistances.append(float(close[i]))
            if close[i] == np.min(window) and np.sum(window == close[i]) == 1 and close[i] < current:
                supports.append(float(close[i]))
        # De-duplicate levels that are very close (0.2%).
        def dedupe(values):
            result = []
            for value in sorted(values):
                if not result or abs(value - result[-1]) / max(abs(value), 1e-9) > 0.002:
                    result.append(value)
            return result
        supports = sorted(dedupe(supports), reverse=True)[:3]
        resistances = sorted(dedupe(resistances))[:3]
        return supports, resistances

    def trade_plan(self, row: pd.Series, supports: list[float], is_intraday: bool,
                   account_equity: float, risk_pct: float, max_position_pct: float,
                   slippage_bps: float) -> dict:
        price = float(row["Close"])
        atr_value = float(row["ATR"])
        sl_mult = self.p["intraday_sl_atr"] if is_intraday else self.p["swing_sl_atr"]
        tp_mult = self.p["intraday_tp_atr"] if is_intraday else self.p["swing_tp_atr"]
        stop = price - sl_mult * atr_value
        target = price + tp_mult * atr_value
        # Conservative immediate entry assumption: estimated slippage is added to entry.
        entry = price * (1 + slippage_bps / 10000)
        stop_distance = max(entry - stop, 0)
        risk_dollars = account_equity * risk_pct / 100
        shares_risk = int(risk_dollars / stop_distance) if stop_distance > 0 else 0
        shares_cap = int((account_equity * max_position_pct / 100) / entry) if entry > 0 else 0
        shares = max(0, min(shares_risk, shares_cap))
        pullback = next((s for s in supports if s < price and s > stop), np.nan)
        rr = (target - entry) / stop_distance if stop_distance > 0 else np.nan
        return {
            "price": price, "entry": entry, "stop": stop, "target": target,
            "tp2": price + (4.0 if not is_intraday else 2.0) * atr_value,
            "tp3": price + (6.0 if not is_intraday else 3.0) * atr_value,
            "pullback": pullback, "risk_dollars": risk_dollars,
            "shares": shares, "position_value": shares * entry,
            "risk_if_stop": shares * stop_distance, "rr": rr,
        }


# ---------------------------------------------------------------------------
# Backtest: long-only, next-bar open, one position at a time per ticker.
# ---------------------------------------------------------------------------

def backtest_long_only(df: pd.DataFrame, scanner: TradingScanner, tf: str,
                       hold_bars: int, slippage_bps: float, commission_per_share: float) -> dict:
    trades = []
    if df.empty or len(df) < max(scanner.p["sma_slow"] + 10, 80):
        return {"trades": pd.DataFrame(), "setups": 0, "closed": 0, "wins": 0,
                "losses": 0, "win_rate": np.nan, "profit_factor": np.nan,
                "expectancy_r": np.nan, "max_drawdown_pct": np.nan}

    i = max(scanner.p["sma_slow"], 50)
    while i < len(df) - 1:
        row = df.iloc[i]
        sig = scanner.signal(row, tf)
        if sig["action"] != "BUY":
            i += 1
            continue

        setups_index = i
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
        exit_reason = "TIME EXIT"

        for j in range(entry_bar, last_bar + 1):
            bar = df.iloc[j]
            hit_stop = float(bar["Low"]) <= stop
            hit_target = float(bar["High"]) >= target
            if hit_stop and hit_target:
                # OHLC bars cannot reveal which happened first; assume stop first.
                exit_price = stop * (1 - slippage_bps / 10000)
                exit_i, exit_reason = j, "STOP (both touched; conservative)"
                break
            if hit_stop:
                exit_price = stop * (1 - slippage_bps / 10000)
                exit_i, exit_reason = j, "STOP"
                break
            if hit_target:
                exit_price = target * (1 - slippage_bps / 10000)
                exit_i, exit_reason = j, "TARGET"
                break

        if exit_price is None:
            exit_price = float(df.iloc[exit_i]["Close"]) * (1 - slippage_bps / 10000)
        shares = 1
        gross = (exit_price - entry) * shares
        costs = 2 * commission_per_share * shares
        net = gross - costs
        initial_risk = max(entry - stop, 1e-9) * shares + costs
        trades.append({
            "Signal Time": df.index[setups_index],
            "Entry Time": df.index[entry_bar],
            "Exit Time": df.index[exit_i],
            "Entry": entry, "Stop": stop, "Target": target, "Exit": exit_price,
            "Exit Reason": exit_reason, "Net P/L per Share": net,
            "R Multiple": net / initial_risk, "Win": net > 0,
            "Bars Held": exit_i - entry_bar + 1,
        })
        # Prevent overlapping positions for the same ticker.
        i = exit_i + 1

    tdf = pd.DataFrame(trades)
    if tdf.empty:
        return {"trades": tdf, "setups": 0, "closed": 0, "wins": 0, "losses": 0,
                "win_rate": np.nan, "profit_factor": np.nan,
                "expectancy_r": np.nan, "max_drawdown_pct": np.nan}
    pnl = tdf["Net P/L per Share"].astype(float)
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = abs(float(pnl[pnl < 0].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else np.nan)
    equity = pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    max_dd = float(drawdown.min())
    return {
        "trades": tdf, "setups": len(tdf), "closed": len(tdf), "wins": wins,
        "losses": losses, "win_rate": 100 * wins / len(tdf),
        "profit_factor": pf, "expectancy_r": float(tdf["R Multiple"].mean()),
        "max_drawdown_pct": max_dd,  # per-share cumulative P/L, not account-level percentage
    }


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def format_money(x):
    return "N/A" if x is None or not np.isfinite(x) else f"${x:,.2f}"


def safe_indicator_row(
    df: pd.DataFrame,
    tf: str | None = None,
) -> pd.Series | None:
    """Return the newest row with all indicators required for this timeframe."""
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

    # Use the latest complete indicator row. If the newest candle has an
    # incomplete value, avoid incorrectly marking the whole timeframe empty.
    latest_valid_index = valid.index[-1]
    return df.loc[latest_valid_index].copy()


def make_params() -> dict:
    with st.sidebar.expander("⚙️ Strategy & Data Parameters", expanded=False):
        st.caption("Changing parameters changes the strategy. Validate changes out of sample.")
        p = {
            "ema_fast": int(st.number_input("Fast EMA", min_value=2, max_value=50, value=9)),
            "ema_slow": int(st.number_input("Slow EMA", min_value=3, max_value=100, value=21)),
            "sma_fast": int(st.number_input("Fast SMA", min_value=5, max_value=150, value=50)),
            "sma_slow": int(st.number_input("Slow SMA", min_value=20, max_value=400, value=200)),
            "rsi_len": int(st.number_input("RSI Length", min_value=2, max_value=50, value=14)),
            "rsi_os": float(st.number_input("RSI Pullback Ceiling", min_value=20.0, max_value=70.0, value=50.0, step=1.0)),
            "rsi_ob": float(st.number_input("Intraday RSI Upper Bound", min_value=40.0, max_value=90.0, value=65.0, step=1.0)),
            "swing_rsi_min": float(st.number_input("Swing RSI Lower Bound", min_value=10.0, max_value=60.0, value=35.0, step=1.0)),
            "short_rsi_min": float(st.number_input("Bearish RSI Minimum", min_value=35.0, max_value=80.0, value=50.0, step=1.0)),
            "adx_thresh": float(st.number_input("Minimum ADX", min_value=5.0, max_value=50.0, value=20.0, step=1.0)),
            "adx_len": int(st.number_input("ADX Length", min_value=5, max_value=40, value=14)),
            "rvol_len": int(st.number_input("RVOL Average Length", min_value=5, max_value=100, value=20)),
            "rvol_min": float(st.number_input("Minimum RVOL", min_value=0.2, max_value=5.0, value=1.0, step=0.1)),
            "bb_len": int(st.number_input("Bollinger Length", min_value=5, max_value=100, value=20)),
            "bb_std": float(st.number_input("Bollinger Standard Deviations", min_value=1.0, max_value=4.0, value=2.0, step=0.25)),
            "atr_len": int(st.number_input("ATR Length", min_value=5, max_value=50, value=14)),
            "intraday_sl_atr": float(st.number_input("Intraday Stop (ATR)", min_value=0.5, max_value=5.0, value=1.5, step=0.25)),
            "intraday_tp_atr": float(st.number_input("Intraday Target (ATR)", min_value=0.5, max_value=5.0, value=1.0, step=0.25)),
            "swing_sl_atr": float(st.number_input("Swing Stop (ATR)", min_value=0.5, max_value=6.0, value=2.0, step=0.25)),
            "swing_tp_atr": float(st.number_input("Swing Target (ATR)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)),
        }
    # Keep lengths logically ordered; avoid accidentally using a "fast" average slower than "slow".
    if p["ema_fast"] >= p["ema_slow"]:
        st.sidebar.warning("Fast EMA must be shorter than slow EMA. Defaults are used for EMA lengths.")
        p["ema_fast"], p["ema_slow"] = 9, 21
    if p["sma_fast"] >= p["sma_slow"]:
        st.sidebar.warning("Fast SMA must be shorter than slow SMA. Defaults are used for SMA lengths.")
        p["sma_fast"], p["sma_slow"] = 50, 200
    if p["swing_rsi_min"] > p["rsi_os"]:
        p["swing_rsi_min"] = min(35.0, p["rsi_os"])
    return p


def analyze_ticker(ticker: str, tf: str, p: dict) -> tuple[pd.DataFrame, TradingScanner, pd.Series | None, dict]:
    raw = get_price_data(ticker, tf)
    scanner = TradingScanner(ticker, tf, p)
    if raw.empty:
        return pd.DataFrame(), scanner, None, {"action": "NO DATA", "reason": "No market data returned."}
    indicators = add_indicators(raw, tf, p)
    row = safe_indicator_row(indicators, tf)
    if row is None:
        return indicators, scanner, None, {"action": "INSUFFICIENT HISTORY", "reason": "Not enough completed candles for indicators."}
    return indicators, scanner, row, scanner.signal(row, tf)


# Sidebar
st.sidebar.header("Terminal Navigation")
mode = st.sidebar.radio(
    "Operating Mode",
    ["Single Ticker Deep-Dive", "Single-TF Watchlist Scanner", "Multi-Timeframe Matrix"],
    key="operating_mode",
)
selected_tf = st.sidebar.select_slider(
    "Timeframe", options=TIMEFRAMES, value="1h", key="selected_tf"
)
params = make_params()

st.sidebar.markdown("---")
st.sidebar.subheader("Risk & Backtest Settings")
account_equity = float(st.sidebar.number_input("Account Equity ($)", min_value=100.0, max_value=10_000_000.0, value=25_000.0, step=1000.0))
risk_pct = float(st.sidebar.number_input("Risk Budget per Trade (%)", min_value=0.1, max_value=5.0, value=0.5, step=0.1))
max_position_pct = float(st.sidebar.number_input("Max Position (% of Equity)", min_value=1.0, max_value=100.0, value=20.0, step=1.0))
slippage_bps = float(st.sidebar.number_input("Estimated Slippage (basis points per side)", min_value=0.0, max_value=100.0, value=5.0, step=1.0))
commission_per_share = float(st.sidebar.number_input("Commission ($/share/side)", min_value=0.0, max_value=1.0, value=0.0, step=0.01))
hold_bars = int(st.sidebar.number_input("Max Backtest Holding Bars", min_value=1, max_value=200, value=10, step=1))
vix_max = float(st.sidebar.number_input("High-VIX Caution Threshold", min_value=10.0, max_value=80.0, value=30.0, step=1.0))

st.info(
    "Research use only. Signals are based on completed candles; data availability and freshness depend on Yahoo Finance. "
    "Backtest results are hypothetical and do not include market impact or guaranteed execution."
)


# ---------------------------------------------------------------------------
# Mode 1 — Deep dive
# ---------------------------------------------------------------------------

if mode == "Single Ticker Deep-Dive":
    ticker = st.sidebar.text_input("Ticker", value="SPY", key="deep_dive_ticker").strip().upper()
    run_clicked = st.sidebar.button("Run Analysis", type="primary")
    auto_run = st.session_state.pop("_auto_run_deep_dive", False)
    run = run_clicked or auto_run
    if run:
        with st.spinner(f"Analyzing {ticker} ({selected_tf})..."):
            df, scanner, latest, sig = analyze_ticker(ticker, selected_tf, params)
        if df.empty:
            st.error(f"No usable market data found for {ticker}. Check the symbol, interval, or Yahoo Finance availability.")
        elif latest is None:
            st.warning(sig["reason"])
            st.write(f"Downloaded/aggregated candles: {len(df)}")
        else:
            price = float(latest["Close"])
            is_intraday = selected_tf in ("5m", "15m")
            supports, resistances = scanner.support_resistance(df)
            vix_value = latest.get("VIX", np.nan)
            cols = st.columns(6)
            cols[0].metric("Last Completed Close", format_money(price))
            cols[1].metric("RSI", f"{latest['RSI']:.1f}")
            cols[2].metric("ADX", f"{latest['ADX']:.1f}")
            cols[3].metric("RVOL", f"{latest['RVOL']:.2f}x")
            cols[4].metric("ATR", format_money(latest["ATR"]))
            cols[5].metric("Prior VIX Close", f"{vix_value:.1f}" if np.isfinite(vix_value) else "Unavailable")

            st.subheader("🤖 Signal & Trade Plan")
            st.write(f"**Signal:** {sig['action']} — {sig['reason']}")
            if sig["action"] == "BUY":
                plan = scanner.trade_plan(
                    latest, supports, is_intraday, account_equity, risk_pct,
                    max_position_pct, slippage_bps
                )
                a, b, c = st.columns(3)
                a.metric("Estimated Entry", format_money(plan["entry"]))
                a.metric("Stop Loss", format_money(plan["stop"]))
                a.metric("Target 1", format_money(plan["target"]))
                d, e, f = st.columns(3)
                d.metric("Pullback Level", format_money(plan["pullback"]) if np.isfinite(plan["pullback"]) else "No valid level")
                e.metric("Estimated Shares", f"{plan['shares']:,}")
                f.metric("Position Value", format_money(plan["position_value"]))
                st.caption(
                    f"Risk budget: ${plan['risk_dollars']:.2f}; estimated risk at stop: "
                    f"${plan['risk_if_stop']:.2f}; target/stop ratio: "
                    f"{plan['rr']:.2f}R. Quantity is constrained by both risk and position-size limits."
                )
                st.write(
                    f"**Additional targets:** TP2 {format_money(plan['tp2'])} · "
                    f"TP3 {format_money(plan['tp3'])}"
                )
                if not np.isfinite(plan["pullback"]):
                    st.caption("No suitable confirmed swing-low pullback level above the proposed stop was detected.")
            st.subheader("🧱 Confirmed Historical Levels")
            l, r = st.columns(2)
            l.write("**Support:** " + (", ".join(format_money(x) for x in supports) if supports else "No confirmed level detected"))
            r.write("**Resistance:** " + (", ".join(format_money(x) for x in resistances) if resistances else "No confirmed level detected"))

            st.subheader("🧪 Historical Backtest (Long-only)")
            bt = backtest_long_only(df, scanner, selected_tf, hold_bars, slippage_bps, commission_per_share)
            if bt["closed"] == 0:
                st.warning("No completed historical trades were generated, or there is insufficient history. This is not a 0% win rate.")
            else:
                m = st.columns(5)
                m[0].metric("Closed Trades", f"{bt['closed']}")
                m[1].metric("Win Rate", f"{bt['win_rate']:.1f}%")
                m[2].metric("Profit Factor", f"{bt['profit_factor']:.2f}" if np.isfinite(bt["profit_factor"]) else "N/A")
                m[3].metric("Expectancy", f"{bt['expectancy_r']:.2f}R")
                m[4].metric("Max Drawdown", f"${bt['max_drawdown_pct']:.2f}/share")
                st.caption(
                    f"Backtest uses next-bar open entries, one position at a time, a maximum hold of {hold_bars} bars, "
                    "slippage and commissions. If a bar touches stop and target, the stop is assumed first. "
                    "Drawdown is cumulative P/L per share, not account-level drawdown. No short trades are tested."
                )
                with st.expander("View historical trades"):
                    st.dataframe(bt["trades"], use_container_width=True, hide_index=True)

            st.subheader(f"Price Action — {ticker} ({selected_tf})")
            visible = 250 if selected_tf in ("5m", "15m") else 180 if selected_tf in ("1h", "4h") else 120
            chart = df.tail(visible)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=chart.index, open=chart["Open"], high=chart["High"],
                low=chart["Low"], close=chart["Close"], name=ticker
            ))
            if selected_tf in ("5m", "15m", "1h", "4h"):
                fig.add_trace(go.Scatter(x=chart.index, y=chart["VWAP"], name="Session VWAP", line=dict(width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(x=chart.index, y=chart["EMA_Fast"], name=f"{params['ema_fast']} EMA", line=dict(width=1.2)))
                fig.add_trace(go.Scatter(x=chart.index, y=chart["EMA_Slow"], name=f"{params['ema_slow']} EMA", line=dict(width=1.2)))
            else:
                fig.add_trace(go.Scatter(x=chart.index, y=chart["SMA_Fast"], name=f"{params['sma_fast']} SMA", line=dict(width=1.4)))
                fig.add_trace(go.Scatter(x=chart.index, y=chart["SMA_Slow"], name=f"{params['sma_slow']} SMA", line=dict(width=1.4)))
            fig.add_trace(go.Scatter(x=chart.index, y=chart["BB_Upper"], name="Upper Bollinger", line=dict(width=1, dash="dash")))
            fig.add_trace(go.Scatter(x=chart.index, y=chart["BB_Lower"], name="Lower Bollinger", line=dict(width=1, dash="dash")))
            for level in supports:
                fig.add_hline(y=level, line_dash="dot", line_color="green", opacity=0.5)
            for level in resistances:
                fig.add_hline(y=level, line_dash="dot", line_color="red", opacity=0.5)
            fig.update_layout(
                template="plotly_dark", height=680, xaxis_rangeslider_visible=False,
                margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h")
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Last candle shown: {df.index[-1]}. Prices are auto-adjusted Yahoo Finance history.")


# ---------------------------------------------------------------------------
# Mode 2 — Single timeframe watchlist scanner
# ---------------------------------------------------------------------------

elif mode == "Single-TF Watchlist Scanner":
    st.subheader(f"🔭 Watchlist Scanner ({selected_tf})")
    tickers_text = st.text_area("Comma-separated ticker symbols", value=DEFAULT_WATCHLIST)
    scan = st.button("Scan Watchlist", type="primary")
    if scan:
        tickers = list(dict.fromkeys(t.strip().upper() for t in tickers_text.replace("\n", ",").split(",") if t.strip()))
        if not tickers:
            st.warning("Enter at least one ticker.")
        else:
            rows = []
            progress = st.progress(0)
            status = st.empty()
            for i, ticker in enumerate(tickers):
                status.text(f"Analyzing {ticker} ({i + 1}/{len(tickers)})")
                df, scanner, latest, sig = analyze_ticker(ticker, selected_tf, params)
                if latest is not None:
                    supports, resistances = scanner.support_resistance(df)
                    row = {
                        "Ticker": ticker, "Signal": sig["action"],
                        "Close": round(float(latest["Close"]), 2),
                        "RSI": round(float(latest["RSI"]), 1),
                        "ADX": round(float(latest["ADX"]), 1),
                        "RVOL": round(float(latest["RVOL"]), 2),
                        "ATR %": round(float(latest["ATR_Pct"]), 2),
                        "Reason": sig["reason"],
                    }
                    if sig["action"] == "BUY":
                        plan = scanner.trade_plan(
                            latest, supports, selected_tf in ("5m", "15m"),
                            account_equity, risk_pct, max_position_pct, slippage_bps
                        )
                        row.update({
                            "Entry": round(plan["entry"], 2),
                            "Pullback": round(plan["pullback"], 2) if np.isfinite(plan["pullback"]) else np.nan,
                            "Stop": round(plan["stop"], 2),
                            "Target": round(plan["target"], 2),
                            "Shares": plan["shares"],
                            "R:R": round(plan["rr"], 2) if np.isfinite(plan["rr"]) else np.nan,
                        })
                    rows.append(row)
                else:
                    rows.append({"Ticker": ticker, "Signal": sig["action"], "Reason": sig["reason"]})
                progress.progress((i + 1) / len(tickers))
            status.success("Scan complete.")
            results = pd.DataFrame(rows)
            buys = results[results["Signal"] == "BUY"] if "Signal" in results else pd.DataFrame()
            st.subheader(f"Qualified BUY Setups: {len(buys)}")
            if not buys.empty:
                st.dataframe(buys.sort_values(["R:R", "RVOL"], ascending=False, na_position="last") if "R:R" in buys else buys,
                             use_container_width=True, hide_index=True)
            else:
                st.info("No BUY setups qualified. This is a valid outcome; do not loosen filters solely to force signals.")
            with st.expander("Show all ticker results, including WAIT/SELL/insufficient history"):
                st.dataframe(results, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Mode 3 — Multi-timeframe matrix
# ---------------------------------------------------------------------------

else:
    st.subheader("🌐 Multi-Timeframe Trend & Entry Matrix")
    st.caption("Trend alignment and actionable entries are shown separately. Missing/insufficient intervals are not counted as bearish.")
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
                    # Higher timeframes get more weight for trend context.
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
            matrix = pd.DataFrame(rows)

            # Apply muted status colors only to timeframe cells, keeping the summary
            # columns neutral so the table remains easy to scan.
            def matrix_cell_style(value: object) -> str:
                text_value = str(value).upper()
                if text_value == "BUY" or text_value.startswith("BULL TREND"):
                    return "background-color: #214D3A; color: #E8FFF1; font-weight: 600;"
                if text_value == "SELL / AVOID" or text_value.startswith("BEAR TREND"):
                    return "background-color: #632F35; color: #FFECEF; font-weight: 600;"
                if text_value == "CHOP" or text_value.startswith("FLAT TREND"):
                    return "background-color: #424852; color: #F1F3F5;"
                if "INSUFFICIENT" in text_value or "NO DATA" in text_value:
                    return "background-color: #30343B; color: #C4CAD2;"
                if "WAIT" in text_value or "HOLD" in text_value or "N/A" in text_value:
                    return "background-color: #424852; color: #F1F3F5;"
                return "background-color: #424852; color: #F1F3F5;"

            styled_matrix = matrix.style.map(matrix_cell_style, subset=MATRIX_TIMEFRAMES)
            st.caption("Click a cell under 5m, 15m, 1h, 4h, or 1d to open that ticker and timeframe in Single Ticker Deep-Dive.")
            matrix_event = st.dataframe(
                styled_matrix,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-cell",
                key="matrix_table",
                height=min(680, 38 + 35 * (len(matrix) + 1)),
            )

            # A selected cell reports its zero-based row and column. Only timeframe
            # columns trigger navigation; selecting a summary/price cell does nothing.
            selected_cells = matrix_event.selection.cells if matrix_event and hasattr(matrix_event, "selection") else []
            if selected_cells:
                selected_cell = selected_cells[0]
                row_idx = selected_cell.get("row")
                col_idx = selected_cell.get("column")
                if row_idx is not None and col_idx is not None:
                    if 0 <= row_idx < len(matrix) and 0 <= col_idx < len(matrix.columns):
                        clicked_column = str(matrix.columns[col_idx])
                        if clicked_column in MATRIX_TIMEFRAMES:
                            clicked_ticker = str(matrix.iloc[row_idx]["Ticker"])
                            st.session_state["_pending_matrix_navigation"] = {
                                "ticker": clicked_ticker,
                                "timeframe": clicked_column,
                            }
                            st.rerun()

            st.caption(
                "Legend: green = BUY/bullish trend; red = SELL/avoid/bearish trend; slate = wait, chop, or mixed signals; "
                "dark gray = insufficient data. Trend Alignment is a weighted descriptive score, not a probability of profit. "
                "BUY Confirmations counts timeframes meeting this strategy's entry rules. A ticker can be bullish without a valid entry, "
                "and no score guarantees an outcome."
            )

st.markdown("---")
st.caption(
    "BM's Trading Algo • Educational/research tool • Not a broker, not an execution engine, and not investment advice. "
    "Always independently verify prices, liquidity, earnings, corporate actions, and order conditions."
)
