import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import argrelextrema
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="BM's Trading Algo", 
    layout="wide", 
    initial_sidebar_state="auto"
)
st.title("📈 BM's Trading Algo")
st.markdown("*Multi-Timeframe Confluence & Quant Execution Engine.*")

# --- UPDATED WATCHLIST ---
DEFAULT_WATCHLIST = "SPY, QQQ, TSLA, AAPL, MSFT, NVDA, AMZN, META, GOOG, PLTR, AMD, SOXX, ADP, UNH, SPCX, MU, NBIS, IREN"

# --- CORE LOGIC ---
class AlgoTradingScanner:
    def __init__(self, ticker='SPY', timeframe='1h', params=None):
        self.ticker = ticker.upper()
        self.vix_ticker = '^VIX'
        
        self.tf_map = {
            '5m':  {'interval': '5m',  'period': '60d'},
            '15m': {'interval': '15m', 'period': '60d'},
            '1h':  {'interval': '1h',  'period': '730d'},
            '4h':  {'interval': '1h',  'period': '730d'}, 
            '1d':  {'interval': '1d',  'period': '5y'},
            '1wk': {'interval': '1wk', 'period': '10y'}
        }
        
        self.anchor_map = {'5m': '1h', '15m': '4h', '1h': '1d', '4h': '1d', '1d': '1wk', '1wk': None}
        self.timeframe = timeframe
        self.major_indexes = ['SPY', 'QQQ', 'DIA', 'IWM', 'SPX', 'NDX']
        self.is_index = self.ticker in self.major_indexes
        
        self.p = params if params else {
            'ema_fast': 9, 'ema_slow': 21,
            'sma_fast': 50, 'sma_slow': 200,
            'rsi_len': 14, 'rsi_ob': 60, 'rsi_os': 50,
            'adx_thresh': 20, 'adx_len': 14
        }

    def check_earnings_risk(self):
        if self.is_index: return False, None 
        try:
            ticker_obj = yf.Ticker(self.ticker)
            calendar = ticker_obj.calendar
            next_earnings = None
            if isinstance(calendar, dict) and 'Earnings Date' in calendar:
                next_earnings = pd.to_datetime(calendar['Earnings Date'][0]).date()
            elif isinstance(calendar, pd.DataFrame) and not calendar.empty and 'Earnings Date' in calendar.index:
                next_earnings = pd.to_datetime(calendar.loc['Earnings Date'][0]).date()
                
            if next_earnings:
                today = pd.Timestamp.today().date()
                days_until = (next_earnings - today).days
                if 0 <= days_until <= 5: return True, f"{next_earnings} ({days_until} days)"
        except Exception: pass 
        return False, None

    @st.cache_data(ttl=300) 
    def fetch_data(_self, current_ticker, current_timeframe):
        fetch_tf = '1h' if current_timeframe == '4h' else current_timeframe
        settings = _self.tf_map.get(fetch_tf, _self.tf_map['1d'])
        
        try:
            data = yf.download(current_ticker, interval=settings['interval'], period=settings['period'], progress=False)
            vix = yf.download(_self.vix_ticker, interval='1d', period=settings['period'], progress=False)
        except: return pd.DataFrame()
        
        if data.empty: return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

        if data.index.tz is not None: data.index = data.index.tz_localize(None)
        if vix.index.tz is not None: vix.index = vix.index.tz_localize(None)

        if current_timeframe == '4h':
            resample_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            data = data.resample('4h').agg(resample_dict).dropna()

        vix_close = vix['Close']
        data_dates = pd.to_datetime(data.index).normalize()
        vix_dates = pd.to_datetime(vix_close.index).normalize()
        vix_map = pd.Series(vix_close.values, index=vix_dates)
        data['VIX'] = data_dates.map(vix_map).values
        data['VIX'] = data['VIX'].ffill().bfill()
        return data

    def apply_indicators(self, df, tf_str):
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['RSI'] = ta.rsi(df['Close'], length=self.p['rsi_len'])
        df['SMA_Fast'] = ta.sma(df['Close'], length=self.p['sma_fast'])
        df['SMA_Slow'] = ta.sma(df['Close'], length=self.p['sma_slow'])
        df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
        
        # ADX Regime Filter
        adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=self.p['adx_len'])
        if adx_df is not None and not adx_df.empty:
            adx_col = [c for c in adx_df.columns if c.startswith('ADX_')]
            df['ADX'] = adx_df[adx_col[0]] if adx_col else 0
        else:
            df['ADX'] = 0

        # Bollinger Bands for Range/Exhaustion Boundaries
        bb = ta.bbands(df['Close'], length=20, std=2.0)
        if bb is not None and not bb.empty:
            bbl_col = [c for c in bb.columns if c.startswith('BBL_')]
            bbu_col = [c for c in bb.columns if c.startswith('BBU_')]
            df['BB_Lower'] = bb[bbl_col[0]] if bbl_col else df['Close']
            df['BB_Upper'] = bb[bbu_col[0]] if bbu_col else df['Close']
        else:
            df['BB_Lower'], df['BB_Upper'] = df['Close'], df['Close']
        
        is_intraday = tf_str in ['5m', '15m']
        if is_intraday:
            df['EMA_Fast'] = ta.ema(df['Close'], length=self.p['ema_fast'])
            df['EMA_Slow'] =
