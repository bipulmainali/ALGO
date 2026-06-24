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
    page_title="Universal Algo Scanner | Level 3.4", 
    layout="wide", 
    initial_sidebar_state="auto" # Auto-collapses on mobile, stays open on desktop
)
st.title("📈 Universal Trading Terminal (Pro)")
st.markdown("*Quantitative Research Engine & Mass-Scanner. Educational use only.*")

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
            'rsi_len': 14, 'rsi_ob': 60, 'rsi_os': 50
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
        
        is_intraday = tf_str in ['5m', '15m']
        
        if is_intraday:
            df['EMA_Fast'] = ta.ema(df['Close'], length=self.p['ema_fast'])
            df['EMA_Slow'] = ta.ema(df['Close'], length=self.p['ema_slow'])
            df['VWAP'] = ta.vwap(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], anchor="D")
            df['MACD_Hist'], df['Vol_SMA'] = 0, 0
        else:
            macd = ta.macd(df['Close'])
            if macd is not None and not macd.empty:
                macd_col = [c for c in macd.columns if 'MACDh' in c]
                df['MACD_Hist'] = macd[macd_col[0]] if macd_col else 0
            else: df['MACD_Hist'] = 0
            df['Vol_SMA'] = ta.sma(df['Volume'], length=20)
            df['EMA_Fast'], df['EMA_Slow'], df['VWAP'] = 0, 0, 0
            
        return df.dropna()

    def get_signal(self, row, is_intraday):
        action = "WAIT 🟡"
        main_trend = "Bullish" if row['SMA_Fast'] > row['SMA_Slow'] else "Bearish"
        
        if is_intraday:
            if self.is_index:
                if row['EMA_Fast'] > row['EMA_Slow'] and row['Close'] > row['VWAP'] and row['RSI'] < self.p['rsi_os'] and row['VIX'] < 25:
                    action = "BUY 🟢"
            else:
                if row['EMA_Fast'] > row['EMA_Slow'] and row['Close'] > row['VWAP'] and row['RSI'] < (self.p['rsi_os'] + 15):
                    action = "BUY 🟢"
        else:
            macd_bullish = row['MACD_Hist'] > 0
            vol_confirmed = row['Volume'] > row['Vol_SMA']
            if self.is_index:
                if "Bullish" in main_trend and row['RSI'] < (self.p['rsi_os'] - 10) and row['VIX'] < 25 and macd_bullish:
                    action = "BUY 🟢"
            else:
                if "Bullish" in main_trend and row['RSI'] < self.p['rsi_os'] and macd_bullish and vol_confirmed:
                    action = "BUY 🟢"
        return action

    def run_backtest(self, df, is_intraday):
        wins, losses, total_setups = 0, 0, 0
        
        for i in range(100, len(df) - 10): 
            row = df.iloc[i]
            signal = self.get_signal(row, is_intraday)
            
            if "BUY" in signal:
                total_setups += 1
                entry = row['Close']
                atr = row['ATR']
                
                if is_intraday:
                    sl, tp = entry - (1.5 * atr), entry + (1.0 * atr) 
                else:
                    sl, tp = entry - (2.0 * atr), entry + (2.0 * atr) 
                
                for j in range(1, 11):
                    future_bar = df.iloc[i + j]
                    if future_bar['Low'] <= sl:
                        losses += 1
                        break
                    elif future_bar['High'] >= tp:
                        wins += 1
                        break
                        
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        return total_setups, win_rate

    def find_support_resistance(self, df, order=20):
        prices = df['Close'].values
        maxima_indices = argrelextrema(prices, np.greater, order=order)[0]
        minima_indices = argrelextrema(prices, np.less, order=order)[0]
        res = sorted([r for r in prices[maxima_indices] if r > prices[-1]])[:3]
        sup = sorted([s for s in prices[minima_indices] if s < prices[-1]], reverse=True)[:3]
        return sup, res

# --- APP LAYOUT ---
st.sidebar.header("Terminal Settings")
app_mode = st.sidebar.radio("Select Mode", ["Single Ticker Deep-Dive", "Watchlist Mass-Scanner"])

st.sidebar.markdown("---")
selected_tf = st.sidebar.select_slider("Time Interval", options=["5m", "15m", "1h", "4h", "1d", "1wk"], value="1h")

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Tuning Parameters (Advanced)"):
    st.markdown("**Intraday Settings**")
    ema_f = st.number_input("Fast EMA", value=9, step=1)
    ema_s = st.number_input("Slow EMA", value=21, step=1)
    st.markdown("**Swing Settings**")
    sma_f = st.number_input("Fast SMA", value=50, step=10)
    sma_s = st.number_input("Slow SMA", value=200, step=10)
    st.markdown("**Momentum Settings**")
    rsi_l = st.number_input("RSI Length", value=14, step=1)
    rsi_os = st.number_input("RSI Oversold Base", value=50, step=5)

algo_params = {
    'ema_fast': ema_f, 'ema_slow': ema_s, 
    'sma_fast': sma_f, 'sma_slow': sma_s, 
    'rsi_len': rsi_l, 'rsi_os': rsi_os
}

# ==========================================
# MODE 1: SINGLE TICKER DEEP-DIVE
# ==========================================
if app_mode == "Single Ticker Deep-Dive":
    input_ticker = st.sidebar.text_input("Enter Ticker", value="SPY").upper()
    run_scan = st.sidebar.button("Run Quantitative Scan", type="primary")

    if run_scan:
        with st.spinner(f"Crunching math & backtesting {input_ticker}..."):
            scanner = AlgoTradingScanner(ticker=input_ticker, timeframe=selected_tf, params=algo_params)
            df = scanner.fetch_data(input_ticker, selected_tf)
            
            if df.empty:
                st.error(f"No data found for '{input_ticker}'.")
                st.stop()
                
            df = scanner.apply_indicators(df, selected_tf)
            supp, res = scanner.find_support_resistance(df, order=15)
            latest = df.iloc[-1]
            price = latest['Close']
            atr = latest['ATR']
            is_intraday = selected_tf in ['5m', '15m']
            main_trend = "Bullish 🟢" if latest['SMA_Fast'] > latest['SMA_Slow'] else "Bearish 🔴"

            anchor_tf = scanner.anchor_map.get(selected_tf)
            anchor_trend_str = "N/A (Macro Supreme)"
            if anchor_tf:
                df_anchor = scanner.fetch_data(input_ticker, anchor_tf)
                if not df_anchor.empty:
                    df_anchor = scanner.apply_indicators(df_anchor, anchor_tf)
                    anchor_latest = df_anchor.iloc[-1]
                    anchor_trend_str = "Bullish 🟢" if anchor_latest['SMA_Fast'] > anchor_latest['SMA_Slow'] else "Bearish 🔴"

            earnings_risk, risk_message = scanner.check_earnings_risk()
            
            # 1. TOP DASHBOARD METRICS
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric(f"Current {input_ticker}", f"${price:.2f}")
            col2.metric("VIX (Macro)", f"{latest['VIX']:.2f}")
            col3.metric("RSI", f"{latest['RSI']:.2f}")
            col4.metric("Local Trend", main_trend)
            col5.metric(f"Anchor ({anchor_tf if anchor_tf else 'Macro'})", anchor_trend_str)

            st.markdown("---")

            # 2. HISTORICAL BACKTEST
            st.subheader("🧪 Historical Strategy Backtest")
            setups, win_rate = scanner.run_backtest(df, is_intraday)
            bt_col1, bt_col2 = st.columns(2)
            bt_col1.write(f"**Historical Setups Found:** {setups}")
            bt_col2.write(f"**Historical Win Rate (TP1):** {win_rate:.1f}%")
            
            if is_intraday:
                st.caption("*Intraday Backtest assumes hitting a 1.0 ATR Target before a 1.5 ATR Stop Loss.*")
            else:
                st.caption("*Swing Backtest assumes hitting a 2.0 ATR Target before a 2.0 ATR Stop Loss.*")
                
            st.markdown("---")

            # 3. STRUCTURAL LEVELS (MOVED HERE)
            st.subheader("🧱 Key Structural Levels")
            nearest_supp_str = f"\${supp[0]:.2f}" if len(supp) > 0 else "N/A (No Floor Detected)"
            nearest_res_str = f"\${res[0]:.2f}" if len(res) > 0 else "N/A (Blue Sky Breakout 🚀)"
            
            struct_col1, struct_col2 = st.columns(2)
            struct_col1.write(f"**Nearest Resistance (Ceiling):** {nearest_res_str}")
            struct_col2.write(f"**Nearest Support (Floor):** {nearest_supp_str}")

            st.markdown("---")

            # 4. ALGORITHMIC TRADE PLAN
            st.subheader("🤖 Algorithmic Trade Plan")
            action = scanner.get_signal(latest, is_intraday)

            if earnings_risk:
                action = f"WAIT 🟡 (EARNINGS RISK: {risk_message})"

            if "BUY" in action:
                if is_intraday:
                    sl, tp1, tp2, tp3 = price - (1.5 * atr), price + (1.0 * atr), price + (2.0 * atr), price + (3.0 * atr)
                else:
                    sl, tp1, tp2, tp3 = price - (2.0 * atr), price + (2.0 * atr), price + (4.0 * atr), price + (6.0 * atr)
            else:
                sl = tp1 = tp2 = tp3 = 0.0

            st.write(f"**Recommendation:** {action} *(Engine: {'Intraday' if is_intraday else 'Swing'} | Mode: {'Index ETF' if scanner.is_index else 'Individual Stock'})*")
            
            if "BUY" in action:
                pb_price = supp[0] if len(supp) > 0 else price - atr
                
                plan_col1, plan_col2 = st.columns(2)
                with plan_col1:
                    st.write(f"**Immediate Entry (Momentum):** \${price - (atr*0.2):.2f} - \${price + (atr*0.2):.2f}")
                    st.write(f"**Pullback Entry (Limit Order):** \${pb_price:.2f}")
                    st.write(f"**Stop Loss:** \${sl:.2f} *(Multi: {'1.5x' if is_intraday else '2.0x'})*")
                with plan_col2:
                    st.write(f"**TP 1 (Safe):** \${tp1:.2f} *(Multi: {'1.0x' if is_intraday else '2.0x'})*")
                    st.write(f"**TP 2 (Target):** \${tp2:.2f} *(Multi: {'2.0x' if is_intraday else '4.0x'})*")
                    st.write(f"**TP 3 (Runner):** \${tp3:.2f} *(Multi: {'3.0x' if is_intraday else '6.0x'})*")

            # 5. CHARTING
            st.subheader(f"Interactive Price Action: {input_ticker} ({selected_tf})")
            display_candles = 250 if is_intraday else 150 if selected_tf in ['1h', '4h'] else 100
            chart_df = df.tail(display_candles).copy() 
            
            fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name=input_ticker)])
            
            if is_intraday:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['VWAP'], line=dict(color='yellow', width=2, dash='dot'), name='VWAP'))
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_Fast'], line=dict(color='cyan', width=1.5), name=f'{ema_f} EMA'))
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['EMA_Slow'], line=dict(color='magenta', width=1.5), name=f'{ema_s} EMA'))
            else:
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_Fast'], line=dict(color='orange', width=1.5), name=f'{sma_f} MA'))
                fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['SMA_Slow'], line=dict(color='blue', width=1.5), name=f'{sma_s} MA'))
            
            for r in res: fig.add_hline(y=r, line_dash="dash", line_color="rgba(255, 0, 0, 0.5)")
            for s in supp: fig.add_hline(y=s, line_dash="dash", line_color="rgba(0, 255, 0, 0.5)")

            if selected_tf in ['1d', '1wk']:
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            else:
                hour_bound = 8 if selected_tf == '4h' else 9.5
                fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, hour_bound], pattern="hour")])
                chart_df['Date_Only'] = chart_df.index.date
                new_days = chart_df[chart_df['Date_Only'] != chart_df['Date_Only'].shift(1)]
                for i in range(1, len(new_days)):
                    fig.add_vline(x=new_days.index[i], line_width=1, line_dash="dot", line_color="rgba(255, 255, 255, 0.15)")

            fig.update_layout(xaxis_rangeslider_visible=False, height=800, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# MODE 2: WATCHLIST MASS-SCANNER
# ==========================================
elif app_mode == "Watchlist Mass-Scanner":
    st.subheader(f"🔭 Watchlist Screener ({selected_tf})")
    st.write("Paste a list of tickers separated by commas. The engine will scan them all and only show active BUY setups.")
    
    ticker_input = st.text_area(
        "Watchlist Tickers:", 
        value="SPY, QQQ, TSLA, AAPL, MSFT, NVDA, AMZN, META, GOOG, PLTR, AMD, SOXX, ADP, UHC, F"
    ).upper()
    
    run_mass_scan = st.button("Initialize Mass Scan", type="primary")
    
    if run_mass_scan:
        ticker_list = [t.strip() for t in ticker_input.split(",") if t.strip()]
        
        if not ticker_list:
            st.error("Please enter at least one ticker.")
            st.stop()
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        buy_list = []
        is_intraday = selected_tf in ['5m', '15m']
        
        for i, tick in enumerate(ticker_list):
            status_text.text(f"Scanning {tick} ({i+1}/{len(ticker_list)})...")
            scanner = AlgoTradingScanner(ticker=tick, timeframe=selected_tf, params=algo_params)
            
            df = scanner.fetch_data(tick, selected_tf)
            if not df.empty:
                df = scanner.apply_indicators(df, selected_tf)
                latest = df.iloc[-1]
                earnings_risk, _ = scanner.check_earnings_risk()
                
                if not earnings_risk:
                    action = scanner.get_signal(latest, is_intraday)
                    if "BUY" in action:
                        atr = latest['ATR']
                        price = latest['Close']
                        supp, res = scanner.find_support_resistance(df, order=15)
                        
                        if is_intraday:
                            sl = price - (1.5 * atr)
                            tp1 = price + (1.0 * atr)
                        else:
                            sl = price - (2.0 * atr)
                            tp1 = price + (2.0 * atr)
                            
                        pb_price = supp[0] if len(supp) > 0 else price - atr
                        nearest_res = res[0] if len(res) > 0 else price + (atr * 3)

                        buy_list.append({
                            "Ticker": tick,
                            "Price": f"${price:.2f}",
                            "RSI": round(latest['RSI'], 2),
                            "Immediate Entry": f"${price-(atr*0.2):.2f}",
                            "Pullback (Limit)": f"${pb_price:.2f}",
                            "Nearest Res": f"${nearest_res:.2f}",
                            "Stop Loss": f"${sl:.2f}",
                            "Target (TP1)": f"${tp1:.2f}"
                        })
            
            progress_bar.progress((i + 1) / len(ticker_list))
            
        status_text.text("Scan Complete!")
        st.markdown("---")
        
        if buy_list:
            st.success(f"Found {len(buy_list)} active setups!")
            results_df = pd.DataFrame(buy_list)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        else:
            st.info("No active 'BUY' setups found in this watchlist right now based on your current parameters.")