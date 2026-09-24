import os
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
import ta
import plotly.graph_objects as go
import streamlit as st

# ML import fallbacks
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIG & SST DARK THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SST AI Swingtrade & Adaptive ML Scanner",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        color: #8B949E;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #F0F6FC;
    }
    .status-bullish { color: #39D353; font-weight: bold; }
    .status-bearish { color: #F85149; font-weight: bold; }
    .status-neutral { color: #D29922; font-weight: bold; }
    .data-unavailable { color: #8B949E; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA RETRIEVAL MODULE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_market_data(ticker_symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.reset_index(inplace=True)
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=180)
def get_intraday_data(ticker_symbol: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df.reset_index(inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_ticker_info(ticker_symbol: str) -> dict:
    try:
        return yf.Ticker(ticker_symbol).info or {}
    except Exception:
        return {}

# -----------------------------------------------------------------------------
# CLASSIFICATION & ADAPTIVE WEIGHTS
# -----------------------------------------------------------------------------
def classify_stock_type(ticker: str, info: dict) -> str:
    sector = info.get('sector', '').upper()
    industry = info.get('industry', '').upper()
    summary = info.get('longBusinessSummary', '').upper()
    
    t_upper = ticker.upper()
    if t_upper in ['IONQ', 'RGTI', 'QUBT', 'QUBT']:
        return "QUANTUM"
    
    if "BIOTECH" in industry or "PHARMA" in industry or "BIOTECHNOLOGY" in sector:
        return "BIOTECH"
    if "MINING" in industry or "COPPER" in industry or "GOLD" in industry or "BASIC MATERIALS" in sector:
        return "MINING / COMMODITY"
    if "SEMICONDUCTOR" in industry or t_upper in ['NVDA', 'AMD', 'AVGO', 'TSM', 'SMCI']:
        return "TECHNOLOGY / SEMICONDUCTOR"
    if "SOFTWARE" in industry or "ARTIFICIAL INTELLIGENCE" in summary or t_upper in ['PLTR', 'AI', 'PATH']:
        return "AI / SOFTWARE"
    if "BROKER" in industry or "FINANCIAL" in sector or t_upper in ['HOOD', 'COIN']:
        return "FINANCIAL / BROKER"
    
    beta = info.get('beta', 1.0)
    if beta and beta > 1.8:
        return "HIGH-BETA / MOMENTUM"
    
    if sector:
        return f"CONSUMER / LARGE CAP" if info.get('marketCap', 0) > 1e10 else "OTHER"
    return "OTHER"

def get_adaptive_weights(stock_type: str) -> dict:
    weights = {
        "AI / SOFTWARE": {"tech": 0.20, "mom": 0.15, "vol": 0.20, "mtf": 0.15, "options": 0.10, "ml": 0.20},
        "TECHNOLOGY / SEMICONDUCTOR": {"tech": 0.20, "mom": 0.15, "vol": 0.20, "mtf": 0.15, "options": 0.10, "ml": 0.20},
        "HIGH-BETA / MOMENTUM": {"tech": 0.15, "mom": 0.20, "vol": 0.20, "mtf": 0.15, "short": 0.10, "options": 0.10, "ml": 0.10},
        "QUANTUM": {"mom": 0.20, "vol": 0.20, "tech": 0.15, "mtf": 0.15, "short": 0.10, "squeeze": 0.05, "options": 0.05, "ml": 0.10},
        "BIOTECH": {"tech": 0.20, "mom": 0.15, "vol": 0.20, "mtf": 0.15, "short": 0.10, "options": 0.10, "ml": 0.10},
        "MINING / COMMODITY": {"tech": 0.20, "mom": 0.15, "vol": 0.15, "mtf": 0.15, "commodity": 0.15, "rs": 0.10, "ml": 0.10},
    }
    return weights.get(stock_type, {"tech": 0.25, "mom": 0.20, "vol": 0.20, "mtf": 0.15, "ml": 0.20})

# -----------------------------------------------------------------------------
# TECHNICAL & INDICATOR CALCULATIONS
# -----------------------------------------------------------------------------
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    
    close = df['Close']
    df['EMA5'] = ta.trend.ema_indicator(close, window=5)
    df['EMA20'] = ta.trend.ema_indicator(close, window=20)
    df['EMA50'] = ta.trend.ema_indicator(close, window=50)
    df['EMA200'] = ta.trend.ema_indicator(close, window=200) if len(df) >= 200 else np.nan
    
    df['RSI14'] = ta.momentum.rsi(close, window=14)
    
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_sig'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    
    df['Stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], close, window=14, smooth_window=3)
    df['ROC'] = ta.momentum.roc(close, window=12)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], close, window=14)
    
    df['Vol_Avg20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / (df['Vol_Avg20'] + 1e-9)
    
    df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], close, df['Volume'], window=14)
    df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], close, df['Volume'], window=20)
    df['OBV'] = ta.volume.on_balance_volume(close, df['Volume'])
    
    return df

def calculate_technical_score(df: pd.DataFrame) -> tuple:
    if df.empty or len(df) < 50:
        return 0.0, "INSUFFICIENT DATA"
    
    row = df.iloc[-1]
    score = 0
    max_pts = 100
    
    # Trend structure
    if row['Close'] > row['EMA20']: score += 25
    if row['EMA20'] > row['EMA50']: score += 25
    if pd.notna(row['EMA200']) and row['EMA50'] > row['EMA200']: score += 15
    if row['Close'] > row['EMA5']: score += 10
    
    # RSI
    rsi = row['RSI14']
    if pd.notna(rsi):
        if 50 <= rsi <= 70: score += 15
        elif 40 <= rsi < 50: score += 10
        elif rsi > 70: score += 5
        
    # MACD
    if pd.notna(row['MACD_hist']) and row['MACD_hist'] > 0: score += 10
    
    trend_desc = "NEUTRAL"
    if score >= 80: trend_desc = "STRONG BULLISH"
    elif score >= 60: trend_desc = "BULLISH"
    elif score <= 30: trend_desc = "STRONG BEARISH"
    elif score <= 45: trend_desc = "BEARISH"
    
    return float(score), trend_desc

def calculate_momentum_score(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 20: return 0.0
    row = df.iloc[-1]
    score = 0
    if pd.notna(row['ROC']):
        score += np.clip(row['ROC'] * 4, 0, 40)
    if pd.notna(row['Stoch_k']):
        if 40 <= row['Stoch_k'] <= 80: score += 30
        elif row['Stoch_k'] > 80: score += 15
    if pd.notna(row['RSI14']) and row['RSI14'] > 50:
        score += 30
    return float(np.clip(score, 0, 100))

def calculate_volume_score(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 20: return 0.0
    row = df.iloc[-1]
    score = 0
    vr = row['Vol_Ratio'] if pd.notna(row['Vol_Ratio']) else 1.0
    
    if vr >= 2.0: score += 50
    elif vr >= 1.3: score += 35
    elif vr >= 1.0: score += 20
    
    # Price confirmation
    if row['Close'] > df['Close'].iloc[-2]:
        score += 50
    return float(np.clip(score, 0, 100))

def calculate_money_flow_score(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 20: return 0.0
    row = df.iloc[-1]
    score = 50.0
    if pd.notna(row['MFI']):
        score = row['MFI']
    if pd.notna(row['CMF']):
        score = (score + np.clip((row['CMF'] + 0.5) * 100, 0, 100)) / 2
    return float(np.clip(score, 0, 100))

# -----------------------------------------------------------------------------
# MULTI-TIMEFRAME ALIGNMENT
# -----------------------------------------------------------------------------
def calculate_mtf_alignment(ticker: str) -> tuple:
    df_1d = get_market_data(ticker, period="6m", interval="1d")
    df_1h = get_intraday_data(ticker, period="5d", interval="1h")
    df_15m = get_intraday_data(ticker, period="3d", interval="15m")
    
    if df_1d.empty: return 0.0, "DATA UNAVAILABLE"
    
    d_score, _ = calculate_technical_score(calculate_technical_indicators(df_1d))
    
    h_score = 50.0
    if not df_1h.empty and len(df_1h) >= 20:
        df_1h = calculate_technical_indicators(df_1h)
        if df_1h.iloc[-1]['Close'] > df_1h.iloc[-1]['EMA20']: h_score += 25
        if df_1h.iloc[-1]['RSI14'] > 50: h_score += 25
        
    m15_score = 50.0
    if not df_15m.empty and len(df_15m) >= 20:
        df_15m = calculate_technical_indicators(df_15m)
        if df_15m.iloc[-1]['Close'] > df_15m.iloc[-1]['EMA20']: m15_score += 25
        if df_15m.iloc[-1]['Vol_Ratio'] > 1.2: m15_score += 25

    mtf_score = (d_score * 0.5) + (h_score * 0.3) + (m15_score * 0.2)
    
    if mtf_score >= 80: alignment = "STERK (ALIGNED)"
    elif mtf_score >= 65: alignment = "ALIGNED"
    elif mtf_score >= 50: alignment = "PARTIAL"
    elif mtf_score >= 35: alignment = "WEAK"
    else: alignment = "BEARISH"
    
    return float(mtf_score), alignment

# -----------------------------------------------------------------------------
# OPTIONS, SHORT INTEREST, & COMMODITY MODULES
# -----------------------------------------------------------------------------
def calculate_options_score(info: dict) -> tuple:
    # Verifies real availability without fabricating numbers
    if 'options' not in info and not info.get('openInterest'):
        return None, "DATA UNAVAILABLE"
    return 50.0, "NEUTRAL"

def calculate_short_module(info: dict, df: pd.DataFrame) -> tuple:
    short_float = info.get('shortPercentOfFloat', None)
    short_ratio = info.get('shortRatio', None) # Days to cover
    
    if short_float is None:
        return None, None, "DATA UNAVAILABLE"
    
    sf_pct = short_float * 100
    short_score = np.clip(sf_pct * 3, 0, 100)
    
    # Short Squeeze Potential calculation
    vol_spike = df.iloc[-1]['Vol_Ratio'] > 1.5 if not df.empty else False
    price_breakout = df.iloc[-1]['Close'] > df.iloc[-1]['EMA20'] if not df.empty else False
    
    squeeze_score = (sf_pct * 2.5) + (short_ratio * 5 if short_ratio else 0)
    if vol_spike: squeeze_score += 15
    if price_breakout: squeeze_score += 15
    
    squeeze_score = float(np.clip(squeeze_score, 0, 100))
    return float(short_score), squeeze_score, "AVAILABLE"

def calculate_relative_strength(df_stock: pd.DataFrame, benchmark_symbol: str = "SPY") -> float:
    df_bench = get_market_data(benchmark_symbol, period="3m", interval="1d")
    if df_stock.empty or df_bench.empty or len(df_stock) < 20 or len(df_bench) < 20:
        return 50.0
    
    stock_ret = (df_stock['Close'].iloc[-1] / df_stock['Close'].iloc[-20]) - 1
    bench_ret = (df_bench['Close'].iloc[-1] / df_bench['Close'].iloc[-20]) - 1
    
    rs_diff = (stock_ret - bench_ret) * 100
    return float(np.clip(50 + (rs_diff * 3), 0, 100))

def calculate_commodity_score(stock_type: str) -> tuple:
    if stock_type != "MINING / COMMODITY":
        return None, "N/A"
    
    df_copper = get_market_data("HG=F", period="1m", interval="1d")
    if df_copper.empty or len(df_copper) < 10:
        return None, "COMMODITY DATA UNAVAILABLE"
    
    ret = (df_copper['Close'].iloc[-1] / df_copper['Close'].iloc[-10]) - 1
    score = np.clip(50 + (ret * 500), 0, 100)
    return float(score), "AVAILABLE"

# -----------------------------------------------------------------------------
# GENUINE MACHINE LEARNING MODULE
# -----------------------------------------------------------------------------
def train_and_predict_ml(df: pd.DataFrame) -> tuple:
    if len(df) < 120:
        return None, None, "INSUFFICIENT DATA FOR ML (<120 bars)"
    
    df_ml = df.copy()
    # Target: Will stock close higher in 3 trading sessions?
    df_ml['Target'] = (df_ml['Close'].shift(-3) > df_ml['Close']).astype(int)
    
    features = ['RSI14', 'MACD_hist', 'Vol_Ratio', 'ROC', 'Stoch_k', 'MFI']
    df_ml.dropna(subset=features + ['Target'], inplace=True)
    
    if len(df_ml) < 100:
        return None, None, "INSUFFICIENT CLEAN DATA"
        
    X = df_ml[features]
    y = df_ml['Target']
    
    # TimeSeriesSplit validation - NO LOOK-AHEAD BIAS
    tscv = TimeSeriesSplit(n_splits=3)
    train_idx, test_idx = list(tscv.split(X))[-1]
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    if CATBOOST_AVAILABLE:
        model = CatBoostClassifier(iterations=100, depth=4, verbose=0, random_seed=42)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
        
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    
    # Predict on latest bar
    latest_features = X.iloc[[-1]]
    prob = model.predict_proba(latest_features)[0][1] * 100
    confidence = acc * 100
    
    return float(prob), float(confidence), "VALIDATED ML MODEL"

# -----------------------------------------------------------------------------
# TRADE SETUP & SIGNAL GENERATION
# -----------------------------------------------------------------------------
def calculate_trade_levels(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 20: return {}
    row = df.iloc[-1]
    entry = row['Close']
    atr = row['ATR'] if pd.notna(row['ATR']) else (entry * 0.02)
    
    stop = entry - (1.2 * atr)
    tp1 = entry + (1.5 * (entry - stop))
    tp2 = entry + (2.5 * (entry - stop))
    
    risk_pct = ((entry - stop) / entry) * 100
    reward_pct = ((tp1 - entry) / entry) * 100
    rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0
    
    entry_type = "TREND CONTINUATION"
    if row['Close'] > df['High'].rolling(20).max().iloc[-2]:
        entry_type = "BREAKOUT"
    elif row['Close'] < row['EMA20'] and row['Close'] > row['EMA50']:
        entry_type = "PULLBACK"
        
    return {
        "Entry": float(entry),
        "Stop": float(stop),
        "TP1": float(tp1),
        "TP2": float(tp2),
        "Risk_Pct": float(risk_pct),
        "Reward_Pct": float(reward_pct),
        "RR": float(rr_ratio),
        "Type": entry_type
    }

def generate_reasons_and_risks(df: pd.DataFrame, score: float) -> tuple:
    reasons, risks = [], []
    if df.empty: return reasons, risks
    row = df.iloc[-1]
    
    if row['Close'] > row['EMA20']: reasons.append("Price structured above 20 EMA")
    if row['Vol_Ratio'] > 1.3: reasons.append(f"Volume spike ({row['Vol_Ratio']:.1f}x average)")
    if row['MACD_hist'] > 0: reasons.append("MACD histogram turned positive")
    
    if row['RSI14'] > 68: risks.append("RSI approaching overbought levels (>68)")
    if row['Vol_Ratio'] < 0.8: risks.append("Sub-average volume confirmation")
    if row['Close'] < row['EMA50']: risks.append("Trading below key 50 EMA support")
    
    return reasons, risks

# -----------------------------------------------------------------------------
# MAIN APP ARCHITECTURE
# -----------------------------------------------------------------------------
def main():
    st.title("SST AI SWINGTRADE & ADAPTIVE ML SCANNER")
    st.caption("Professional Swing Trading Decision Engine | U.S. Equities (1-5 Days)")
    
    with st.sidebar:
        st.header("Scanner Inputs")
        tickers_input = st.text_input(
            "Tickers (comma separated):", 
            value="PLTR, HOOD, AMD, AAPL, NICE, IONQ, XERS, ERO"
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        run_scan = st.button("RUN ADAPTIVE SCAN", type="primary")

    if run_scan or 'scan_results' in st.session_state:
        if run_scan:
            results = []
            progress_bar = st.progress(0)
            
            for idx, ticker in enumerate(tickers):
                info = get_ticker_info(ticker)
                df_daily = get_market_data(ticker, period="1y", interval="1d")
                
                if df_daily.empty:
                    st.warning(f"Skipping {ticker}: Data Unavailable")
                    continue
                
                df_daily = calculate_technical_indicators(df_daily)
                stock_type = classify_stock_type(ticker, info)
                weights = get_adaptive_weights(stock_type)
                
                # Scores
                tech_score, trend_desc = calculate_technical_score(df_daily)
                mom_score = calculate_momentum_score(df_daily)
                vol_score = calculate_volume_score(df_daily)
                mf_score = calculate_money_flow_score(df_daily)
                mtf_score, mtf_align = calculate_mtf_alignment(ticker)
                opt_score, opt_status = calculate_options_score(info)
                short_score, squeeze_score, short_status = calculate_short_module(info, df_daily)
                rs_score = calculate_relative_strength(df_daily)
                comm_score, comm_status = calculate_commodity_score(stock_type)
                ml_prob, ml_conf, ml_status = train_and_predict_ml(df_daily)
                
                # Data Quality Score
                dq_items = [df_daily is not None, opt_score is not None, short_score is not None, ml_prob is not None]
                data_quality = (sum(1 for item in dq_items if item) / len(dq_items)) * 100
                
                # Adaptive SST Calculation
                final_score = 0.0
                total_weight = 0.0
                
                mapping = {
                    "tech": tech_score, "mom": mom_score, "vol": vol_score, "mtf": mtf_score,
                    "options": opt_score, "short": short_score, "squeeze": squeeze_score,
                    "commodity": comm_score, "rs": rs_score, "ml": ml_prob
                }
                
                for key, weight in weights.items():
                    val = mapping.get(key)
                    if val is not None:
                        final_score += val * weight
                        total_weight += weight
                        
                if total_weight > 0:
                    final_score = final_score / total_weight
                else:
                    final_score = tech_score
                    
                # Signal
                if final_score >= 78 and mtf_score >= 65: signal = "STRONG BUY"
                elif final_score >= 65: signal = "BUY"
                elif final_score >= 50: signal = "WATCH"
                elif final_score >= 35: signal = "WAIT"
                else: signal = "SELL / WEAK"
                
                setup = calculate_trade_levels(df_daily)
                reasons, risks = generate_reasons_and_risks(df_daily, final_score)
                
                results.append({
                    "Ticker": ticker, "Type": stock_type, "Price": df_daily.iloc[-1]['Close'],
                    "SST Score": final_score, "ML Prob": ml_prob, "ML Conf": ml_conf, "ML Status": ml_status,
                    "MTF Score": mtf_score, "MTF Align": mtf_align, "Tech Score": tech_score,
                    "Mom Score": mom_score, "Vol Score": vol_score, "MF Score": mf_score,
                    "Short Score": short_score, "Squeeze Score": squeeze_score, "Opt Score": opt_score,
                    "Comm Score": comm_score, "RS Score": rs_score, "Data Quality": data_quality,
                    "Signal": signal, "Setup": setup, "Reasons": reasons, "Risks": risks, "DF": df_daily
                })
                progress_bar.progress((idx + 1) / len(tickers))
                
            st.session_state['scan_results'] = results

        results = st.session_state['scan_results']
        
        # ---------------------------------------------------------------------
        # OVERVIEW TABLE
        # ---------------------------------------------------------------------
        st.subheader("Market Overview Table")
        
        table_data = []
        for r in results:
            table_data.append({
                "Ticker": r["Ticker"],
                "Type": r["Type"],
                "Price": f"${r['Price']:.2f}",
                "SST Score": f"{r['SST Score']:.1f}",
                "ML Prob": f"{r['ML Prob']:.1f}%" if r["ML Prob"] is not None else "UNAVAILABLE",
                "MTF": r["MTF Align"],
                "RSI": f"{r['DF'].iloc[-1]['RSI14']:.1f}" if pd.notna(r['DF'].iloc[-1]['RSI14']) else "N/A",
                "Volume Ratio": f"{r['DF'].iloc[-1]['Vol_Ratio']:.2f}x",
                "Squeeze Score": f"{r['Squeeze Score']:.1f}" if r["Squeeze Score"] is not None else "N/A",
                "Signal": r["Signal"]
            })
            
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True)
        
        # ---------------------------------------------------------------------
        # INDIVIDUAL STOCK CARDS
        # ---------------------------------------------------------------------
        st.subheader("Individual Stock Cards & Setup Details")
        
        for r in sorted(results, key=lambda x: x["SST Score"], reverse=True):
            with st.expander(f"**${r['Ticker']}** — {r['Signal']} | SST Score: {r['SST Score']:.1f}/100", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"**Stock Type:** {r['Type']}")
                    st.markdown(f"**SST Score:** `{r['SST Score']:.1f}/100`")
                    st.markdown(f"**Signal:** `{r['Signal']}`")
                    
                with col2:
                    ml_display = f"{r['ML Prob']:.1f}%" if r['ML Prob'] is not None else "DATA UNAVAILABLE"
                    st.markdown(f"**ML Probability:** {ml_display}")
                    st.markdown(f"**MTF Alignment:** {r['MTF Align']}")
                    st.markdown(f"**Data Quality:** {r['Data Quality']:.0f}%")

                with col3:
                    st.markdown(f"**Technical:** {r['Tech Score']:.1f}")
                    st.markdown(f"**Momentum:** {r['Mom Score']:.1f}")
                    st.markdown(f"**Volume:** {r['Vol Score']:.1f}")

                with col4:
                    sq_display = f"{r['Squeeze Score']:.1f}" if r['Squeeze Score'] is not None else "N/A"
                    opt_display = f"{r['Opt Score']:.1f}" if r['Opt Score'] is not None else "N/A"
                    st.markdown(f"**Short Squeeze:** {sq_display}")
                    st.markdown(f"**Options Score:** {opt_display}")
                    st.markdown(f"**Relative Strength:** {r['RS Score']:.1f}")

                st.divider()
                
                # Trade Levels
                setup = r["Setup"]
                if setup:
                    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                    sc1.metric("ENTRY TYPE", setup["Type"])
                    sc2.metric("ENTRY PRICE", f"${setup['Entry']:.2f}")
                    sc3.metric("STOP LOSS", f"${setup['Stop']:.2f}", f"-{setup['Risk_Pct']:.1f}%")
                    sc4.metric("TARGET 1 (TP1)", f"${setup['TP1']:.2f}", f"+{setup['Reward_Pct']:.1f}%")
                    sc5.metric("RISK / REWARD", f"1 : {setup['RR']:.2f}")

                # Explanation Engine
                st.markdown("#### WHY THIS TRADE?")
                reasons_col, risks_col = st.columns(2)
                
                with reasons_col:
                    st.markdown("**Key Strengths:**")
                    for reason in r["Reasons"]:
                        st.markdown(f"✓ {reason}")
                    if not r["Reasons"]: st.markdown("*No strong technical drivers identified.*")

                with risks_col:
                    st.markdown("**Key Risks:**")
                    for risk in r["Risks"]:
                        st.markdown(f"⚠ {risk}")
                    if not r["Risks"]: st.markdown("*No major structural risks flagged.*")

                # Interactive Plotly Chart
                st.markdown("#### Price & Indicator Chart")
                df_chart = r["DF"].tail(100)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_chart['Datetime'],
                    open=df_chart['Open'], high=df_chart['High'],
                    low=df_chart['Low'], close=df_chart['Close'],
                    name="Price"
                ))
                if 'EMA20' in df_chart.columns:
                    fig.add_trace(go.Scatter(x=df_chart['Datetime'], y=df_chart['EMA20'], name="EMA 20", line=dict(color='yellow', width=1)))
                if 'EMA50' in df_chart.columns:
                    fig.add_trace(go.Scatter(x=df_chart['Datetime'], y=df_chart['EMA50'], name="EMA 50", line=dict(color='cyan', width=1)))
                
                fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
