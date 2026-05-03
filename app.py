import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import urllib3
import random
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import hmac
import hashlib
import urllib.parse
import os
import joblib

# Matikan peringatan SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. KONFIGURASI HALAMAN & UI 
# ==========================================
st.set_page_config(
    page_title="Quantum Hedge Fund V7.5 - Complete Edition",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 0rem; max-width: 98%; }
    h1, h2, h3, p, span { color: #E0E0E0; font-family: 'Courier New', Courier, monospace; }
    .stMetric-value { color: #00FF00 !important; font-weight: bold; }
    div[data-testid="stMetricDelta"] svg { display: none; }
    .ai-box { background-color: #1A1A1A; padding: 20px; border-left: 5px solid #BB86FC; border-radius: 5px; margin-bottom: 15px;}
    .portfolio-box { background-color: #262730; padding: 15px; border-radius: 8px; border: 1px solid #444; }
    hr { margin-top: 1rem; margin-bottom: 1rem; border-color: #333; }
    </style>
""", unsafe_allow_html=True)

FEE_RATE = 0.003 # 0.3% Taker Fee

# ==========================================
# 2. STATE MANAGEMENT & CREDENTIALS
# ==========================================
if 'capital' not in st.session_state: st.session_state.capital = 1000000000.0  
if 'risk_perc' not in st.session_state: st.session_state.risk_perc = 2.0         
if 'positions' not in st.session_state: st.session_state.positions = {}          
if 'cash' not in st.session_state: st.session_state.cash = 1000000000.0     
if 'data_source_status' not in st.session_state: st.session_state.data_source_status = "Live"
if 'auto_pilot' not in st.session_state: st.session_state.auto_pilot = False 
if 'last_action' not in st.session_state: st.session_state.last_action = "NONE" 
if 'trade_history' not in st.session_state: st.session_state.trade_history = []
if 'buy_amount_idr' not in st.session_state: st.session_state.buy_amount_idr = 0.0
if 'scan_speed' not in st.session_state: st.session_state.scan_speed = 5
if 'atr_multiplier' not in st.session_state: st.session_state.atr_multiplier = 2.0

# ==========================================
# 3. INDODAX LIVE TRADE ENGINE
# ==========================================
def indodax_private_api(api_key, secret_key, method, **kwargs):
    if not api_key or not secret_key: return {"success": 0, "error": "API Key/Secret kosong."}
    url = "https://indodax.com/tapi"
    data = {'method': method, 'timestamp': int(time.time() * 1000), 'recvWindow': 5000}
    data.update(kwargs)
    query_string = urllib.parse.urlencode(data)
    signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha512).hexdigest()
    headers = {'Key': api_key, 'Sign': signature}
    try: return requests.post(url, headers=headers, data=data, timeout=10).json()
    except Exception as e: return {"success": 0, "error": str(e)}

# ==========================================
# 4. MATH ENGINE & DATA PIPELINE
# ==========================================
def calculate_technical_indicators(df):
    if df.empty: return df
    df = df.sort_values('Date').reset_index(drop=True)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (std_20 * 2)
    df['BB_Lower'] = df['SMA_20'] - (std_20 * 2)
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = np.max(ranges, axis=1).rolling(14).mean()
    
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df = df.bfill().fillna(0) 
    return df

@st.cache_data(ttl=5)
def fetch_indodax_live():
    try: return requests.get("https://indodax.com/api/tickers", timeout=5, verify=False).json()['tickers']
    except Exception: return None

@st.cache_data(ttl=3600)
def fetch_global_sentiment():
    try:
        res = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        data = res.json()
        return int(data['data'][0]['value'])
    except Exception:
        return 50

def generate_synthetic_klines(ticker_data, limit=120, interval_minutes=15):
    try:
        current_price = float(ticker_data['last']); high_price = float(ticker_data['high']); low_price = float(ticker_data['low'])
        dates = [datetime.now() - timedelta(minutes=i*interval_minutes) for i in range(limit, -1, -1)]
        data = []
        sim_price = low_price + ((high_price - low_price) * 0.5) 
        for i, date in enumerate(dates):
            if i == len(dates) - 1:
                close_p = current_price; open_p = sim_price
                high_p = max(open_p, close_p) * (1 + random.uniform(0, 0.002)); low_p = min(open_p, close_p) * (1 - random.uniform(0, 0.002))
            else:
                volatility = (high_price - low_price) * 0.05
                open_p = sim_price; close_p = max(low_price, min(high_price, open_p + random.uniform(-volatility, volatility)))
                high_p = max(open_p, close_p) + random.uniform(0, volatility * 0.5); low_p = min(open_p, close_p) - random.uniform(0, volatility * 0.5)
                sim_price = close_p
            data.append([date, open_p, high_p, low_p, close_p, random.uniform(10, 1000)])
        return pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_indodax_klines_safe(symbol, tf, limit, ticker_data=None):
    interval_min = 15
    try:
        if tf == "1D": tf_api = "1D"; multiplier = 86400; interval_min = 1440
        elif tf == "4h": tf_api = "240"; multiplier = 240 * 60; interval_min = 240
        elif tf == "1h": tf_api = "60"; multiplier = 60 * 60; interval_min = 60
        else: tf_api = "15"; multiplier = 15 * 60; interval_min = 15
        
        end_time = int(time.time()); start_time = end_time - (limit * multiplier)
        url = f"https://indodax.com/tradingview/history_v2?symbol={symbol}&resolution={tf_api}&from={start_time}&to={end_time}"
        headers = {'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json'}
        res = requests.get(url, headers=headers, timeout=5, verify=False)
        data = res.json()
        
        if isinstance(data, dict) and data.get('s') == 'ok':
            st.session_state.data_source_status = "Direct API"
            return pd.DataFrame({'Date': pd.to_datetime(data['t'], unit='s'), 'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Close': data['c'], 'Volume': data['v']})
        else: raise ValueError("JSON Error")
    except Exception:
        st.session_state.data_source_status = "Synthetic Engine Active"
        if ticker_data: return generate_synthetic_klines(ticker_data, limit, interval_min)
        else: return pd.DataFrame()

# ==========================================
# 5. NEURAL NETWORK AI ENGINE 
# ==========================================
def ai_neural_quant_brain(df_chart, coin, current_price, timeframe, sentimen_global):
    narasi = f"**🧠 AI Execution Engine: {coin} ({timeframe})**\n\nSpot: **Rp {current_price:,}** | Sentimen Global: **{sentimen_global}/100**\n\n"
    if len(df_chart) < 50: return narasi + "Data belum cukup untuk analisis.", "HOLD"
    
    df = df_chart.copy()
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df.fillna(0, inplace=True) 
    df['Sentiment'] = sentimen_global
    
    LOOKAHEAD_WINDOW = 4
    df['Future_Max'] = df['Close'].rolling(window=LOOKAHEAD_WINDOW).max().shift(-LOOKAHEAD_WINDOW)
    df['Target'] = (df['Future_Max'] > (df['Close'] * (1 + (FEE_RATE * 2)))).astype(int)
    
    train_data = df.dropna(subset=['Future_Max'])
    latest_data = df.iloc[-1:] 
    
    features = ['RSI', 'MACD_Hist', 'BB_Position', 'Volume', 'OBV', 'Sentiment']
    X_train = train_data[features]
    y_train = train_data['Target']
    X_latest = latest_data[features]
    
    model_file = f'ai_model_{coin}_{timeframe}_v3.pkl' 
    scaler_file = f'ai_scaler_{coin}_{timeframe}_v3.pkl'
    
    if os.path.exists(model_file) and os.path.exists(scaler_file):
        scaler = joblib.load(scaler_file)
        model = joblib.load(model_file)
        narasi += f"💾 *Memori AI V3 ({timeframe}) dimuat...*\n"
    else:
        scaler = StandardScaler()
        model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=1, random_state=42)
        narasi += f"🌱 *Menciptakan jaringan saraf Visi Masa Depan untuk {coin}...*\n"

    try:
        if not hasattr(scaler, 'n_samples_seen_'):
            X_train_scaled = scaler.fit_transform(X_train)
        else:
            X_train_scaled = scaler.transform(X_train)
            
        X_latest_scaled = scaler.transform(X_latest)
        model.partial_fit(X_train_scaled, y_train, classes=np.array([0, 1]))
        
        joblib.dump(scaler, scaler_file)
        joblib.dump(model, model_file)
        
        probabilitas = model.predict_proba(X_latest_scaled)[0]
        prob_turun = probabilitas[0] * 100
        prob_naik = probabilitas[1] * 100
        
        narasi += f"- Probabilitas Profit (Net) : **{prob_naik:.1f}%**\n- Probabilitas Terkoreksi: **{prob_turun:.1f}%**\n\n"
        
        if prob_naik > 60:
            narasi += "✅ Jaringan Saraf mendeteksi tren kenaikan valid (Mampu menembus Fee)."
            konklusi = "BUY"
        elif prob_turun > 60:
            narasi += "❌ Jaringan Saraf merekomendasikan pelepasan aset."
            konklusi = "SELL"
        else:
            narasi += "⚖️ Pasar tidak menentu. AI menahan diri (HOLD)."
            konklusi = "HOLD"
            
        return narasi, konklusi
    except Exception as e: 
        return narasi + f"⚠️ Kesalahan Kognitif AI: {e}", "ERROR"

# ==========================================
# 6. MAIN DASHBOARD V7.5 (TABS)
# ==========================================
def main():
    with st.sidebar:
        st.markdown("### 🤖 AUTO-PILOT CONTROL")
        auto_pilot_toggle = st.toggle("Aktifkan Auto-Pilot", value=st.session_state.auto_pilot)
        st.session_state.auto_pilot = auto_pilot_toggle
        
        if st.session_state.auto_pilot: st.success("⚡ AUTO-PILOT ON")
        else: st.warning("⏸️ AUTO-PILOT OFF")
        
        scan_speed = st.slider("⚡ Kecepatan Pindai Bot (Detik)", 3, 60, 5, 1)
        st.session_state.scan_speed = scan_speed
        st.session_state.atr_multiplier = st.slider("🛡️ Jarak Trailing Stop (Pengali ATR)", 1.0, 5.0, 2.0, 0.1)
            
        st.markdown("---")
        st.markdown("### 🔐 LIVE API CREDENTIALS")
        api_key = st.text_input("Indodax API Key", type="password")
        secret_key = st.text_input("Indodax Secret Key", type="password")
        mode_trading = "🔴 LIVE TRADING" if api_key and secret_key else "🟢 SIMULATION"
        st.markdown(f"**Status Mode:** {mode_trading}")

        st.markdown("---")
        st.markdown("### 🏦 Capital & Sizing Engine")
        if st.button("🔄 Reset Portfolio & History"):
            st.session_state.capital = 1000000000.0
            st.session_state.cash = 1000000000.0
            st.session_state.positions = {}
            st.session_state.trade_history = []
            st.rerun()

        new_cap = st.number_input("Target Total AUM (IDR)", value=st.session_state.capital, step=10000000.0)
        st.session_state.capital = new_cap
        
        position_size_perc = st.slider("Alokasi Beli per Trade (%)", 10.0, 100.0, 50.0, 5.0)
        st.session_state.risk_perc = st.slider("Max Cut-Loss Tolerance (%)", 0.1, 10.0, 2.0, 0.1)
        
        st.session_state.buy_amount_idr = st.session_state.capital * (position_size_perc / 100)
        max_loss_allowed = st.session_state.buy_amount_idr * (st.session_state.risk_perc / 100)
        
        st.info(f"**Dana Dieksekusi:** Rp {st.session_state.buy_amount_idr:,.0f}\n\n**Batas Toleransi Kerugian:** Rp {max_loss_allowed:,.0f}")
        
        st.markdown("---")
        crypto_map = {
            "Bitcoin": {"ticker": "btc_idr", "tv": "BTCIDR"},
            "Ethereum": {"ticker": "eth_idr", "tv": "ETHIDR"},
            "Solana": {"ticker": "sol_idr", "tv": "SOLIDR"}
        }

    st.title(f"🦅 QUANTUM DESK V7.5 - Complete Edition")

    # ==========================================
    # PEMBUATAN 2 TABS UTAMA
    # ==========================================
    tab_live, tab_backtest = st.tabs(["🔴 Live Trading Dashboard", "⏪ Mesin Backtesting"])

    # ----------------------------------------------------------------------------------------
    # TAB 1: LIVE DASHBOARD
    # ----------------------------------------------------------------------------------------
    with tab_live:
        pilihan_koin = st.selectbox("Pilih Aset Kripto (Tampilan Manual)", list(crypto_map.keys()), key="live_koin")
        interval_chart = st.selectbox("Timeframe", ["15m", "1h", "4h", "1D"], index=0, key="live_tf")

        ticker_koin = crypto_map[pilihan_koin]["ticker"]
        tv_koin = crypto_map[pilihan_koin]["tv"]
        data_live = fetch_indodax_live()
        
        if data_live:
            ticker_data = data_live[ticker_koin]
            harga_sekarang = int(ticker_data['last'])
            df_chart = fetch_indodax_klines_safe(tv_koin, interval_chart, 120, ticker_data)
            
            if not df_chart.empty:
                df_chart = calculate_technical_indicators(df_chart)
                c_chart, c_panel = st.columns([7, 3])
                
                with c_chart:
                    st.markdown(f"### 📈 Institutional Chart - {tv_koin}")
                    if "Synthetic" in st.session_state.data_source_status:
                        st.warning("⚠️ Indodax memblokir grafik. Menampilkan grafik cadangan (Sintetis).")
                        
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df_chart['Date'], open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Spot'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['BB_Upper'], line=dict(color='rgba(255,255,255,0.2)', dash='dash'), name='BB Up'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['BB_Lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dash'), name='BB Low', fill='tonexty'), row=1, col=1)
                    colors = ['green' if val >= 0 else 'red' for val in df_chart['MACD_Hist']]
                    fig.add_trace(go.Bar(x=df_chart['Date'], y=df_chart['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MACD'], line=dict(color='#2196F3'), name='MACD'), row=2, col=1)
                    fig.update_layout(height=650, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#121212", plot_bgcolor="#121212", xaxis_rangeslider_visible=False, font=dict(color="#E0E0E0"), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                with c_panel:
                    st.markdown("### 🧠 AI Analysis")
                    sentimen_sekarang = fetch_global_sentiment()
                    narasi_ai, konklusi_ai = ai_neural_quant_brain(df_chart, pilihan_koin, harga_sekarang, interval_chart, sentimen_sekarang)
                    st.markdown(f"<div class='ai-box'>{narasi_ai}</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### ⚡ Execution Panel")

            buy_amount_idr = st.session_state.buy_amount_idr 
            MINIMAL_ORDER = 10000.0
            
            if buy_amount_idr < MINIMAL_ORDER:
                st.error(f"⚠️ Peringatan: Ukuran Eksekusi Anda (Rp {int(buy_amount_idr):,}) terlalu kecil. Indodax mewajibkan minimal Rp 10.000 per transaksi.")
                    
            def catat_log(aksi, koin, harga, jumlah, nilai, pnl="0"):
                st.session_state.trade_history.append({
                    "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Aksi": aksi,
                    "Koin": koin,
                    "Harga (IDR)": f"Rp {int(harga):,}",
                    "Kuantitas Koin": f"{jumlah:.6f}",
                    "Total IDR (Bersih)": f"Rp {int(nilai):,}",
                    "PnL (Net)": f"Rp {int(pnl):,}" if pnl != "0" and pnl != "-" else "-"
                })

            if st.session_state.auto_pilot:
                st.info(f"⚡ Pemindai Multi-Koin Aktif! Mengawasi BTC, ETH, dan SOL... (Log: {st.session_state.last_action})")
                
                for koin_target, data_koin in crypto_map.items():
                    ticker_target = data_koin["ticker"]
                    tv_target = data_koin["tv"]
                    
                    koin_dimiliki_ap = st.session_state.positions.get(koin_target, {}).get('amount', 0.0)
                    sedang_punya_koin_ap = koin_dimiliki_ap > 0
                    
                    if data_live and ticker_target in data_live:
                        harga_sekarang_ap = int(data_live[ticker_target]['last'])
                        df_chart_ap = fetch_indodax_klines_safe(tv_target, interval_chart, 120, data_live[ticker_target])
                        
                        if not df_chart_ap.empty:
                            df_chart_ap = calculate_technical_indicators(df_chart_ap)
                            _, konklusi_ai_ap = ai_neural_quant_brain(df_chart_ap, koin_target, harga_sekarang_ap, interval_chart, sentimen_sekarang)
                            
                            if konklusi_ai_ap == "BUY" and not sedang_punya_koin_ap:
                                if buy_amount_idr >= MINIMAL_ORDER:
                                    if api_key and secret_key:
                                        res = indodax_private_api(api_key, secret_key, 'trade', pair=ticker_target, type='buy', price=int(harga_sekarang_ap*1.01), idr=buy_amount_idr)
                                        if res.get('success') == 1:
                                            koin_diterima_bersih = (buy_amount_idr / harga_sekarang_ap) * (1 - FEE_RATE)
                                            st.session_state.positions[koin_target] = {'amount': koin_diterima_bersih, 'avg_price': harga_sekarang_ap, 'highest_price': harga_sekarang_ap}
                                            catat_log("🟢 LIVE AUTO BUY", koin_target, harga_sekarang_ap, koin_diterima_bersih, buy_amount_idr, "-")
                                            st.toast(f"✅ Auto-Buy {koin_target} Berhasil!", icon="🟢")
                                    else:
                                        if buy_amount_idr <= st.session_state.cash:
                                            koin_diterima_bersih = (buy_amount_idr / harga_sekarang_ap) * (1 - FEE_RATE)
                                            st.session_state.cash -= buy_amount_idr
                                            st.session_state.positions[koin_target] = {'amount': koin_diterima_bersih, 'avg_price': harga_sekarang_ap, 'highest_price': harga_sekarang_ap}
                                            catat_log("🟢 SIM AUTO BUY", koin_target, harga_sekarang_ap, koin_diterima_bersih, buy_amount_idr, "-")
                                            st.toast(f"✅ Simulasi Auto-Buy {koin_target} Berhasil!", icon="🟢")
                                    st.session_state.last_action = f"Membeli {koin_target}..."

                            elif sedang_punya_koin_ap:
                                harga_tercatat = st.session_state.positions[koin_target].get('highest_price', st.session_state.positions[koin_target]['avg_price'])
                                if harga_sekarang_ap > harga_tercatat:
                                    st.session_state.positions[koin_target]['highest_price'] = harga_sekarang_ap
                                    
                                harga_tertinggi = st.session_state.positions[koin_target].get('highest_price', harga_sekarang_ap)
                                harga_beli_rata2_ap = st.session_state.positions[koin_target]['avg_price']
                                atr_sekarang = df_chart_ap['ATR'].iloc[-1]
                                
                                batas_trailing_stop = harga_tertinggi - (atr_sekarang * st.session_state.atr_multiplier)
                                batas_take_profit_ap = harga_beli_rata2_ap * (1 + (FEE_RATE * 2) + 0.001)
                                
                                if (konklusi_ai_ap == "SELL" and harga_sekarang_ap >= batas_take_profit_ap) or (harga_sekarang_ap <= batas_trailing_stop):
                                    nilai_jual_kotor_ap = koin_dimiliki_ap * harga_sekarang_ap
                                    nilai_jual_bersih_ap = nilai_jual_kotor_ap * (1 - FEE_RATE)
                                    modal_awal_idr_ap = koin_dimiliki_ap * harga_beli_rata2_ap / (1 - FEE_RATE)
                                    pnl_bersih_akhir_ap = nilai_jual_bersih_ap - modal_awal_idr_ap
                                    
                                    aksi_jual = "🔴 LIVE AUTO SELL" if konklusi_ai_ap == "SELL" else "🛡️ LIVE TRAILING STOP"
                                    aksi_jual_sim = "🔴 SIM AUTO SELL" if konklusi_ai_ap == "SELL" else "🛡️ SIM TRAILING STOP"
                                    
                                    if api_key and secret_key:
                                        res = indodax_private_api(api_key, secret_key, 'trade', pair=ticker_target, type='sell', price=int(harga_sekarang_ap*0.99), **{ticker_target.split('_')[0]: koin_dimiliki_ap})
                                        if res.get('success') == 1:
                                            catat_log(aksi_jual, koin_target, harga_sekarang_ap, koin_dimiliki_ap, nilai_jual_bersih_ap, pnl_bersih_akhir_ap)
                                            del st.session_state.positions[koin_target]
                                            st.toast(f"✅ Sell {koin_target} Dieksekusi!", icon="🔴")
                                    else:
                                        st.session_state.cash += nilai_jual_bersih_ap
                                        catat_log(aksi_jual_sim, koin_target, harga_sekarang_ap, koin_dimiliki_ap, nilai_jual_bersih_ap, pnl_bersih_akhir_ap)
                                        del st.session_state.positions[koin_target]
                                        st.toast(f"✅ Simulasi Sell {koin_target} Berhasil!", icon="🔴")
                                    st.session_state.last_action = f"Menjual {koin_target} (PnL: Rp {int(pnl_bersih_akhir_ap):,})."
                                else:
                                    st.session_state.last_action = f"Mengamankan {koin_target} (Batas Perlindungan: Rp {int(batas_trailing_stop):,})."
            else:
                col_buy, col_sell = st.columns(2)
                with col_buy:
                    if st.button("🟢 MANUAL BUY", use_container_width=True):
                        if api_key and secret_key:
                            res = indodax_private_api(api_key, secret_key, 'trade', pair=ticker_koin, type='buy', price=int(harga_sekarang*1.01), idr=buy_amount_idr)
                            if res.get('success') == 1:
                                koin_diterima_bersih = (buy_amount_idr / harga_sekarang) * (1 - FEE_RATE)
                                st.session_state.positions[pilihan_koin] = {'amount': koin_diterima_bersih, 'avg_price': harga_sekarang, 'highest_price': harga_sekarang}
                                catat_log("🟢 LIVE BUY", pilihan_koin, harga_sekarang, koin_diterima_bersih, buy_amount_idr, "-")
                                st.rerun()
                            else: st.error(f"Gagal Beli: {res.get('error')}")
                        else:
                            if buy_amount_idr <= st.session_state.cash:
                                koin_diterima_bersih = (buy_amount_idr / harga_sekarang) * (1 - FEE_RATE)
                                st.session_state.cash -= buy_amount_idr
                                if pilihan_koin in st.session_state.positions:
                                    pos_lama = st.session_state.positions[pilihan_koin]
                                    total_koin = pos_lama['amount'] + koin_diterima_bersih
                                    avg_price = ((pos_lama['amount'] * pos_lama['avg_price']) + (koin_diterima_bersih * harga_sekarang)) / total_koin
                                    tertinggi_baru = max(pos_lama.get('highest_price', harga_sekarang), harga_sekarang)
                                    st.session_state.positions[pilihan_koin] = {'amount': total_koin, 'avg_price': avg_price, 'highest_price': tertinggi_baru}
                                else:
                                    st.session_state.positions[pilihan_koin] = {'amount': koin_diterima_bersih, 'avg_price': harga_sekarang, 'highest_price': harga_sekarang}
                                catat_log("🟢 SIM BUY", pilihan_koin, harga_sekarang, koin_diterima_bersih, buy_amount_idr, "-")
                                st.rerun()
                with col_sell:
                    if st.button("🔴 MANUAL SELL", use_container_width=True):
                        if pilihan_koin in st.session_state.positions:
                            koin_dimiliki_manual = st.session_state.positions[pilihan_koin]['amount']
                            harga_beli_rata2 = st.session_state.positions[pilihan_koin]['avg_price']
                            
                            nilai_jual_bersih = (koin_dimiliki_manual * harga_sekarang) * (1 - FEE_RATE)
                            modal_awal_idr = koin_dimiliki_manual * harga_beli_rata2 / (1 - FEE_RATE)
                            pnl_bersih_akhir = nilai_jual_bersih - modal_awal_idr
                            
                            if api_key and secret_key:
                                res = indodax_private_api(api_key, secret_key, 'trade', pair=ticker_koin, type='sell', price=int(harga_sekarang*0.99), **{ticker_koin.split('_')[0]: koin_dimiliki_manual})
                                if res.get('success') == 1:
                                    catat_log("🔴 LIVE SELL", pilihan_koin, harga_sekarang, koin_dimiliki_manual, nilai_jual_bersih, pnl_bersih_akhir)
                                    del st.session_state.positions[pilihan_koin] 
                                    st.rerun()
                                else: st.error(f"Gagal Jual: {res.get('error')}")
                            else:
                                st.session_state.cash += nilai_jual_bersih
                                catat_log("🔴 SIM SELL", pilihan_koin, harga_sekarang, koin_dimiliki_manual, nilai_jual_bersih, pnl_bersih_akhir)
                                del st.session_state.positions[pilihan_koin] 
                                st.rerun()

            st.markdown("---")
            st.markdown("### 📋 Portofolio & Saldo (Wallet)")
            
            if api_key and secret_key:
                st.markdown("#### 🏦 Saldo Asli Indodax Anda")
                info_wallet = indodax_private_api(api_key, secret_key, 'getInfo')
                
                if info_wallet.get('success') == 1:
                    saldo_asli = info_wallet['return']['balance']
                    idr_asli = float(saldo_asli.get('idr', 0))
                    
                    st.session_state.cash = idr_asli 
                    
                    st.info(f"💵 **Uang Kas (IDR):** Rp {idr_asli:,.0f}")
                    
                    koin_ditemukan = False
                    for koin_nama, data_koin in crypto_map.items():
                        simbol = data_koin['ticker'].split('_')[0] 
                        jumlah = float(saldo_asli.get(simbol, 0))
                        if jumlah > 0:
                            st.success(f"🪙 **{koin_nama}:** {jumlah:.6f}")
                            koin_ditemukan = True
                    
                    if not koin_ditemukan:
                        st.caption("Belum ada koin kripto utama di dompet Indodax Anda.")
                else:
                    st.error(f"Gagal memuat dompet: {info_wallet.get('error')}")
            else:
                st.markdown("#### 🏦 Saldo Simulasi (Virtual)")
                st.info(f"💵 **Uang Kas Simulasi (IDR):** Rp {st.session_state.cash:,.0f}")

            st.markdown("#### 🤖 Posisi Terbuka (Dikelola Bot)")
            if st.session_state.positions:
                for koin, data in st.session_state.positions.items():
                    hrg_koin_ini = int(data_live[crypto_map[koin]["ticker"]]['last'])
                    nilai_jual_bersih = (data['amount'] * hrg_koin_ini) * (1 - FEE_RATE)
                    modal_awal_idr = (data['amount'] * data['avg_price']) / (1 - FEE_RATE)
                    
                    pnl_asli = nilai_jual_bersih - modal_awal_idr
                    pnl_persen = (pnl_asli / modal_awal_idr) * 100
                    
                    warna = "#00FF00" if pnl_asli >= 0 else "#FF0000"
                    st.markdown(f"<div class='portfolio-box'><strong>{koin}</strong><br>Koin Diterima Bersih: {data['amount']:.5f} | Avg: Rp {data['avg_price']:,.0f}<br>Estimasi Jual Tunai: Rp {nilai_jual_bersih:,.0f} <span style='color:{warna};'>({pnl_persen:+.2f}%)</span></div>", unsafe_allow_html=True)
            else:
                st.caption("Bot sedang tidak memegang posisi apa pun.")

            st.markdown("---")
            st.markdown("### 📜 Riwayat Transaksi (Trade History)")
            if st.session_state.trade_history:
                df_history = pd.DataFrame(st.session_state.trade_history)
                df_history = df_history.iloc[::-1].reset_index(drop=True)
                st.dataframe(df_history, use_container_width=True)
            else:
                st.caption("Belum ada transaksi jual/beli yang terekam.")

    # ----------------------------------------------------------------------------------------
    # TAB 2: BACKTESTING ROOM (FITUR BARU)
    # ----------------------------------------------------------------------------------------
    with tab_backtest:
        st.markdown("### ⏪ Mesin Waktu Backtesting (Penguji Strategi)")
        st.markdown("Simulasikan kecerdasan AI dan ketahanan Trailing Stop-Loss Anda pada data harga masa lalu. Simulasi ini menggunakan **modal awal fiktif Rp 10.000.000**.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            bt_koin = st.selectbox("Koin untuk diuji", list(crypto_map.keys()), key="bt_coin")
        with col2:
            bt_durasi = st.selectbox("Durasi Data Masa Lalu", ["7 Hari", "14 Hari", "30 Hari"], index=0)
        with col3:
            bt_tf = st.selectbox("Timeframe Analisis", ["15m", "1h", "4h"], index=1, key="bt_tf")
            
        if st.button("▶️ JALANKAN SIMULASI", type="primary", use_container_width=True):
            map_hari = {"7 Hari": 7, "14 Hari": 14, "30 Hari": 30}
            hari = map_hari[bt_durasi]
            
            if bt_tf == "15m": limit_lilin = hari * 24 * 4
            elif bt_tf == "1h": limit_lilin = hari * 24
            else: limit_lilin = hari * 6
            
            limit_lilin = min(limit_lilin, 1000) 
            
            with st.spinner(f"⏳ Mengunduh {limit_lilin} data masa lalu dari Indodax..."):
                tv_simbol = crypto_map[bt_koin]["tv"]
                
                # PERBAIKAN: Mengambil data acuan harga saat ini untuk disuntikkan ke mesin sintetis
                bt_ticker = crypto_map[bt_koin]["ticker"]
                bt_ticker_data = data_live[bt_ticker] if data_live else None
                
                # Menambahkan parameter bt_ticker_data agar mesin cadangan bisa menyala jika diblokir
                df_history_bt = fetch_indodax_klines_safe(tv_simbol, bt_tf, limit_lilin, bt_ticker_data)
                
            if df_history_bt is not None and not df_history_bt.empty:
                # PERBAIKAN: Memberitahu pengguna jika kita terpaksa menggunakan data sintetis
                if "Synthetic" in st.session_state.data_source_status:
                    st.warning("⚠️ Indodax menolak permintaan data riwayat yang sangat besar. Menggunakan Mesin Data Sintetis untuk simulasi Backtesting.")
                else:
                    st.success("✅ Data asli berhasil diunduh! Menjalankan simulasi AI...")
                
                df_history_bt = calculate_technical_indicators(df_history_bt)
                
                modal_awal = 10000000.0
                kas_virtual = modal_awal
                koin_virtual = 0.0
                harga_beli_avg = 0.0
                harga_tertinggi_virtual = 0.0
                total_trade = 0
                trade_menang = 0
                log_simulasi = []
                
                for i in range(50, len(df_history_bt)):
                    data_saat_ini = df_history_bt.iloc[:i+1]
                    baris_terakhir = data_saat_ini.iloc[-1]
                    harga_sekarang_bt = baris_terakhir['Close']
                    waktu = baris_terakhir['Date']
                    atr_sekarang_bt = baris_terakhir['ATR']
                    
                    _, keputusan = ai_neural_quant_brain(data_saat_ini, bt_koin, harga_sekarang_bt, bt_tf, 50)
                    
                    if keputusan == "BUY" and koin_virtual == 0:
                        ukuran_beli = kas_virtual * 0.50 
                        koin_kotor = ukuran_beli / harga_sekarang_bt
                        koin_virtual = koin_kotor * (1 - FEE_RATE)
                        kas_virtual -= ukuran_beli
                        harga_beli_avg = harga_sekarang_bt
                        harga_tertinggi_virtual = harga_sekarang_bt
                        log_simulasi.append(f"[{waktu}] 🟢 BELI: Harga Rp {int(harga_sekarang_bt):,}")
                        
                    elif koin_virtual > 0:
                        if harga_sekarang_bt > harga_tertinggi_virtual:
                            harga_tertinggi_virtual = harga_sekarang_bt
                            
                        batas_ts = harga_tertinggi_virtual - (atr_sekarang_bt * st.session_state.atr_multiplier)
                        batas_tp = harga_beli_avg * (1 + (FEE_RATE * 2) + 0.001)
                        
                        if (keputusan == "SELL" and harga_sekarang_bt >= batas_tp) or (harga_sekarang_bt <= batas_ts):
                            nilai_jual_kotor = koin_virtual * harga_sekarang_bt
                            nilai_jual_bersih = nilai_jual_kotor * (1 - FEE_RATE)
                            pnl_trade = nilai_jual_bersih - (koin_virtual * harga_beli_avg / (1 - FEE_RATE))
                            
                            kas_virtual += nilai_jual_bersih
                            koin_virtual = 0.0
                            total_trade += 1
                            if pnl_trade > 0: trade_menang += 1
                            
                            alasan = "Take Profit AI" if keputusan == "SELL" else "Trailing Stop ATR"
                            log_simulasi.append(f"[{waktu}] 🔴 JUAL ({alasan}): Harga Rp {int(harga_sekarang_bt):,} | PnL: Rp {int(pnl_trade):,}")

                pnl_bersih_total = kas_virtual - modal_awal
                win_rate = (trade_menang / total_trade * 100) if total_trade > 0 else 0
                
                st.markdown("### 📊 Hasil Backtesting")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Modal Awal", f"Rp {int(modal_awal):,}")
                c2.metric("Saldo Akhir (Estimasi)", f"Rp {int(kas_virtual):,}", f"{int(pnl_bersih_total):,}")
                c3.metric("Total Transaksi Selesai", total_trade)
                c4.metric("Win Rate", f"{win_rate:.1f}%")
                
                with st.expander("Lihat Rincian Jurnal Perdagangan Virtual"):
                    for catatan in log_simulasi:
                        st.text(catatan)
            else:
                st.error("Gagal mengambil data masa lalu. Indodax mungkin sedang membatasi koneksi dan mesin sintetis gagal dimuat.")

    if st.session_state.auto_pilot:
        time.sleep(st.session_state.scan_speed)
        st.rerun()

if __name__ == "__main__":
    main()
