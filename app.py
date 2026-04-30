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

# Matikan peringatan SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. KONFIGURASI HALAMAN & UI 
# ==========================================
st.set_page_config(
    page_title="Quantum Hedge Fund V5 - Live Execution",
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
    .live-trade-box { background-color: #4A148C; padding: 15px; border-radius: 8px; border: 1px solid #BB86FC; margin-top: 10px; }
    hr { margin-top: 1rem; margin-bottom: 1rem; border-color: #333; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE MANAGEMENT & CREDENTIALS
# ==========================================
if 'capital' not in st.session_state: st.session_state.capital = 1000000000.0  
if 'risk_perc' not in st.session_state: st.session_state.risk_perc = 1.0         
if 'positions' not in st.session_state: st.session_state.positions = {}          
if 'cash' not in st.session_state: st.session_state.cash = 1000000000.0     
if 'data_source_status' not in st.session_state: st.session_state.data_source_status = "Live"

# ==========================================
# 3. INDODAX LIVE TRADE ENGINE (HMAC-SHA512)
# ==========================================
def indodax_private_api(api_key, secret_key, method, **kwargs):
    """Mesin Kriptografi untuk mengirim perintah Trade ke Indodax"""
    if not api_key or not secret_key:
        return {"success": 0, "error": "API Key atau Secret Key kosong."}
        
    url = "https://indodax.com/tapi"
    data = {
        'method': method,
        'timestamp': int(time.time() * 1000),
        'recvWindow': 5000
    }
    data.update(kwargs)
    
    # Proses Enkripsi
    query_string = urllib.parse.urlencode(data)
    signature = hmac.new(secret_key.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha512).hexdigest()
    
    headers = {
        'Key': api_key,
        'Sign': signature
    }
    
    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        return response.json()
    except Exception as e:
        return {"success": 0, "error": str(e)}

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
    df['ATR'] = df['ATR'].bfill().fillna(0) 
    return df

@st.cache_data(ttl=5)
def fetch_indodax_live():
    try:
        url = "https://indodax.com/api/tickers"
        res = requests.get(url, timeout=5, verify=False)
        return res.json()['tickers']
    except Exception: return None

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

@st.cache_data(ttl=60)
def fetch_indodax_klines_safe(symbol, tf, limit, ticker_data):
    try:
        if tf == "1D": tf_api = "1D"; multiplier = 86400; interval_min = 1440
        elif tf == "4h": tf_api = "240"; multiplier = 240 * 60; interval_min = 240
        elif tf == "1h": tf_api = "60"; multiplier = 60 * 60; interval_min = 60
        else: tf_api = "15"; multiplier = 15 * 60; interval_min = 15
        end_time = int(time.time()); start_time = end_time - (limit * multiplier)
        url = f"https://indodax.com/tradingview/history_v2?symbol={symbol}&resolution={tf_api}&from={start_time}&to={end_time}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36', 'Accept': 'application/json'}
        res = requests.get(url, headers=headers, timeout=5, verify=False)
        data = res.json()
        if isinstance(data, dict) and data.get('s') == 'ok':
            st.session_state.data_source_status = "Direct API"
            return pd.DataFrame({'Date': pd.to_datetime(data['t'], unit='s'), 'Open': data['o'], 'High': data['h'], 'Low': data['l'], 'Close': data['c'], 'Volume': data['v']})
        else: raise ValueError("JSON Error")
    except Exception:
        st.session_state.data_source_status = "Synthetic Engine Active"
        return generate_synthetic_klines(ticker_data, limit, interval_min)

# ==========================================
# 5. NEURAL NETWORK AI ENGINE
# ==========================================
def ai_neural_quant_brain(df_chart, coin, current_price):
    narasi = f"**🧠 AI Execution Engine: {coin}**\n\nSpot: **Rp {current_price:,}**.\n\n"
    if len(df_chart) < 50: return narasi + "Data belum cukup.", "⚖️ SYSTEM WARMING UP"
    df = df_chart.copy()
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df.fillna(0, inplace=True) 
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    train_data = df.iloc[:-1]; latest_data = df.iloc[-1:]
    features = ['RSI', 'MACD_Hist', 'BB_Position', 'Volume']
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[features])
    X_latest_scaled = scaler.transform(latest_data[features])
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    try:
        model.fit(X_train_scaled, train_data['Target'])
        probabilitas = model.predict_proba(X_latest_scaled)[0]
        prob_turun = probabilitas[0] * 100; prob_naik = probabilitas[1] * 100
        narasi += f"- Uptrend Prob : **{prob_naik:.1f}%**\n- Downtrend Prob: **{prob_turun:.1f}%**\n\n"
        if prob_naik > 65:
            narasi += "✅ Deep Learning merekomendasikan EKSEKUSI BELI."
            konklusi = "✅ AI CONFIDENCE: STRONG BUY"
        elif prob_turun > 65:
            narasi += "❌ Deep Learning merekomendasikan LIKUIDASI / JUAL."
            konklusi = "❌ AI CONFIDENCE: STRONG SELL"
        else:
            narasi += "⚖️ Algoritma Netral. Eksekusi manual disarankan jika ingin masuk."
            konklusi = "⚖️ AI CONFIDENCE: HOLD"
        return narasi, konklusi
    except Exception as e: return narasi + f"Error: {e}", "⚙️ ERROR"

# ==========================================
# 6. MAIN DASHBOARD V5.0
# ==========================================
def main():
    with st.sidebar:
        st.markdown("### 🔐 LIVE API CREDENTIALS")
        st.warning("⚠️ Masukkan API Key untuk Live Trading. Biarkan kosong untuk mode Simulasi (Paper Trade).")
        api_key = st.text_input("Indodax API Key", type="password")
        secret_key = st.text_input("Indodax Secret Key", type="password")
        
        mode_trading = "🔴 LIVE TRADING ACTIVE" if api_key and secret_key else "🟢 SIMULATION MODE"
        st.markdown(f"**Status Mode:** {mode_trading}")
        
        st.markdown("---")
        st.markdown("### 🏦 Risk Engine")
        new_cap = st.number_input("Target Total AUM (IDR)", value=st.session_state.capital, step=10000000.0)
        st.session_state.capital = new_cap
        st.session_state.risk_perc = st.slider("Max Risk / Trade (%)", 0.1, 5.0, st.session_state.risk_perc, 0.1)
        max_loss = (st.session_state.risk_perc / 100) * st.session_state.capital
        st.info(f"Max Risk Limit: Rp {max_loss:,.0f}")
        
        st.markdown("---")
        crypto_map = {
            "Bitcoin": {"ticker": "btc_idr", "tv": "BTCIDR"},
            "Ethereum": {"ticker": "eth_idr", "tv": "ETHIDR"},
            "Solana": {"ticker": "sol_idr", "tv": "SOLIDR"},
            "Dogecoin": {"ticker": "doge_idr", "tv": "DOGEIDR"}
        }
        pilihan_koin = st.selectbox("Pilih Aset Kripto", list(crypto_map.keys()))
        interval_chart = st.selectbox("Timeframe", ["15m", "1h", "4h", "1D"], index=0)
        
        if st.button("⚡ Force Sync System"): st.cache_data.clear()

    ticker_koin = crypto_map[pilihan_koin]["ticker"]
    tv_koin = crypto_map[pilihan_koin]["tv"]
    data_live = fetch_indodax_live()
    
    st.title(f"🦅 QUANTUM DESK V5.0 - {mode_trading}")
    
    if data_live:
        ticker_data = data_live[ticker_koin]
        harga_sekarang = int(ticker_data['last'])
        
        df_chart = fetch_indodax_klines_safe(tv_koin, interval_chart, 120, ticker_data)
        
        if not df_chart.empty:
            df_chart = calculate_technical_indicators(df_chart)
            
            c_chart, c_panel = st.columns([7, 3])
            
            with c_chart:
                st.markdown(f"### 📈 Institutional Chart - {tv_koin}")
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df_chart['Date'], open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='Spot'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['BB_Upper'], line=dict(color='rgba(255,255,255,0.2)', dash='dash'), name='BB Up'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['BB_Lower'], line=dict(color='rgba(255,255,255,0.2)', dash='dash'), name='BB Low', fill='tonexty'), row=1, col=1)
                colors = ['green' if val >= 0 else 'red' for val in df_chart['MACD_Hist']]
                fig.add_trace(go.Bar(x=df_chart['Date'], y=df_chart['MACD_Hist'], marker_color=colors, name='MACD Hist'), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_chart['Date'], y=df_chart['MACD'], line=dict(color='#2196F3'), name='MACD'), row=2, col=1)
                fig.update_layout(height=650, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#121212", plot_bgcolor="#121212", xaxis_rangeslider_visible=False, font=dict(color="#E0E0E0"), showlegend=False)
                fig.update_yaxes(gridcolor='#333333'); fig.update_xaxes(gridcolor='#333333')
                st.plotly_chart(fig, use_container_width=True)
                
            with c_panel:
                st.markdown("### 🧠 AI Analysis")
                narasi_ai, konklusi_ai = ai_neural_quant_brain(df_chart, pilihan_koin, harga_sekarang)
                st.markdown(f"<div class='ai-box'>{narasi_ai}</div>", unsafe_allow_html=True)
                if "BUY" in konklusi_ai: st.success(konklusi_ai)
                elif "SELL" in konklusi_ai: st.error(konklusi_ai)
                else: st.warning(konklusi_ai)
                    
                st.markdown("---")
                st.markdown("### ⚡ Live Execution Panel")
                
                # Menggunakan AI rekomendasi untuk menentukan harga (Market Price)
                buy_amount_idr = st.number_input("Input Modal Pembelian (IDR)", min_value=10000.0, value=float(max_loss), step=50000.0)
                sell_amount_coin = st.number_input(f"Input Jumlah {pilihan_koin} untuk Dijual", min_value=0.0000, value=0.001, step=0.001, format="%.6f")

                col_buy, col_sell = st.columns(2)
                with col_buy:
                    if st.button("🟢 EXECUTE BUY", use_container_width=True):
                        if api_key and secret_key:
                            # LIVE TRADING LOGIC
                            with st.spinner('Menghubungkan ke server Indodax...'):
                                # Indodax API memerlukan "price" sebagai limit order. Untuk menyerupai market buy, 
                                # kita menembak harga sedikit di atas harga last agar langsung *matched*.
                                eksekusi_harga = int(harga_sekarang * 1.01) 
                                response = indodax_private_api(
                                    api_key, secret_key, 'trade',
                                    pair=ticker_koin, type='buy',
                                    price=eksekusi_harga, idr=buy_amount_idr
                                )
                                if response.get('success') == 1:
                                    st.success(f"Order Beli IDR {buy_amount_idr} Terkirim!")
                                else:
                                    st.error(f"Gagal Live Trade: {response.get('error')}")
                        else:
                            # SIMULATION LOGIC
                            st.info("Simulasi Beli Berhasil. (Masukkan API Key untuk Live Trading)")

                with col_sell:
                    if st.button("🔴 EXECUTE SELL", use_container_width=True):
                        if api_key and secret_key:
                            with st.spinner('Mengeksekusi Jual...'):
                                eksekusi_harga = int(harga_sekarang * 0.99) # Limit di bawah harga untuk instan match
                                response = indodax_private_api(
                                    api_key, secret_key, 'trade',
                                    pair=ticker_koin, type='sell',
                                    price=eksekusi_harga,
                                    **{ticker_koin.split('_')[0]: sell_amount_coin}
                                )
                                if response.get('success') == 1:
                                    st.success(f"Order Jual {sell_amount_coin} {ticker_koin} Terkirim!")
                                else:
                                    st.error(f"Gagal Live Trade: {response.get('error')}")
                        else:
                            st.info("Simulasi Jual Berhasil. (Masukkan API Key untuk Live)")
                            
                # Cek Saldo Real jika API dimasukkan
                if api_key and secret_key:
                    if st.button("💳 Cek Saldo Dompet Real", use_container_width=True):
                        saldo_res = indodax_private_api(api_key, secret_key, 'getInfo')
                        if saldo_res.get('success') == 1:
                            sisa_idr = saldo_res['return']['balance']['idr']
                            st.markdown(f"<div class='live-trade-box'>**Live Balance:**<br>Rp {int(sisa_idr):,}</div>", unsafe_allow_html=True)
                        else:
                            st.error("Gagal mengambil saldo.")

    else: st.error("Koneksi Indodax terputus.")

if __name__ == "__main__":
    main()