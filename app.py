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
    page_title="Quantum Hedge Fund V6.4 - Net Profit Engine",
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

# Konstanta Global Untuk Biaya Transaksi
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
    df['ATR'] = df['ATR'].bfill().fillna(0) 
    return df

@st.cache_data(ttl=5)
def fetch_indodax_live():
    try: return requests.get("https://indodax.com/api/tickers", timeout=5, verify=False).json()['tickers']
    except Exception: return None

@st.cache_data(ttl=30)
def fetch_indodax_klines_safe(symbol, tf, limit, ticker_data):
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
        # Jika gagal, fallback ke return DataFrame kosong untuk keamanan
        return pd.DataFrame()

# ==========================================
# 5. NEURAL NETWORK AI ENGINE (FEE AWARE)
# ==========================================
def ai_neural_quant_brain(df_chart, coin, current_price):
    narasi = f"**🧠 AI Execution Engine: {coin}**\n\nSpot: **Rp {current_price:,}**.\n\n"
    if len(df_chart) < 50: return narasi + "Data belum cukup.", "HOLD"
    
    df = df_chart.copy()
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df.fillna(0, inplace=True) 
    
    # PERBAIKAN LOGIKA: AI belajar mencari pergerakan NAIK yang melebih total Fee Bolak-Balik (0.6%)
    # Harga tutup berikutnya harus lebih besar dari Harga Tutup Saat Ini * 1.006
    df['Target'] = (df['Close'].shift(-1) > (df['Close'] * (1 + (FEE_RATE * 2)))).astype(int)
    
    train_data = df.iloc[:-1]; latest_data = df.iloc[-1:]
    features = ['RSI', 'MACD_Hist', 'BB_Position', 'Volume']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data[features])
    X_latest_scaled = scaler.transform(latest_data[features])
    
    # Model dilatih untuk mendeteksi pergerakan yang "Profit-Bersih"
    model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    
    try:
        model.fit(X_train_scaled, train_data['Target'])
        probabilitas = model.predict_proba(X_latest_scaled)[0]
        prob_turun = probabilitas[0] * 100; prob_naik = probabilitas[1] * 100
        
        narasi += f"- Probabilitas Profit > Fee : **{prob_naik:.1f}%**\n- Probabilitas Terkoreksi: **{prob_turun:.1f}%**\n\n"
        
        if prob_naik > 65:
            narasi += "✅ Deep Learning mendeteksi Momentum Kenaikan Melebihi Spread & Fee bursa."
            konklusi = "BUY"
        elif prob_turun > 65:
            narasi += "❌ Deep Learning mendeteksi Momentum Bearish Kuat."
            konklusi = "SELL"
        else:
            narasi += "⚖️ Algoritma Netral. Potensi kenaikan tidak cukup besar untuk menutupi biaya transaksi (Trading Fee)."
            konklusi = "HOLD"
        return narasi, konklusi
    except Exception as e: return narasi + f"Error Engine: {e}", "ERROR"

# ==========================================
# 6. MAIN DASHBOARD V6.4
# ==========================================
def main():
    with st.sidebar:
        st.markdown("### 🤖 AUTO-PILOT CONTROL")
        auto_pilot_toggle = st.toggle("Aktifkan Auto-Pilot", value=st.session_state.auto_pilot)
        st.session_state.auto_pilot = auto_pilot_toggle
        
        if st.session_state.auto_pilot: st.success("⚡ AUTO-PILOT ON")
        else: st.warning("⏸️ AUTO-PILOT OFF")
            
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
        pilihan_koin = st.selectbox("Pilih Aset Kripto", list(crypto_map.keys()))
        interval_chart = st.selectbox("Timeframe", ["15m", "1h", "4h", "1D"], index=0)

    ticker_koin = crypto_map[pilihan_koin]["ticker"]
    tv_koin = crypto_map[pilihan_koin]["tv"]
    data_live = fetch_indodax_live()
    
    st.title(f"🦅 QUANTUM DESK V6.4 - Realistic Trading")
    
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
                st.plotly_chart(fig, use_container_width=True)
                
            with c_panel:
                st.markdown("### 🧠 AI Analysis")
                narasi_ai, konklusi_ai = ai_neural_quant_brain(df_chart, pilihan_koin, harga_sekarang)
                st.markdown(f"<div class='ai-box'>{narasi_ai}</div>", unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### ⚡ Execution Panel")
                
                buy_amount_idr = st.session_state.buy_amount_idr 
                koin_dimiliki = st.session_state.positions.get(pilihan_koin, {}).get('amount', 0.0)
                sedang_punya_koin = koin_dimiliki > 0
                
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

                # Logika AUTO-PILOT
                if st.session_state.auto_pilot:
                    st.info(f"Mengawasi pasar... (Log: {st.session_state.last_action})")
                    
                    if konklusi_ai == "BUY" and not sedang_punya_koin:
                        if api_key and secret_key:
                            res = indodax_private_api(api_key, secret_key, 'trade', pair=ticker_koin, type='buy', price=int(harga_sekarang*1.01), idr=buy_amount_idr)
                            if res.get('success') == 1:
                                jumlah_koin_kotor = buy_amount_idr / harga_sekarang
                                koin_diterima_bersih = jumlah_koin_kotor * (1 - FEE_RATE)
                                st.session_state.positions[pilihan_koin] = {'amount': koin_diterima_bersih, 'avg_price': harga_sekarang}
                                catat_log("🟢 LIVE AUTO BUY", pilihan_koin, harga_sekarang, koin_diterima_bersih, buy_amount_idr, "-")
                                st.toast("✅ Membeli di Indodax!", icon="🟢")
                        else:
                            if buy_amount_idr <= st.session_state.cash:
                                jumlah_koin_kotor = buy_amount_idr / harga_sekarang
                                koin_diterima_bersih = jumlah_koin_kotor * (1 - FEE_RATE) # Dipotong fee simulasi
                                st.session_state.cash -= buy_amount_idr
                                st.session_state.positions[pilihan_koin] = {'amount': koin_diterima_bersih, 'avg_price': harga_sekarang}
                                catat_log("🟢 SIM AUTO BUY", pilihan_koin, harga_sekarang, koin_diterima_bersih, buy_amount_idr, "-")
                                st.toast("✅ Simulasi Pembelian Berhasil!", icon="🟢")
                        st.session_state.last_action = "Mengeksekusi Pembelian Otomatis..."
                            
                    elif konklusi_ai == "SELL" and sedang_punya_koin:
                        nilai_jual_kotor = koin_dimiliki * harga_sekarang
                        nilai_jual_bersih = nilai_jual_kotor * (1 - FEE_RATE)
                        
                        # Modal awal sudah termasuk fee, jadi kita membandingkan nilai jual bersih dengan nilai beli asli
                        modal_awal_idr = koin_dimiliki * st.session_state.positions[pilihan_koin]['avg_price'] / (1 - FEE_RATE)
                        pnl_bersih_akhir = nilai_jual_bersih - modal_awal_idr
                        
                        if api_key and secret_key:
                            res = indodax_private_api(api_key, secret_key, 'trade', pair=ticker_koin, type='sell', price=int(harga_sekarang*0.99), **{ticker_koin.split('_')[0]: koin_dimiliki})
                            if res.get('success') == 1:
                                catat_log("🔴 LIVE AUTO SELL", pilihan_koin, harga_sekarang, koin_dimiliki, nilai_jual_bersih, pnl_bersih_akhir)
                                del st.session_state.positions[pilihan_koin]
                                st.toast("✅ Menjual di Indodax!", icon="🔴")
                        else:
                            st.session_state.cash += nilai_jual_bersih
                            catat_log("🔴 SIM AUTO SELL", pilihan_koin, harga_sekarang, koin_dimiliki, nilai_jual_bersih, pnl_bersih_akhir)
                            del st.session_state.positions[pilihan_koin]
                            st.toast("✅ Simulasi Penjualan Berhasil!", icon="🔴")
                        st.session_state.last_action = "Mengeksekusi Penjualan Otomatis..."
                        
                # Logika MANUAL
                else:
                    col_buy, col_sell = st.columns(2)
                    with col_buy:
                        if st.button("🟢 MANUAL BUY", use_container_width=True):
                            if api_key and secret_key:
                                res = indodax_private_api(api_key, secret_key, 'trade', pair=ticker_koin, type='buy', price=int(harga_sekarang*1.01), idr=buy_amount_idr)
                                if res.get('success') == 1:
                                    koin_diterima_bersih = (buy_amount_idr / harga_sekarang) * (1 - FEE_RATE)
                                    st.session_state.positions[pilihan_koin] = {'amount': koin_diterima_bersih, 'avg_price': harga_sekarang}
                                    catat_log("🟢 LIVE BUY", pilihan_koin, harga_sekarang, koin_diterima_bersih, buy_amount_idr, "-")
                                    st.rerun()
                                else: st.error(f"Gagal Beli: {res.get('error')}")
                            else:
                                if buy_amount_idr <= st.session_state.cash:
                                    koin_diterima_bersih = (buy_amount_idr / harga_sekarang) * (1 - FEE_RATE)
                                    st.session_state.cash -= buy_amount_idr
                                    if pilihan_koin in st.session_state.positions:
                                        # Averaging jika sudah punya koin
                                        pos_lama = st.session_state.positions[pilihan_koin]
                                        total_koin = pos_lama['amount'] + koin_diterima_bersih
                                        avg_price = ((pos_lama['amount'] * pos_lama['avg_price']) + (koin_diterima_bersih * harga_sekarang)) / total_koin
                                        st.session_state.positions[pilihan_koin] = {'amount': total_koin, 'avg_price': avg_price}
                                    else:
                                        st.session_state.positions[pilihan_koin] = {'amount': koin_diterima_bersih, 'avg_price': harga_sekarang}
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
                st.markdown("### 📋 Active Portfolio (Net of Fees)")
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
                    st.caption("Tidak ada open position.")

        st.markdown("---")
        st.markdown("### 📜 Riwayat Transaksi (Trade History)")
        if st.session_state.trade_history:
            df_history = pd.DataFrame(st.session_state.trade_history)
            df_history = df_history.iloc[::-1].reset_index(drop=True)
            st.dataframe(df_history, use_container_width=True)
        else:
            st.caption("Belum ada transaksi jual/beli yang terekam.")

    if st.session_state.auto_pilot:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
