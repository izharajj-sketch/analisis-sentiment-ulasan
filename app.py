import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud 
import numpy as np

# Library untuk Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

# Download resource NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

st.set_page_config(
    page_title="Dashboard Analisis Sentimen + ML",
    page_icon="🤖",
    layout="wide"
)

# --- Kamus Kata ---
positive_words = ["bagus", "lezat", "keren", "enak", "puas", "mantap", "cepat", "nikmat", "bersih", "rapi", "ramah", "love", "manis", "pas", "premium", "juara", "lembut", "renyah", "creamy", "halus", "harum", "segar", "asli", "pekat", "murni", "favorit", "langganan", "terbaik", "bahagia", "cantik", "mewah", "murah", "banyak", "yummy", "sweet"]
negative_words = ["hancur", "jelek", "rusak", "ramai", "kurang", "parah", "lambat", "kecewa", "kotor", "tidak", "lama", "marah", "pahit", "asam", "busuk", "kasar", "grainy", "crumbly", "lembek", "basah", "lummer", "eneg", "mahal", "basi", "kadaluarsa", "buatan", "apek", "aneh", "kesal", "lengket", "hambar", "rugi", "zonk", "penyok", "salah"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    return text

def detect_sentiment(text):
    tokens = word_tokenize(clean_text(text))
    pos_count = sum(1 for w in tokens if w in positive_words)
    neg_count = sum(1 for w in tokens if w in negative_words)
    
    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Netral"

# --- Sidebar ---
st.sidebar.title("🗂️ Menu Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "🏠 Dashboard", 
        "📥 Input Data", 
        "📊 Visualisasi", 
        "🔍 Data & Filter", 
        "🧠 Training Model",
        "📈 Evaluasi Model",
        "💾 Unduh Data"
    ]
)

# --- Inisialisasi Session State ---
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=["text", "sentimen"])

# --- Logika Menu ---

if menu == "🏠 Dashboard":
    st.title("🏠 Dashboard Analisis Sentimen")
    df = st.session_state.df
    if df.empty:
        st.info("👋 Selamat datang! Silahkan input data terlebih dahulu di menu **📥 Input Data**")
    else:
        total = len(df)
        pos = (df["sentimen"] == "Positive").sum()
        neg = (df["sentimen"] == "Negative").sum()
        net = (df["sentimen"] == "Netral").sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📝 Total Ulasan", total) 
        col2.metric("😊 Positive", pos) 
        col3.metric("😡 Negative", neg) 
        col4.metric("😐 Netral", net)

elif menu == "📥 Input Data":
    st.title("📥 Input Data Ulasan") 
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✍️ Input Manual")
        manual_text = st.text_area("Masukin ulasan (1 ulasan per baris)", height=150, placeholder="Contoh:\nMakanan sangat enak\nPelayanan lambat")
    
    with col2:
        st.subheader("📁 Unggah File")
        uploaded_file = st.file_uploader("Unggah file CSV anda", type=['csv'])
    
    df_preview = None
    selected_col = None

    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df_preview = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        
        st.success(f"✅ File '{uploaded_file.name}' berhasil diunggah.")
        selected_col = st.selectbox("🎯 Pilih kolom ulasan:", options=df_preview.columns)

    if st.button("🚀 Proses Analisis Sentimen", use_container_width=True):
        data_frames = []
        if manual_text.strip():
            df_manual = pd.DataFrame({"text": manual_text.split("\n")})
            data_frames.append(df_manual)
        if df_preview is not None and selected_col:
            df_to_add = df_preview[[selected_col]].rename(columns={selected_col: "text"})
            data_frames.append(df_to_add)
            
        if not data_frames:
            st.warning("⚠️ Masukkan data terlebih dahulu!")
        else:
            df_final = pd.concat(data_frames, ignore_index=True)
            df_final = df_final.dropna(subset=["text"])
            df_final["sentimen"] = df_final["text"].apply(detect_sentiment)
            st.session_state.df = df_final
            st.balloons()
            st.success(f"✨ Analisis selesai! {len(df_final)} baris diproses.")

elif menu == "📊 Visualisasi":
    st.title("📊 Visualisasi Data Sentimen")
    df = st.session_state.df
    if df.empty:
        st.warning("⚠️ Data belum tersedia. Harap input data terlebih dahulu.")
    else:
        counts = df["sentimen"].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📉 Distribusi Sentimen")
            fig, ax = plt.subplots()
            counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#95a5a6'])
            st.pyplot(fig)
        with col2:
            st.subheader("🍕 Proporsi Sentimen")
            fig2, ax2 = plt.subplots()
            ax2.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=['#2ecc71', '#e74c3c', '#95a5a6'])
            st.pyplot(fig2)
                
        st.divider()
        st.subheader("☁️ WordCloud Kata Populer")
        all_text = " ".join(df["text"].astype(str))
        if all_text.strip():
            wc = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(all_text)
            fig3, ax3 = plt.subplots()
            ax3.imshow(wc, interpolation="bilinear")
            ax3.axis("off")
            st.pyplot(fig3)

elif menu == "🔍 Data & Filter":
    st.title("🔍 Telusuri Data Ulasan")
    df = st.session_state.df
    if df.empty:
        st.info("ℹ️ Belum ada data untuk ditampilkan.")
    else:
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            search = st.text_input("🔎 Cari kata kunci dalam teks:")
        with col_s2:
            filter_sentimen = st.multiselect("🏷️ Filter Kategori:", ["Positive", "Negative", "Netral"], default=["Positive", "Negative", "Netral"])
        
        filtered_df = df[df["sentimen"].isin(filter_sentimen)]
        if search:
            filtered_df = filtered_df[filtered_df["text"].str.contains(search, case=False, na=False)]
        
        st.dataframe(filtered_df, use_container_width=True)

elif menu == "🧠 Training Model":
    st.title("🧠 Machine Learning - Training")
    df = st.session_state.df
    
    if df.empty:
        st.warning("⚠️ Data belum tersedia untuk training.")
    else:
        st.info("Menu ini akan melatih model Linear Regression berdasarkan pelabelan otomatis kamus kata.")
        
        sentiment_map = {"Positive": 1, "Netral": 0, "Negative": -1}
        df_ml = df.copy()
        df_ml['score'] = df_ml['sentimen'].map(sentiment_map)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(df_ml['text'])
        y = df_ml['score']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if st.button("🏗️ Mulai Training Model", use_container_width=True):
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Simpan hasil ke session state
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            
            st.success("🎯 Model Berhasil Dilatih! Silahkan cek menu 'Evaluasi Model' untuk detail performa.")

        if "model" in st.session_state:
            st.divider()
            st.subheader("🧪 Uji Coba Prediksi")
            user_input = st.text_input("Masukkan kalimat baru:")
            if user_input:
                vec = st.session_state.vectorizer.transform([user_input])
                res = st.session_state.model.predict(vec)[0]
                st.write(f"Skor Prediksi: **{res:.2f}**")

elif menu == "📈 Evaluasi Model":
    st.title("📈 Evaluasi Kinerja Model")
    
    if "model" not in st.session_state:
        st.warning("⚠️ Model belum dilatih. Silahkan ke menu **🧠 Training Model** terlebih dahulu.")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Prediksi
        y_pred_cont = model.predict(X_test)
        
        # Konversi Linear (kontinu) ke Kategorikal (-1, 0, 1) untuk Confusion Matrix
        def categorize(val):
            if val > 0.3: return 1
            if val < -0.3: return -1
            return 0
        
        y_pred = [categorize(x) for x in y_pred_cont]
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred_cont)
        st.metric("📉 Mean Squared Error (MSE)", f"{mse:.4f}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🧩 Confusion Matrix")
            labels = [-1, 0, 1]
            label_names = ["Negative", "Netral", "Positive"]
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=label_names, yticklabels=label_names, ax=ax)
            plt.xlabel('Prediksi')
            plt.ylabel('Aktual')
            st.pyplot(fig)
            
        with col2:
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, 
                                            target_names=label_names, 
                                            output_dict=True, 
                                            zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format(precision=2))

      

elif menu == "💾 Unduh Data":
    st.title("💾 Unduh Hasil Analisis")
    df = st.session_state.df
    if df.empty:
        st.warning("⚠️ Data kosong.")
    else:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download CSV", data=csv, file_name="hasil_sentimen.csv", mime="text/csv")