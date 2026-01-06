import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Konfigurasi Halaman
st.set_page_config(page_title="Student Performance Clustering", layout="wide")
st.title("📊 Analisis Pengelompokan Performa Siswa (K-Means)")

# Membaca Dataset
@st.cache_data
def load_data():
    # Pastikan file CSV ini ada di folder yang sama atau berikan path lengkapnya
    df = pd.read_csv('StudentPerformanceFactors.csv', sep=',', engine='python')
    return df

try:
    spf = load_data()
    
    # Sidebar
    st.sidebar.header("Opsi Tampilan")
    show_raw_data = st.sidebar.checkbox("Tampilkan Data Mentah")

    if show_raw_data:
        st.subheader("Data Mentah (Top 5)")
        st.write(spf.head())

    # --- PEMROSESAN DATA ---
    # Perbaikan Syntax Error di baris ini:
    kolom_pilihan = ['Hours_Studied', 'Exam_Score', 'Attendance', 'Sleep_Hours']
    x_train = spf[kolom_pilihan]
    
    # Feature Scaling
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    # --- CLUSTERING ---
    st.sidebar.subheader("Konfigurasi K-Means")
    k_value = st.sidebar.slider("Pilih Jumlah Cluster (K)", min_value=2, max_value=6, value=3)

    kmeans = KMeans(n_clusters=k_value, random_state=42)
    spf['Cluster'] = kmeans.fit_predict(x_train_scaled)
    
    # Memberikan nama cluster secara otomatis (Opsional, seperti di notebook Anda)
    cluster_names = {0: 'Kelompok A', 1: 'Kelompok B', 2: 'Kelompok C', 3: 'Kelompok D', 4: 'Kelompok E', 5: 'Kelompok F'}
    spf['Cluster_Name'] = spf['Cluster'].map(cluster_names)

    # --- VISUALISASI ---
    st.subheader(f"Hasil Klasterisasi (K={k_value})")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Hours Studied vs Exam Score**")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=spf, x='Hours_Studied', y='Exam_Score', hue='Cluster', palette='viridis', ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("**Attendance vs Exam Score**")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=spf, x='Attendance', y='Exam_Score', hue='Cluster', palette='magma', ax=ax2)
        st.pyplot(fig2)

    # Menampilkan Statistik Cluster
    st.subheader("Rata-rata Performa per Cluster")
    summary = spf.groupby('Cluster')[kolom_pilihan].mean()
    st.table(summary)

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")