import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ======== Konfigurasi Halaman ========
st.set_page_config(
    page_title="Streamlit Apps - UAS Data Mining",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======== Sidebar Navigasi ========
st.sidebar.title("Main Page")
menu = st.sidebar.radio(
    "📂 Pilih Halaman:",
    ("🏠 Home", "🧬 Classification", "🗺️ Clustering")
)

# ======== Load Model & Data ========
model_knn = joblib.load("model_knn.pkl")
scaler_knn = joblib.load("scaler_knn.pkl")
df_diabetes = pd.read_csv("diabetes.csv")

model_kmeans = joblib.load("model_kmeans.pkl")
scaler_kmeans = joblib.load("scaler_kmeans.pkl")
df_gerai = pd.read_csv("lokasi_gerai_kopi_clean.csv")

# ======== Halaman Utama (Home) ========
if menu == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>Streamlit Apps</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Collection of my apps deployed in Streamlit</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### Nama: Siti Ruhul Askya")
    st.write("### NIM: 22146017")

# ======== Halaman Klasifikasi (KNN) ========
elif menu == "🧬 Classification":
    st.subheader("Prediksi Diabetes Menggunakan K-Nearest Neighbors")
    st.markdown("Masukkan data medis pasien untuk memprediksi apakah pasien berisiko diabetes.")

    col1, col2 = st.columns(2)
    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20)
        Glucose = st.number_input("Glucose", 0.0, 200.0)
        BloodPressure = st.number_input("Blood Pressure", 0.0, 150.0)
        SkinThickness = st.number_input("Skin Thickness", 0.0, 100.0)
    with col2:
        Insulin = st.number_input("Insulin", 0.0, 900.0)
        BMI = st.number_input("BMI", 0.0, 70.0)
        DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
        Age = st.number_input("Age", 0, 120)

    if st.button("🔍 Prediksi"):
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DPF, Age]])
        scaled_input = scaler_knn.transform(input_data)
        pred = model_knn.predict(scaled_input)[0]
        label = "✅ Positif Diabetes" if pred == 1 else "❌ Negatif Diabetes"
        st.success(f"Hasil Prediksi: {label}")

    with st.expander("📊 Confusion Matrix dan Metrik Evaluasi"):
        from sklearn.metrics import classification_report, confusion_matrix
        X = df_diabetes.drop("Outcome", axis=1)
        y = df_diabetes["Outcome"]
        X_scaled = scaler_knn.transform(X)
        y_pred_all = model_knn.predict(X_scaled)
        matrix = confusion_matrix(y, y_pred_all)
        report = classification_report(y, y_pred_all, output_dict=True)

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negatif", "Positif"], yticklabels=["Negatif", "Positif"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        st.write("### Classification Report")
        st.json(report)

# ======== Halaman Clustering (KMeans) ========
elif menu == "🗺️ Clustering":
    st.subheader("Clustering Lokasi Gerai Kopi (KMeans)")
    st.markdown("Masukkan parameter lokasi baru untuk melihat termasuk cluster mana.")

    fitur = ["x", "y", "population_density", "traffic_flow", "competitor_count", "is_commercial"]
    input_cluster = []
    col1, col2 = st.columns(2)
    with col1:
        input_cluster.append(st.number_input("x (Koordinat X)", 0.0, 100.0))
        input_cluster.append(st.number_input("y (Koordinat Y)", 0.0, 100.0))
        input_cluster.append(st.number_input("Population Density", 0.0, 10000.0))
    with col2:
        input_cluster.append(st.number_input("Traffic Flow", 0.0, 5000.0))
        input_cluster.append(st.number_input("Competitor Count", 0, 20))
        input_cluster.append(st.selectbox("Is Commercial Area?", [0, 1]))

    if st.button("🧩 Lihat Cluster"):
        scaled_input = scaler_kmeans.transform([input_cluster])
        cluster_result = model_kmeans.predict(scaled_input)[0]
        st.success(f"Lokasi termasuk dalam **Cluster {cluster_result}**")

    with st.expander("📍 Visualisasi Cluster"):
        df_plot = df_gerai.copy()
        df_plot["Cluster"] = model_kmeans.predict(scaler_kmeans.transform(df_plot[fitur]))
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df_plot, x="x", y="y", hue="Cluster", palette="tab10", s=80)
        plt.title("Visualisasi Cluster Lokasi Gerai")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        st.pyplot(fig2)
