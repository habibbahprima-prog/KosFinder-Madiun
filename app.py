import os
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from google import genai

st.set_page_config(page_title="KosFinder Madiun", page_icon="🏠", layout="wide")

DATA_URL = "https://raw.githubusercontent.com/habibbahprima-prog/KosFinder-Madiun/main/dataset_kos_madiun_diperluas.xlsx"
MODEL_PATH = "model_kosfinder.joblib"

@st.cache_data
def load_data():
    return pd.read_excel(DATA_URL, sheet_name="Data Harga")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def gemini_explain(prompt):
    key = os.getenv("GEMINI_API_KEY", "")
    try:
        if not key:
            key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    if not key:
        return "Penjelasan AI belum aktif. Tambahkan GEMINI_API_KEY pada Streamlit Secrets."
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text or "Gemini tidak mengembalikan penjelasan."
    except Exception as e:
        return f"Gemini belum dapat digunakan. Periksa API key/kuota. Detail: {e}"

st.title("🏠 KosFinder Madiun")
st.caption("Dashboard analisis dan prediksi harga kos di Kota Madiun")
st.markdown("Temukan gambaran harga kos berdasarkan jenis kos dan kecamatan.")

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Data atau model gagal dimuat: {e}")
    st.info("Pastikan dataset tersedia di GitHub dan file model ada pada folder models.")
    st.stop()

df["Harga Bulanan (Rp)"] = pd.to_numeric(df["Harga Bulanan (Rp)"], errors="coerce")
data = df.dropna(subset=["Harga Bulanan (Rp)"]).copy()

a, b, c = st.columns(3)
a.metric("Jumlah data harga", f"{len(data)}")
b.metric("Rata-rata harga", f"Rp{data['Harga Bulanan (Rp)'].mean():,.0f}")
c.metric("Median harga", f"Rp{data['Harga Bulanan (Rp)'].median():,.0f}")

tab1, tab2, tab3 = st.tabs(["Dashboard", "Prediksi Harga", "Tentang Data"])

with tab1:
    st.subheader("Data kos")
    show_cols = [c for c in ["Nama Kos", "Jenis Kos", "Kecamatan", "Harga Bulanan (Rp)", "Periode"] if c in data.columns]
    st.dataframe(data[show_cols], use_container_width=True)
    st.subheader("Distribusi harga kos")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(data["Harga Bulanan (Rp)"], bins=8, edgecolor="black")
    ax.set_xlabel("Harga bulanan (Rp)")
    ax.set_ylabel("Jumlah listing")
    ax.set_title("Distribusi Harga Kos")
    st.pyplot(fig)
    prompt = (
        "Jelaskan grafik distribusi harga kos berikut dalam bahasa Indonesia sederhana. "
        "Jangan mengarang fakta, jangan menyimpulkan sebab-akibat. "
        f"Jumlah listing={len(data)}, rata-rata=Rp{data['Harga Bulanan (Rp)'].mean():.0f}, "
        f"median=Rp{data['Harga Bulanan (Rp)'].median():.0f}, "
        f"minimum=Rp{data['Harga Bulanan (Rp)'].min():.0f}, "
        f"maksimum=Rp{data['Harga Bulanan (Rp)'].max():.0f}. "
        "Sebutkan keterbatasan dataset kecil."
    )
    with st.expander("✨ Penjelasan grafik oleh Gemini", expanded=True):
        st.write(gemini_explain(prompt))

    st.subheader("Rata-rata harga berdasarkan kecamatan")
    by_area = data.groupby("Kecamatan")["Harga Bulanan (Rp)"].mean().sort_values(ascending=False)
    st.bar_chart(by_area)
    prompt_area = (
        "Jelaskan tabel rata-rata harga kos per kecamatan berikut secara sederhana. "
        "Gunakan angka yang tersedia saja, jangan mengarang penyebab atau sebab-akibat. "
        f"Data: {by_area.round(0).to_dict()}. "
        "Sebutkan bahwa jumlah data per kecamatan mungkin berbeda."
    )
    with st.expander("✨ Penjelasan tabel kecamatan"):
        st.write(gemini_explain(prompt_area))

with tab2:
    st.subheader("Prediksi harga bulanan")
    st.write("Pilih jenis kos dan kecamatan sesuai kategori yang tersedia pada dataset.")
    jenis_opts = sorted(data["Jenis Kos"].dropna().astype(str).unique())
    area_opts = sorted(data["Kecamatan"].dropna().astype(str).unique())
    with st.form("prediction_form"):
        jenis = st.selectbox("Jenis kos", jenis_opts)
        area = st.selectbox("Kecamatan", area_opts)
        submitted = st.form_submit_button("Prediksi harga")
    if submitted:
        input_df = pd.DataFrame([{"Jenis Kos": jenis, "Kecamatan": area}])
        estimate = float(model.predict(input_df)[0])
        st.success(f"Perkiraan harga: Rp{estimate:,.0f} per bulan")
        st.write(gemini_explain(
            "Jelaskan hasil prediksi prototipe KosFinder Madiun. "
            f"Input jenis kos={jenis}; kecamatan={area}; hasil model=Rp{estimate:,.0f}/bulan. "
            "Jelaskan bahwa ini estimasi model dari dataset terbatas, bukan harga pasti."
        ))

with tab3:
    st.subheader("Informasi dataset")
    st.write("Dataset diambil dari repository GitHub proyek KosFinder-Madiun.")
    st.write("Sheet yang digunakan untuk model: **Data Harga**.")
    st.write("Fitur model: **Jenis Kos** dan **Kecamatan**.")
    st.write("Target model: **Harga Bulanan (Rp)**.")
    st.warning(
        "Dataset masih berukuran kecil dan catatan verifikasi menunjukkan sebagian listing "
        "perlu dikonfirmasi. Prediksi hanya untuk demonstrasi akademik, bukan harga pasar yang dijamin."
    )

st.divider()
st.caption("Mini Project Sistem Cerdas | KosFinder Madiun")
