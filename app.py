import os
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from google import genai

# ==========================================================
# KOSFINDER MADIUN - Dashboard Analisis & Prediksi Harga Kos
# ==========================================================

st.set_page_config(
    page_title="KosFinder Madiun",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_URL = (
    "https://raw.githubusercontent.com/habibbahprima-prog/"
    "KosFinder-Madiun/main/dataset_kos_madiun_diperluas.xlsx"
)
MODEL_PATH = "model_kosfinder.joblib"

# -------------------- Tampilan --------------------
st.markdown("""
<style>
    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: rgba(245,247,251,0.92); }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    .hero {
        background: linear-gradient(115deg, #102a43 0%, #176b87 58%, #37a6a0 100%);
        padding: 30px 34px; border-radius: 22px; color: white;
        margin-bottom: 22px; box-shadow: 0 10px 28px rgba(16,42,67,.14);
    }
    .hero h1 { color: white; font-size: 2.25rem; margin: 0 0 8px 0; }
    .hero p { color: #e5f4f5; font-size: 1rem; margin: 0; }
    .section-card {
        background: white; border: 1px solid #e7edf3; border-radius: 16px;
        padding: 20px 22px; margin: 10px 0 18px 0;
        box-shadow: 0 4px 14px rgba(16,42,67,.045);
    }
    .ai-box {
        background: #eef8f7; border-left: 5px solid #168b83;
        padding: 16px 18px; border-radius: 10px; margin: 8px 0 22px 0;
    }
    .small-muted { color: #607080; font-size: .9rem; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e7edf3; padding: 16px;
        border-radius: 15px; box-shadow: 0 3px 12px rgba(16,42,67,.04);
    }
    div.stButton > button, div.stFormSubmitButton > button {
        background: #176b87; color: white; border: 0; border-radius: 10px;
        padding: .55rem 1.1rem; font-weight: 600;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: #12546b; color: white;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Data & model --------------------
@st.cache_data
def load_data():
    return pd.read_excel(DATA_URL, sheet_name="Data Harga")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def get_gemini_key():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        return ""

def explain_with_gemini(title, prompt):
    """Meminta Gemini menjelaskan ringkasan hasil hitung Python."""
    key = get_gemini_key()
    if not key:
        return (
            "Penjelasan AI belum aktif karena GEMINI_API_KEY belum diatur. "
            "Buka pengaturan aplikasi di Streamlit Community Cloud → Settings/Secrets, "
            "lalu tambahkan GEMINI_API_KEY = \"API_KEY_KAMU\"."
        )
    try:
        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=(
                "Anda adalah asisten analisis data untuk proyek mahasiswa TI semester 3. "
                "Gunakan bahasa Indonesia yang sederhana, jelas, dan tidak terlalu panjang. "
                "Jelaskan hanya berdasarkan angka yang diberikan. Jangan mengarang fakta, "
                "jangan menyatakan sebab-akibat, dan selalu sebutkan keterbatasan dataset kecil. "
                "Susun jawaban menjadi 2-3 paragraf pendek.\\n\\n" + prompt
            )
        )
        answer = getattr(response, "text", None)
        if answer and answer.strip():
            return answer.strip()
        return "Gemini berhasil dipanggil, tetapi tidak mengembalikan teks. Coba muat ulang aplikasi."
    except Exception as e:
        return (
            "Gemini belum berhasil membuat penjelasan. Periksa API key, kuota, dan akses model. "
            f"Detail teknis: {type(e).__name__}: {e}"
        )

def show_ai_explanation(title, prompt):
    st.markdown(f"**✨ Penjelasan AI — {title}**")
    with st.container(border=True):
        with st.spinner("Gemini sedang menganalisis ringkasan data..."):
            explanation = explain_with_gemini(title, prompt)
        st.markdown(explanation)

# -------------------- Header --------------------
st.markdown("""
<div class="hero">
    <h1>🏠 KosFinder Madiun</h1>
    <p>Dashboard analisis dan prediksi harga kos di Kota Madiun</p>
    <p style="margin-top:12px;font-size:.9rem;">Informasi data • Visualisasi • Prediksi Machine Learning • Insight AI</p>
</div>
""", unsafe_allow_html=True)

try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error(f"Data atau model gagal dimuat: {e}")
    st.info(
        "Pastikan file dataset tersedia di GitHub dan file "
        "`model_kosfinder.joblib` berada di halaman utama repository."
    )
    st.stop()

required = ["Nama Kos", "Jenis Kos", "Kecamatan", "Harga Bulanan (Rp)"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {missing}")
    st.stop()

df["Harga Bulanan (Rp)"] = pd.to_numeric(df["Harga Bulanan (Rp)"], errors="coerce")
data = df.dropna(subset=["Harga Bulanan (Rp)"]).copy()
data["Kecamatan"] = data["Kecamatan"].fillna("Tidak diketahui").astype(str)
data["Jenis Kos"] = data["Jenis Kos"].fillna("Tidak diketahui").astype(str)

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## 🏠 KosFinder")
    st.caption("Mini Project Sistem Cerdas")
    page = st.radio(
        "Navigasi",
        ["Ringkasan Dashboard", "Analisis Data", "Prediksi Harga", "Tentang Proyek"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("Sumber data: GitHub — KosFinder-Madiun")
    st.caption(f"Jumlah listing harga: {len(data)}")

# -------------------- Summary metrics --------------------
avg_price = data["Harga Bulanan (Rp)"].mean()
median_price = data["Harga Bulanan (Rp)"].median()
min_price = data["Harga Bulanan (Rp)"].min()
max_price = data["Harga Bulanan (Rp)"].max()

if page == "Ringkasan Dashboard":
    st.subheader("Ringkasan data kos")
    st.caption("Gambaran umum berdasarkan sheet Data Harga pada dataset.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah listing", f"{len(data)}")
    c2.metric("Rata-rata harga", f"Rp{avg_price:,.0f}")
    c3.metric("Median harga", f"Rp{median_price:,.0f}")
    c4.metric("Rentang harga", f"Rp{min_price:,.0f} – Rp{max_price:,.0f}")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("### Distribusi harga kos")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data["Harga Bulanan (Rp)"], bins=8, color="#238b8e", edgecolor="white")
        ax.set_xlabel("Harga bulanan (Rp)")
        ax.set_ylabel("Jumlah listing")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("### Komposisi jenis kos")
        jenis_counts = data["Jenis Kos"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.pie(
            jenis_counts.values, labels=jenis_counts.index, autopct="%1.0f%%",
            startangle=90, colors=["#176b87", "#37a6a0", "#f2b56b", "#8b9dc3", "#9bc1bc"]
        )
        ax2.axis("equal")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    prompt = (
        f"Ringkasan distribusi harga kos di Madiun: jumlah listing {len(data)}, "
        f"rata-rata Rp{avg_price:.0f}, median Rp{median_price:.0f}, "
        f"minimum Rp{min_price:.0f}, maksimum Rp{max_price:.0f}. "
        f"Jumlah listing menurut jenis kos: {jenis_counts.to_dict()}. "
        "Jelaskan sebaran harga dan komposisi jenis kos secara hati-hati."
    )
    show_ai_explanation("Ringkasan Dashboard", prompt)

    st.markdown("### Data kos terbaru")
    st.dataframe(
        data[["Nama Kos", "Jenis Kos", "Kecamatan", "Harga Bulanan (Rp)"]].head(10),
        use_container_width=True,
        hide_index=True
    )

elif page == "Analisis Data":
    st.subheader("Analisis data kos")
    st.caption("Grafik dan tabel dihitung langsung menggunakan Pandas.")
    by_area = (
        data.groupby("Kecamatan")["Harga Bulanan (Rp)"]
        .agg(["count", "mean", "median"])
        .sort_values("mean", ascending=False)
    )
    st.markdown("### Rata-rata harga berdasarkan kecamatan")
    st.bar_chart(by_area["mean"], color="#238b8e")
    st.dataframe(
        by_area.rename(columns={
            "count": "Jumlah listing", "mean": "Rata-rata harga (Rp)",
            "median": "Median harga (Rp)"
        }).style.format({
            "Rata-rata harga (Rp)": "Rp{:,.0f}",
            "Median harga (Rp)": "Rp{:,.0f}"
        }),
        use_container_width=True
    )
    prompt_area = (
        "Tabel ringkasan harga per kecamatan (jumlah listing, rata-rata, median): "
        f"{by_area.round(0).to_dict(orient='index')}. "
        "Bandingkan nilai yang terlihat, sebutkan kecamatan dengan rata-rata tertinggi "
        "dan terendah hanya jika datanya tersedia. Ingatkan bahwa jumlah listing tiap "
        "kecamatan tidak sama dan data sedikit."
    )
    show_ai_explanation("Harga per Kecamatan", prompt_area)

    st.markdown("### Rata-rata harga berdasarkan jenis kos")
    by_type = data.groupby("Jenis Kos")["Harga Bulanan (Rp)"].agg(["count", "mean", "median"])
    st.bar_chart(by_type["mean"], color="#e6a34a")
    st.dataframe(
        by_type.rename(columns={
            "count": "Jumlah listing", "mean": "Rata-rata harga (Rp)",
            "median": "Median harga (Rp)"
        }).style.format({
            "Rata-rata harga (Rp)": "Rp{:,.0f}",
            "Median harga (Rp)": "Rp{:,.0f}"
        }),
        use_container_width=True
    )
    prompt_type = (
        "Tabel ringkasan harga per jenis kos (jumlah listing, rata-rata, median): "
        f"{by_type.round(0).to_dict(orient='index')}. "
        "Jelaskan perbedaan nilai rata-rata yang terlihat tanpa menyimpulkan penyebab."
    )
    show_ai_explanation("Harga per Jenis Kos", prompt_type)

elif page == "Prediksi Harga":
    st.subheader("Prediksi harga kos")
    st.write(
        "Pilih jenis kos dan kecamatan. Model menggunakan dua fitur tersebut "
        "untuk memperkirakan harga bulanan."
    )
    jenis_opts = sorted(data["Jenis Kos"].dropna().unique().tolist())
    area_opts = sorted(data["Kecamatan"].dropna().unique().tolist())
    with st.container(border=True):
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                jenis = st.selectbox("Jenis kos", jenis_opts)
            with col2:
                area = st.selectbox("Kecamatan", area_opts)
            submitted = st.form_submit_button("✨ Prediksi harga", use_container_width=True)
    if submitted:
        input_df = pd.DataFrame([{"Jenis Kos": jenis, "Kecamatan": area}])
        estimate = float(model.predict(input_df)[0])
        st.success(f"Perkiraan harga: Rp{estimate:,.0f} per bulan")
        show_ai_explanation(
            "Hasil Prediksi",
            f"Model memprediksi harga Rp{estimate:.0f} per bulan untuk jenis kos '{jenis}' "
            f"di Kecamatan '{area}'. Fitur model hanya Jenis Kos dan Kecamatan. "
            "Jelaskan makna hasil dan tekankan bahwa prediksi bukan harga pasti, "
            "karena dataset terbatas dan faktor fasilitas/luas/kondisi tidak menjadi fitur model."
        )
    st.info(
        "Catatan: hasil adalah estimasi model pembelajaran, bukan harga pasar yang dijamin. "
        "Model saat ini hanya menggunakan jenis kos dan kecamatan."
    )

else:
    st.subheader("Tentang KosFinder Madiun")
    st.markdown("""
    KosFinder Madiun merupakan mini project Sistem Cerdas untuk membantu melihat
    gambaran harga kos di Kota Madiun melalui analisis data dan prediksi Machine Learning.

    **Sumber dan pengolahan data**
    - Dataset dibaca dari repository GitHub proyek.
    - Sheet yang digunakan untuk analisis harga adalah `Data Harga`.
    - Fitur model: `Jenis Kos` dan `Kecamatan`.
    - Target prediksi: `Harga Bulanan (Rp)`.

    **Teknologi**
    - Python dan Pandas untuk pengolahan data.
    - Matplotlib dan Streamlit untuk visualisasi dan antarmuka web.
    - Scikit-learn untuk model regresi.
    - Google Gemini API untuk membuat penjelasan naratif dari ringkasan statistik yang dihitung Python.
    """)
    st.warning(
        "Keterbatasan: jumlah data relatif kecil dan sebagian listing masih memerlukan "
        "verifikasi. Hasil analisis dan prediksi hanya untuk demonstrasi akademik."
    )

st.divider()
st.caption("KosFinder Madiun • Mini Project Sistem Cerdas • Dibuat dengan Python, Streamlit, dan Gemini API")
