# KosFinder Madiun

Aplikasi dashboard analisis dan prototipe prediksi harga kos di Kota Madiun.

## File
- `app.py`: aplikasi Streamlit
- `models/model_kosfinder.joblib`: model Ridge Regression yang dilatih dari sheet `Data Harga`
- Dataset dibaca langsung dari GitHub

## Menjalankan lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Gemini API
Atur secret `GEMINI_API_KEY` pada Streamlit Community Cloud:
```toml
GEMINI_API_KEY = "ISI_API_KEY_DI_DASHBOARD_SECRETS"
```
Jangan commit API key ke GitHub.

## Deployment
1. Upload isi folder proyek ini ke repository GitHub.
2. Buka https://share.streamlit.io/
3. Pilih repository, branch `main`, dan file `app.py`.
4. Pada Advanced settings / Secrets, masukkan `GEMINI_API_KEY`.
5. Deploy.

## Batasan
Dataset kecil; model hanya menggunakan `Jenis Kos` dan `Kecamatan` untuk memprediksi `Harga Bulanan (Rp)`. Hasil adalah prototipe pembelajaran dan bukan harga pasar yang terjamin.
