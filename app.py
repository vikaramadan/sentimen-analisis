import streamlit as st #Digunakan untuk membuat tampilan aplikasi berbasis web.
from joblib import load #Untuk memuat model machine learning dan vectorizer yang sudah dilatih sebelumnya.
import re #Digunakan untuk membersihkan teks dari karakter yang tidak diperlukan.
import pandas as pd #Untuk mengolah data dalam bentuk tabel.
from io import BytesIO #Untuk menyimpan hasil analisis dalam format Excel dan memungkinkan pengguna mengunduhnya.

# Load Model dan Vectorizer
#Model machine learning (sentiment_model.pkl) yang telah dilatih sebelumnya dimuat kembali agar bisa digunakan untuk prediksi.
#Vectorizer (vectorizer.pkl) juga dimuat kembali agar teks yang dimasukkan bisa diubah menjadi bentuk numerik sebelum dianalisis oleh model.
model = load('models/sentiment_model.pkl')
vectorizer = load('models/vectorizer.pkl')

# Fungsi untuk membersihkan teks
def clean_text(text):
    text = re.sub(r'@\w+', '', text)  # Hapus mention
    text = re.sub(r'#', '', text)     # Hapus hashtag
    text = re.sub(r'http\S+|www.\S+', '', text)  # Hapus URL
    text = re.sub(r'[^\w\s,]', '', text)  # Hapus karakter non-alfanumerik
    text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi berlebih
    return text

# Fungsi untuk analisis sentimen
#Fungsi ini menerima daftar teks (texts) sebagai input.
#Teks tersebut diubah menjadi bentuk numerik menggunakan vectorizer.transform() (TF-IDF).
#Model kemudian melakukan prediksi sentimen (model.predict()).
#Mengembalikan hasil prediksi (0 = Negatif, 1 = Positif).
def analyze_sentiment(texts):
    features = vectorizer.transform(texts)
    predictions = model.predict(features)
    return predictions

# Streamlit UI
#Menampilkan judul dan subjudul di halaman web.
#Streamlit (st.title() dan st.subheader()) digunakan untuk membuat tampilan lebih menarik.
st.title("Analisis Sentimen Teks ")
st.subheader("Masukkan beberapa teks (pisahkan dengan baris baru/enter) untuk menganalisis sentimennya.")

# Input banyak kalimat
#st.text_area() memungkinkan pengguna memasukkan beberapa teks sekaligus.
#Teks dapat dipisahkan dengan baris baru (Enter).
input_text = st.text_area("")

if st.button("Analisis Sentimen"):
    try:
        if not input_text.strip():
            st.warning("Silakan masukkan teks untuk dianalisis.")
        else:
            # Pisahkan teks menjadi daftar berdasarkan baris
            text_list = input_text.strip().split("\n")
            
            # Bersihkan teks yang dimasukkan
            cleaned_texts = [clean_text(text) for text in text_list]
            
            # Analisis sentimen
            sentiments = analyze_sentiment(cleaned_texts)

            # Membuat tabel (DataFrame) yang berisi teks asli, teks setelah dibersihkan, dan hasil sentimen.
            df_result = pd.DataFrame({
                "Teks Asli": text_list,
                "Teks Setelah Dibersihkan": cleaned_texts,
                "Sentimen": ["Positif" if s == 1 else "Negatif" for s in sentiments]
            })

            # Hitung jumlah sentimen
            sentiment_counts = df_result["Sentimen"].value_counts().reset_index()
            sentiment_counts.columns = ["Kategori", "Jumlah"]

            # Tampilkan tabel hasil
            st.write("### 📊 Hasil Analisis Sentimen:")
            st.dataframe(df_result)

            # Tampilkan jumlah masing-masing sentimen
            st.write("### 📊 Jumlah Sentimen:")
            st.dataframe(sentiment_counts)

            # Simpan hasil ke Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_result.to_excel(writer, sheet_name="Hasil Sentimen", index=False)
                sentiment_counts.to_excel(writer, sheet_name="Ringkasan", index=False)

            output.seek(0)

            # Tombol unduh hasil dalam format Excel
            st.download_button(
                label="📥 Unduh Hasil dalam Excel",
                data=output,
                file_name="Hasil_Analisis_Sentimen.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
