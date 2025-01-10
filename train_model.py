import pandas as pd #membaca dan memproses data dalam bentuk tabel
from sklearn.model_selection import train_test_split #membagi dataset menjadi data latih (train) dan data uji (test).
from sklearn.feature_extraction.text import TfidfVectorizer #Untuk mengubah teks menjadi representasi numerik menggunakan metode TF-IDF.
from sklearn.ensemble import RandomForestClassifier #Model pembelajaran mesin untuk klasifikasi berbasis Random Forest.
from sklearn.metrics import classification_report, accuracy_score #Untuk mengevaluasi performa model.
from joblib import dump #ntuk menyimpan model yang sudah dilatih.
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #Library NLP untuk melakukan stemming dalam bahasa Indonesia.
import numpy as np #Digunakan untuk manipulasi data numerik.

# Inisialisasi stemmer Sastrawi
#Library Sastrawi digunakan untuk stemming, yaitu mengubah kata menjadi bentuk dasarnya. Misalnya, kata makanan akan diubah menjadi makan.
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Fungsi preprocessing (Mengubah teks menjadi huruf kecil, melakukan stemming,Mengembalikan teks yang sudah diproses) 
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Ubah ke huruf kecil
        text = stemmer.stem(text)  # Stemming
        return text
    return ""

# Load dataset
dataset_path = "datasets/TestDataBersih.xlsx"  # Path ke file dataset
df = pd.read_excel(dataset_path, engine='openpyxl')

# Validasi kolom
#Menghapus spasi pada nama kolom agar tidak ada kesalahan akses kolom.
#Memeriksa apakah kolom 'TeksTweet' dan 'label' ada dalam dataset. Jika tidak, akan muncul error.
df.columns = df.columns.str.strip()
if 'TeksTweet' not in df.columns or 'label' not in df.columns:
    raise ValueError("Kolom 'TeksTweet' atau 'label' tidak ditemukan di dataset.")

# Hapus data dengan nilai kosong di TeksTweet
df = df.dropna(subset=['TeksTweet'])

# Pastikan hanya label 0 (Negatif) dan 1 (Positif)
df = df[df['label'].isin([0, 1])]

# Jika setelah filter dataset kosong, hentikan eksekusi
if df.empty:
    raise ValueError("Dataset kosong setelah pembersihan. Periksa ulang dataset!")

# Preprocessing teks
df['TeksTweet'] = df['TeksTweet'].apply(preprocess_text)

# Split dataset
#X adalah teks tweet, y adalah label sentimen.
#Dataset dibagi menjadi 80% data latih dan 20% data uji menggunakan train_test_split().
X = df['TeksTweet']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
#TfidfVectorizer mengubah teks menjadi angka agar bisa diproses oleh model machine learning.
#max_features=5000 berarti hanya menggunakan 5000 kata terpenting.
#fit_transform() digunakan untuk data latih, sedangkan transform() digunakan untuk data uji.
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train Random Forest Classifier
#RandomForestClassifier adalah model machine learning yang terdiri dari banyak pohon keputusan (decision trees).
#n_estimators=100 berarti model akan memiliki 100 pohon keputusan.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vectorized, y_train)

# Prediksi dan Evaluasi
#Menampilkan laporan klasifikasi yang berisi precision, recall, dan F1-score.
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi Model:", accuracy)
print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=["Negatif", "Positif"]))

# Simpan model dan vectorizer
dump(model, 'models/sentiment_model.pkl')
dump(vectorizer, 'models/vectorizer.pkl')

print("Model dan vectorizer berhasil disimpan!")
