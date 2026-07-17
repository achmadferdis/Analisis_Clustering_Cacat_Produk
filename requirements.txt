import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="Analisis Clustering Cacat Produk",
    page_icon="📊",
    layout="wide"
)

st.title(" Analisis Clustering Cacat Produk Industri Manufaktur")
st.info("""
### IDENTITAS MAHASISWA

- **Nama** : Achmad Ferdi Santosa
- **NIM** : E12.2024.01950
- **Mata Kuliah** : Project Kecerdasan Buatan
""")

st.markdown("""
Aplikasi ini digunakan untuk melakukan analisis clustering terhadap data
cacat produk industri manufaktur menggunakan metode **K-Means Clustering**.
""")

st.divider()

# ==========================
# MEMBACA DATA
# ==========================

try:
    df = pd.read_csv("defects_data.csv")

    st.subheader("📄 Dataset")

    st.write(f"Jumlah Data : {df.shape[0]}")
    st.write(f"Jumlah Kolom : {df.shape[1]}")

    df.index = range(1, len(df) + 1)
    st.dataframe(df)

except FileNotFoundError:
    st.error("File defects_data.csv belum ditemukan.")

    st.info("Silakan letakkan file defects_data.csv pada folder yang sama dengan app.py")

    st.stop()

    st.divider()

st.header("⚙️ Preprocessing Data")

df_pre = df.copy()

# Menghapus kolom yang tidak digunakan
df_pre = df_pre.drop(columns=["defect_id", "defect_date"])

# Encoding data kategorikal
encoder = LabelEncoder()

for kolom in df_pre.select_dtypes(include="object").columns:
    df_pre[kolom] = encoder.fit_transform(df_pre[kolom])

st.write("Data setelah preprocessing")

st.dataframe(df_pre.head())

st.divider()

st.header("📈 Menentukan Jumlah Cluster (Elbow Method)")

# Standarisasi Data
scaler = StandardScaler()
X = scaler.fit_transform(df_pre)

# Elbow Method
wcss = []

for i in range(1,11):
    km = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    km.fit(X)
    wcss.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8,5))

ax.plot(range(1,11), wcss, marker='o')
ax.set_xlabel("Jumlah Cluster (K)")
ax.set_ylabel("WCSS")
ax.set_title("Metode Elbow")

st.pyplot(fig)
st.info("""
 **Interpretasi Elbow Method**

Metode Elbow digunakan untuk menentukan jumlah cluster yang optimal. Titik siku (elbow) pada grafik menunjukkan jumlah cluster yang memberikan keseimbangan antara kualitas pengelompokan dan kompleksitas model. Nilai tersebut dapat dijadikan acuan dalam memilih jumlah cluster sebelum dilakukan proses K-Means Clustering.
""")

st.divider()

st.header("🎯 K-Means Clustering")

k = st.number_input(
    "Pilih Jumlah Cluster",
    min_value=2,
    max_value=10,
    value=3,
    step=1
)
model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

cluster = model.fit_predict(X)

hasil = df.copy()
hasil["Cluster"] = cluster

st.success(f"Jumlah Cluster : {k}")

st.dataframe(hasil.head(20))

st.divider()

st.header("📊 Visualisasi Cluster")

fig, ax = plt.subplots(figsize=(8,6))

scatter = ax.scatter(
    X[:,0],
    X[:,1],
    c=cluster,
    cmap="viridis"
)

ax.set_xlabel("Fitur 1")
ax.set_ylabel("Fitur 2")
ax.set_title("Visualisasi Hasil Clustering")

st.pyplot(fig)

st.divider()

st.header("📝 Interpretasi Hasil")

jumlah_cluster = hasil["Cluster"].value_counts().sort_index()

# ===========================
# RINGKASAN HASIL CLUSTERING
# ===========================

st.write(f"""
Berdasarkan hasil **K-Means Clustering** dengan jumlah cluster sebanyak **{k}**,
data cacat produk berhasil dikelompokkan menjadi **{k} kelompok** berdasarkan
kemiripan karakteristik.
""")

st.write("### Jumlah Data pada Setiap Cluster")

for cluster_id, jumlah in jumlah_cluster.items():
    st.write(f"✅ **Cluster {cluster_id + 1} : {jumlah} data**")

st.divider()

# ===========================
# MENENTUKAN PENJELASAN SESUAI JUMLAH CLUSTER
# ===========================

if k <= 3:

    tingkat = "lebih sederhana sehingga cocok untuk melihat gambaran umum pola cacat produk."

    interpretasi = """
Jumlah cluster yang sedikit menghasilkan pengelompokan yang lebih umum.
Setiap cluster masih mencakup cukup banyak data sehingga perusahaan dapat
lebih mudah memahami pola cacat utama sebelum melakukan analisis yang lebih rinci.
"""

    bisnis = [
        "Menentukan prioritas penanganan cacat utama.",
        "Memudahkan identifikasi kategori cacat terbesar.",
        "Sebagai analisis awal untuk evaluasi proses produksi.",
        "Membantu penyusunan strategi peningkatan kualitas secara umum."
    ]

elif k <= 6:r

    tingkat = "lebih rinci sehingga setiap cluster mulai menunjukkan karakteristik cacat yang lebih spesifik."

    interpretasi = """
Dengan jumlah cluster menengah, karakteristik setiap kelompok mulai terlihat
lebih jelas sehingga perusahaan dapat mengetahui variasi pola cacat yang
terjadi pada proses produksi.
"""

    bisnis = [
        "Mengidentifikasi karakteristik setiap kelompok cacat.",
        "Menentukan prioritas evaluasi pada proses produksi tertentu.",
        "Menganalisis penyebab cacat dengan lebih detail.",
        "Meningkatkan efektivitas pengendalian kualitas produk."
    ]

else:

    tingkat = "sangat rinci sehingga variasi karakteristik cacat dapat dianalisis lebih mendalam."

    interpretasi = """
Jumlah cluster yang lebih banyak menghasilkan segmentasi yang semakin detail.
Namun interpretasinya juga menjadi lebih kompleks sehingga perlu disesuaikan
dengan kebutuhan perusahaan.
"""

    bisnis = [
        "Menemukan pola cacat yang sangat spesifik.",
        "Membantu evaluasi mesin, operator, metode kerja, dan bahan baku.",
        "Menyusun strategi pengendalian kualitas berdasarkan setiap cluster.",
        "Mendukung pengambilan keputusan yang lebih akurat."
    ]

# ===========================
# PENJELASAN MODEL
# ===========================

st.subheader(" Penjelasan Model")

st.write(f"""
Metode **K-Means Clustering** merupakan algoritma **unsupervised learning**
yang digunakan untuk mengelompokkan data berdasarkan tingkat kemiripan
karakteristik. Algoritma bekerja dengan menentukan titik pusat cluster
(**centroid**), kemudian setiap data ditempatkan pada cluster dengan
jarak paling dekat terhadap centroid tersebut.

Pada analisis ini digunakan sebanyak **{k} cluster**, sehingga data cacat
produk dikelompokkan menjadi **{k} kelompok**.
Dengan jumlah cluster tersebut proses pengelompokan menjadi **{tingkat}**
""")

st.divider()

# ===========================
# WAWASAN BISNIS
# ===========================

st.subheader(" Wawasan Bisnis")

st.write(f"""
Hasil clustering menggunakan **{k} cluster** memberikan informasi yang
dapat dimanfaatkan perusahaan dalam meningkatkan kualitas produk.
Beberapa manfaat yang diperoleh yaitu:
""")

for item in bisnis:
    st.write(f"• {item}")
