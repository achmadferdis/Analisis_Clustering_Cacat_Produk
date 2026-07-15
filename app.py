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

st.title("📊 Analisis Clustering Cacat Produk Industri Manufaktur")
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

st.write(f"""
Berdasarkan hasil K-Means Clustering dengan jumlah cluster sebanyak **{k}**,
data cacat produk berhasil dikelompokkan menjadi **{k} kelompok** yang memiliki
karakteristik berbeda.

Jumlah data pada setiap cluster adalah sebagai berikut:
""")

for cluster_id, jumlah in jumlah_cluster.items():
    st.write(f"✅ Cluster {cluster_id + 1} : **{jumlah} data**")

st.write("")

if k == 2:
    st.info("""
Interpretasi:

• Data terbagi menjadi 2 kelompok besar.

• Pembagian ini cocok untuk membedakan karakteristik cacat produk secara umum,
misalnya kelompok cacat rendah dan kelompok cacat tinggi.

• Cocok digunakan apabila perusahaan hanya ingin mengetahui dua kategori utama.
""")

elif k == 3:
    st.info("""
Interpretasi:

• Data terbagi menjadi 3 kelompok.

• Pengelompokan menjadi tiga cluster biasanya menghasilkan pemisahan yang lebih
jelas dibanding dua cluster.

• Misalnya dapat merepresentasikan kelompok cacat rendah, sedang, dan tinggi.
""")

elif k <= 5:
    st.info(f"""
Interpretasi:

• Data berhasil dibagi menjadi {k} kelompok.

• Jumlah cluster yang lebih banyak membuat karakteristik setiap kelompok menjadi
lebih spesifik.

• Hal ini membantu perusahaan menentukan prioritas penanganan berdasarkan jenis
cacat yang lebih rinci.
""")

else:
    st.info(f"""
Interpretasi:

• Data berhasil dibagi menjadi {k} cluster.

• Semakin banyak cluster menghasilkan segmentasi yang semakin detail.

• Namun jumlah cluster yang terlalu banyak juga dapat membuat interpretasi menjadi
lebih kompleks sehingga perlu dipilih sesuai kebutuhan analisis.
""")