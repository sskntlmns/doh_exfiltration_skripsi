# ============================================================================
# LAMPIRAN 2 - POTONGAN SOURCE CODE PENELITIAN (REVISI R17)
# ============================================================================
# Deteksi Trafik Malicious DoH sebagai Proof of Concept
# Deteksi Eksfiltrasi Data Berbasis Metadata Flow
# ============================================================================


# ==========================================================================
# 1. SETUP DAN KONFIGURASI EKSPERIMEN
# ==========================================================================

# --- Import pustaka standar Python ---
import json
import time
import warnings
from pathlib import Path

# --- Import pustaka pihak ketiga untuk manipulasi data dan visualisasi ---
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Import komponen machine learning dari scikit-learn ---
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Menyembunyikan peringatan non-kritis agar output eksperimen tetap bersih
warnings.filterwarnings("ignore")

# Mengatur gaya visualisasi matplotlib agar konsisten di seluruh notebook
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("default")

# --- Konstanta global eksperimen untuk menjamin reprodusibilitas ---
RANDOM_STATE = 42   # Seed acak untuk seluruh proses stokastik
TEST_SIZE = 0.2      # Proporsi data uji (20%)

# Daftar jalur kandidat berkas dataset (mendukung eksekusi dari root atau subfolder)
DATA_CANDIDATES = [
    Path("../data/BCCC-CIRA-CIC-DoHBrw-2020.csv"),
    Path("data/BCCC-CIRA-CIC-DoHBrw-2020.csv"),
]

# Daftar jalur kandidat direktori output untuk menyimpan artefak eksperimen
OUTPUT_CANDIDATES = [
    Path("../outputs"),
    Path("outputs"),
]

# Resolusi otomatis jalur dataset dan pembuatan direktori output
DATA_PATH = next(
    (path for path in DATA_CANDIDATES if path.exists()),
    DATA_CANDIDATES[0],
)
OUTPUT_DIR = (
    OUTPUT_CANDIDATES[0] if DATA_CANDIDATES[0].exists()
    else OUTPUT_CANDIDATES[1]
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Konfirmasi jalur aktif yang digunakan eksperimen
print(f"Dataset path : {DATA_PATH.resolve()}")
print(f"Output dir   : {OUTPUT_DIR.resolve()}")


# ==========================================================================
# 2. PEMUATAN DATASET
# ==========================================================================

# Membaca dataset BCCC-CIRA-CIC-DoHBrw-2020 dari berkas CSV
df_raw = pd.read_csv(DATA_PATH)

# Membersihkan spasi berlebih pada nama kolom agar tidak menyebabkan KeyError
df_raw.columns = df_raw.columns.str.strip()

# Menampilkan dimensi awal dataset (jumlah baris x kolom)
print(f"Shape dataset: {df_raw.shape}")

# Menampilkan 5 baris pertama untuk verifikasi struktur data
display(df_raw.head())


# ==========================================================================
# 3. INSPEKSI AWAL DATASET
# ==========================================================================

# Membuat ringkasan per kolom: tipe data, jumlah & persentase missing, jumlah unik
inspection_summary = pd.DataFrame({
    "dtype": df_raw.dtypes.astype(str),
    "missing": df_raw.isna().sum(),
    "missing_pct": (df_raw.isna().mean() * 100).round(3),
    "unique": df_raw.nunique(dropna=True),
})

print("Ringkasan kolom:")
display(inspection_summary)

# Menghitung total baris duplikat identik dalam dataset
print(f"Duplikasi baris: {df_raw.duplicated().sum():,}")

# Menampilkan statistik deskriptif (mean, std, min, max, dll.) untuk seluruh kolom
display(df_raw.describe(include="all").T)


# ==========================================================================
# 4. DETEKSI OTOMATIS DAN ENCODING KOLOM LABEL
# ==========================================================================


def detect_label_column(dataframe: pd.DataFrame) -> str:
    """Mendeteksi kolom target/label secara otomatis.

    Strategi pencarian:
      1. Mencocokkan nama kolom dengan daftar kandidat eksplisit.
      2. Jika tidak ditemukan, mencari kolom kategorial dengan kardinalitas rendah.
    """
    # Daftar nama kolom label yang umum digunakan pada dataset benchmark
    explicit_candidates = [
        "Label", "label", "Class", "class",
        "Target", "target", "Category", "category",
    ]
    for candidate in explicit_candidates:
        if candidate in dataframe.columns:
            return candidate

    # Fallback: kolom bertipe objek/kategori dengan jumlah nilai unik <= 20
    object_columns = (
        dataframe.select_dtypes(include=["object", "category", "bool"])
        .columns.tolist()
    )
    low_cardinality = [
        col for col in object_columns
        if 1 < dataframe[col].nunique(dropna=True) <= 20
    ]
    if low_cardinality:
        return low_cardinality[-1]

    raise ValueError(
        "Kolom label tidak ditemukan. "
        "Tambahkan nama label ke explicit_candidates."
    )


# Menjalankan deteksi kolom label pada dataset mentah
LABEL_COL = detect_label_column(df_raw)

print(f"Kolom label terdeteksi: {LABEL_COL}")

# Menampilkan distribusi frekuensi setiap kategori pada kolom label
display(
    df_raw[LABEL_COL]
    .value_counts(dropna=False)
    .rename("count")
    .to_frame()
)


# ==========================================================================
# 5. PEMBERSIHAN DATA (CLEANING) DAN SELEKSI FITUR METADATA FLOW
# ==========================================================================

# Membuat salinan dataset agar data mentah asli tidak berubah
df = df_raw.copy()
initial_rows = len(df)

# Mengganti nilai tak-hingga (inf / -inf) menjadi NaN agar bisa ditangani imputer
df = df.replace([np.inf, -np.inf], np.nan)

# Menghapus baris yang kehilangan nilai pada kolom label target
df = df.dropna(subset=[LABEL_COL])

# Menghapus baris duplikat identik untuk mencegah data leakage dan bias
df = df.drop_duplicates().reset_index(drop=True)

# --- Seleksi fitur numerik (metadata flow) ---
# Mengambil seluruh kolom numerik kecuali kolom label
numeric_features = (
    df.drop(columns=[LABEL_COL])
    .select_dtypes(include=[np.number])
    .columns.tolist()
)

# Mengidentifikasi dan mengeliminasi fitur konstan (variansi nol)
# karena tidak memberikan daya diskriminatif pada model
constant_features = [
    col for col in numeric_features
    if df[col].nunique(dropna=True) <= 1
]
numeric_features = [
    col for col in numeric_features
    if col not in constant_features
]

# Membentuk matriks fitur independen (X)
X = df[numeric_features].copy()

# Membersihkan dan memformat string label target
y_raw = df[LABEL_COL].astype(str).str.strip()

# --- Encoding label untuk klasifikasi biner ---
label_encoder = LabelEncoder()
encoded_all = label_encoder.fit_transform(y_raw)
classes = label_encoder.classes_.tolist()

# Memastikan dataset memang bersifat biner (dua kelas)
if len(classes) != 2:
    raise ValueError(
        f"Pipeline ini disiapkan untuk klasifikasi biner. "
        f"Kelas ditemukan: {classes}"
    )

# Memetakan kelas positif (malicious/eksfiltrasi) berdasarkan kata kunci
positive_keywords = ["malicious", "attack", "exfil", "tunnel", "botnet"]
positive_class = next(
    (cls for cls in classes
     if any(kw in cls.lower() for kw in positive_keywords)),
    classes[1],
)
negative_class = next(cls for cls in classes if cls != positive_class)

# Mengonversi label menjadi integer biner (0 = Benign, 1 = Malicious)
y = (y_raw == positive_class).astype(int)

# --- Ringkasan hasil proses cleaning ---
cleaning_summary = {
    "initial_rows": int(initial_rows),
    "rows_after_cleaning": int(len(df)),
    "removed_rows": int(initial_rows - len(df)),
    "label_column": LABEL_COL,
    "negative_class_0": negative_class,
    "positive_class_1": positive_class,
    "n_numeric_features": int(len(numeric_features)),
    "constant_features_removed": constant_features,
}

print(json.dumps(cleaning_summary, indent=2))

# Menampilkan sampel fitur metadata yang telah dibersihkan
display(X.head())


# ==========================================================================
# 6. PEMBAGIAN DATA LATIH DAN DATA UJI (TRAIN-TEST SPLIT)
# ==========================================================================

# Melakukan stratified random split untuk menjaga proporsi kelas
# pada data latih dan data uji agar tetap representatif
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,        # 20% untuk data uji
    random_state=RANDOM_STATE,  # Seed reprodusibilitas
    stratify=y,                 # Stratifikasi berdasarkan label target
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")

# Menampilkan proporsi distribusi label pada data latih
print("Distribusi label train:")
display(
    y_train.value_counts(normalize=True)
    .rename({0: negative_class, 1: positive_class})
    .to_frame("ratio")
)

# Menampilkan proporsi distribusi label pada data uji
print("Distribusi label test:")
display(
    y_test.value_counts(normalize=True)
    .rename({0: negative_class, 1: positive_class})
    .to_frame("ratio")
)


# ==========================================================================
# 7. KONFIGURASI PIPELINE DAN PELATIHAN MODEL
# ==========================================================================
#
# Catatan arsitektur pipeline:
#   - SimpleImputer(strategy="median") digunakan pada SEMUA pipeline untuk
#     menangani kemungkinan missing values secara robust.
#   - StandardScaler() HANYA disertakan pada Logistic Regression karena
#     model linear sensitif terhadap perbedaan skala antar fitur.
#   - Random Forest dan AdaBoost TIDAK menyertakan StandardScaler karena
#     model berbasis pohon keputusan (tree-based models) bersifat invariant
#     terhadap transformasi skala monotonik (monotonic feature scaling)
#     sehingga tidak memerlukan normalisasi maupun standardisasi fitur.

models = {
    # --- Pipeline Logistic Regression (memerlukan StandardScaler) ---
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ]),

    # --- Pipeline Random Forest (tanpa StandardScaler) ---
    # Tree-based model: invariant terhadap skala fitur monotonik,
    # sehingga StandardScaler tidak diperlukan.
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ]),

    # --- Pipeline AdaBoost (tanpa StandardScaler) ---
    # Tree-based model: invariant terhadap skala fitur monotonik,
    # sehingga StandardScaler tidak diperlukan.
    "AdaBoost": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=RANDOM_STATE,
        )),
    ]),
}

# Kamus untuk menyimpan model terlatih dan durasi pelatihan
trained_models = {}
train_times = {}

# Iterasi pelatihan untuk setiap pipeline model
for model_name, model in models.items():
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    train_times[model_name] = elapsed
    trained_models[model_name] = model
    print(f"{model_name} selesai dilatih dalam {elapsed:.4f} detik")
