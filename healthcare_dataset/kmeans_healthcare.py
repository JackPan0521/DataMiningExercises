import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── 0. 參數解析 ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="KMeans clustering on healthcare dataset")
parser.add_argument("--k", type=int, default=None, help="手動指定群數（不指定則自動選最佳）")
parser.add_argument("--k-min", type=int, default=2, help="自動搜尋最小 k（預設 2）")
parser.add_argument("--k-max", type=int, default=12, help="自動搜尋最大 k（預設 12）")
args = parser.parse_args()

# ── 1. 讀取資料 ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "healthcare_dataset.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"找不到資料檔: {CSV_PATH}")

last_error = None
for enc in ["utf-8-sig", "cp950", "utf-8"]:
    try:
        df = pd.read_csv(CSV_PATH, encoding=enc)
        print(f"Loaded {len(df)} rows with encoding: {enc}")
        break
    except Exception as exc:
        last_error = exc
else:
    raise RuntimeError(f"無法讀取資料檔: {CSV_PATH}; last error: {last_error}")

print("\n--- 原始欄位型態 ---")
print(df.dtypes)
print(df.head(3).to_string())

# ── 2. 特徵選擇與前處理 ───────────────────────────────────────────────────────
NUMERIC_COLS = ["Age", "Billing Amount"]
CATEGORY_COLS = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results",
]

# 計算住院天數
df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")
df["Stay Days"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days
NUMERIC_COLS.append("Stay Days")

# Label Encode 類別欄位
df_model = df[NUMERIC_COLS + CATEGORY_COLS].copy()
le = LabelEncoder()
for col in CATEGORY_COLS:
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# 填補缺值並標準化
df_model = df_model.fillna(df_model.median(numeric_only=True))
scaler = StandardScaler()
X = scaler.fit_transform(df_model)

# ── 3. 最佳 K 搜尋（Elbow + Silhouette） ────────────────────────────────────
K_MIN, K_MAX = args.k_min, args.k_max
inertias, silhouettes = [], []

for k in range(K_MIN, K_MAX + 1):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=42))
    print(f"  k={k:2d}  inertia={km.inertia_:,.1f}  silhouette={silhouettes[-1]:.4f}")

best_k_sil = K_MIN + int(np.argmax(silhouettes))

if args.k is not None:
    chosen_k = args.k
    print(f"\n使用者指定 K={chosen_k}（自動最佳 K 為 {best_k_sil}）")
else:
    chosen_k = best_k_sil
    print(f"\nBest K by silhouette score: {chosen_k}")

# ── 4. 用選定 K 訓練最終模型 ─────────────────────────────────────────────────
final_km = KMeans(n_clusters=chosen_k, n_init=20, random_state=42)
labels = final_km.fit_predict(X)
df["Cluster"] = labels
print(f"\nCluster distribution:\n{df['Cluster'].value_counts().sort_index()}")

# ── 5. 各群特徵摘要 ───────────────────────────────────────────────────────────
print("\n--- Cluster Profile (Mean of numeric features) ---")
profile = df.groupby("Cluster")[NUMERIC_COLS].mean().round(2)
print(profile.to_string())

print("\n--- Most common values per cluster ---")
for col in ["Medical Condition", "Admission Type", "Test Results", "Gender"]:
    most_common = df.groupby("Cluster")[col].agg(lambda x: x.value_counts().idxmax())
    print(f"  {col}: {most_common.to_dict()}")

# ── 6. 結果儲存 ───────────────────────────────────────────────────────────────
output_csv = BASE_DIR / "kmeans_results.csv"
df.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved clustered data: {output_csv}")

# ── 7. 視覺化 ─────────────────────────────────────────────────────────────────
k_range = list(range(K_MIN, K_MAX + 1))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 7-1 Elbow curve
axes[0].plot(k_range, inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Curve (Inertia)")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)
axes[0].axvline(best_k_sil, color="red", linestyle="--", label=f"Auto best k={best_k_sil}")
if args.k is not None:
    axes[0].axvline(chosen_k, color="purple", linestyle="-.", label=f"Chosen k={chosen_k}")
axes[0].legend()

# 7-2 Silhouette score
axes[1].plot(k_range, silhouettes, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Score")
axes[1].grid(True, alpha=0.3)
axes[1].axvline(best_k_sil, color="red", linestyle="--", label=f"Auto best k={best_k_sil}")
if args.k is not None:
    axes[1].axvline(chosen_k, color="purple", linestyle="-.", label=f"Chosen k={chosen_k}")
axes[1].legend()

# 7-3 PCA 2D scatter
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
scatter = axes[2].scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=df["Cluster"],
    cmap="tab10",
    alpha=0.5,
    s=10,
)
axes[2].set_title(f"PCA 2D — k={chosen_k} clusters")
axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
plt.colorbar(scatter, ax=axes[2], label="Cluster")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
output_png = BASE_DIR / "kmeans_healthcare.png"
fig.savefig(output_png, dpi=160)
plt.close(fig)
print(f"Saved chart: {output_png}")