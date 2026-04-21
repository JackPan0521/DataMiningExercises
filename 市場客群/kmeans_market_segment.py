import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -- 0. arguments ------------------------------------------------------------
parser = argparse.ArgumentParser(description="KMeans clustering on market customer dataset")
parser.add_argument("--k", type=int, default=None, help="Manually set number of clusters")
parser.add_argument("--k-min", type=int, default=2, help="Minimum k for auto search")
parser.add_argument("--k-max", type=int, default=12, help="Maximum k for auto search")
parser.add_argument(
    "--auto-k-mode",
    type=str,
    choices=["silhouette", "elbow", "capped-silhouette"],
    default="capped-silhouette",
    help="Auto K strategy when --k is not provided",
)
parser.add_argument(
    "--auto-k-cap",
    type=int,
    default=8,
    help="Upper bound used by capped-silhouette mode",
)
args = parser.parse_args()

if args.k_min < 2 or args.k_max < args.k_min:
    raise ValueError("Invalid k range: ensure k-min >= 2 and k-max >= k-min")

# -- 1. load data ------------------------------------------------------------
base_dir = Path(__file__).resolve().parent
csv_path = base_dir / "Mall_Customers.csv"

if not csv_path.exists():
    raise FileNotFoundError(f"Cannot find dataset: {csv_path}")

last_error = None
for enc in ["utf-8-sig", "cp950", "utf-8"]:
    try:
        df = pd.read_csv(csv_path, encoding=enc)
        print(f"Loaded {len(df)} rows with encoding: {enc}")
        break
    except Exception as exc:
        last_error = exc
else:
    raise RuntimeError(f"Failed to read CSV {csv_path}; last error: {last_error}")

print("\n--- Raw dtypes ---")
print(df.dtypes)
print(df.head(3).to_string(index=False))

# -- 2. preprocessing --------------------------------------------------------
numeric_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
category_cols = ["Genre"]

model_df = df[numeric_cols + category_cols].copy()
label_encoder = LabelEncoder()
for col in category_cols:
    model_df[col] = label_encoder.fit_transform(model_df[col].astype(str))

model_df = model_df.fillna(model_df.median(numeric_only=True))
scaler = StandardScaler()
X = scaler.fit_transform(model_df)

# -- 3. k search (elbow + silhouette) ---------------------------------------
k_min, k_max = args.k_min, args.k_max
inertias, silhouettes = [], []

for k in range(k_min, k_max + 1):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X, labels, sample_size=min(3000, len(X)), random_state=42))
    print(f"  k={k:2d}  inertia={km.inertia_:,.2f}  silhouette={silhouettes[-1]:.4f}")

k_values = np.array(list(range(k_min, k_max + 1)))
sil_array = np.array(silhouettes)
inertia_array = np.array(inertias)

sil_best_k = int(k_values[np.argmax(sil_array)])

# Elbow K by largest absolute 2nd difference of inertia.
if len(inertia_array) >= 3:
    second_diff = np.diff(inertia_array, n=2)
    elbow_index = int(np.argmax(np.abs(second_diff))) + 2
    elbow_k = int(k_values[elbow_index])
else:
    elbow_k = sil_best_k

cap_k = max(k_min, min(args.auto_k_cap, k_max))
cap_mask = k_values <= cap_k
if np.any(cap_mask):
    capped_best_k = int(k_values[cap_mask][np.argmax(sil_array[cap_mask])])
else:
    capped_best_k = sil_best_k

if args.auto_k_mode == "silhouette":
    auto_best_k = sil_best_k
elif args.auto_k_mode == "elbow":
    auto_best_k = elbow_k
else:
    auto_best_k = capped_best_k

if args.k is not None:
    chosen_k = args.k
    print(
        f"\nUser-selected K={chosen_k} "
        f"(auto best K={auto_best_k}, mode={args.auto_k_mode}, cap={cap_k})"
    )
else:
    chosen_k = auto_best_k
    print(
        f"\nAuto-selected K={chosen_k} "
        f"(mode={args.auto_k_mode}, silhouette_best={sil_best_k}, elbow={elbow_k}, cap_best={capped_best_k})"
    )

# -- 4. final clustering -----------------------------------------------------
final_km = KMeans(n_clusters=chosen_k, n_init=20, random_state=42)
df["Cluster"] = final_km.fit_predict(X)
print(f"\nCluster distribution:\n{df['Cluster'].value_counts().sort_index()}")

# -- 5. cluster summary ------------------------------------------------------
print("\n--- Cluster Profile (numeric means) ---")
profile = df.groupby("Cluster")[numeric_cols].mean().round(2)
print(profile.to_string())

print("\n--- Dominant Genre per cluster ---")
dominant_genre = df.groupby("Cluster")["Genre"].agg(lambda x: x.value_counts().idxmax())
print(dominant_genre.to_string())

# -- 6. save output data -----------------------------------------------------
out_csv = base_dir / "kmeans_market_results.csv"
df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved clustered data: {out_csv}")

# -- 7. visualizations -------------------------------------------------------
k_range = list(range(k_min, k_max + 1))
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 7-1 elbow
axes[0].plot(k_range, inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow Curve (Inertia)")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].grid(True, alpha=0.3)
axes[0].axvline(auto_best_k, color="red", linestyle="--", label=f"Auto best k={auto_best_k}")
axes[0].axvline(elbow_k, color="green", linestyle=":", label=f"Elbow k={elbow_k}")
if args.k is not None:
    axes[0].axvline(chosen_k, color="purple", linestyle="-.", label=f"Chosen k={chosen_k}")
axes[0].legend()

# 7-2 silhouette
axes[1].plot(k_range, silhouettes, marker="o", color="darkorange")
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Score")
axes[1].grid(True, alpha=0.3)
axes[1].axvline(auto_best_k, color="red", linestyle="--", label=f"Auto best k={auto_best_k}")
axes[1].axvline(sil_best_k, color="green", linestyle=":", label=f"Silhouette best k={sil_best_k}")
if args.k is not None:
    axes[1].axvline(chosen_k, color="purple", linestyle="-.", label=f"Chosen k={chosen_k}")
axes[1].legend()

# 7-3 PCA 2D
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X)
scatter = axes[2].scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=df["Cluster"],
    cmap="tab10",
    alpha=0.75,
    s=40,
)
axes[2].set_title(f"PCA 2D - k={chosen_k} clusters")
axes[2].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
axes[2].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
axes[2].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[2], label="Cluster")

plt.tight_layout()
out_png = base_dir / "kmeans_market_segment.png"
fig.savefig(out_png, dpi=160)
plt.close(fig)
print(f"Saved chart: {out_png}")
