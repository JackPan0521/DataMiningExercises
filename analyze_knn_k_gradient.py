import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_csv_with_fallback(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp950", "big5", "utf-8"]
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded file with encoding: {enc}")
            return df
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to read CSV. Last error: {last_error}")


def split_type(type_name: str) -> tuple[str, str]:
    parts = str(type_name).split("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], "一般"


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    if "總層數" not in df.columns:
        raise ValueError("CSV 必須包含欄位: 總層數")

    value_cols = [c for c in df.columns if c != "總層數"]
    long_df = df.melt(
        id_vars=["總層數"],
        value_vars=value_cols,
        var_name="型別",
        value_name="單價",
    )

    parsed = long_df["型別"].apply(split_type)
    long_df["構造"] = parsed.apply(lambda x: x[0])
    long_df["類別"] = parsed.apply(lambda x: x[1])
    long_df["單價"] = pd.to_numeric(long_df["單價"], errors="coerce")
    return long_df


def build_pipeline(n_neighbors: int) -> Pipeline:
    numeric_features = ["總層數"]
    categorical_features = ["構造", "類別"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def evaluate_k_values(train_df: pd.DataFrame, k_min: int, k_max: int, cv_splits: int) -> pd.DataFrame:
    x = train_df[["總層數", "構造", "類別"]]
    y = train_df["單價"]

    k_values = list(range(k_min, k_max + 1))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    rows = []
    for k in k_values:
        pipeline = build_pipeline(n_neighbors=k)
        neg_mse_scores = cross_val_score(
            pipeline,
            x,
            y,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=None,
        )
        mse = -neg_mse_scores
        rmse = np.sqrt(mse)
        rows.append(
            {
                "k": k,
                "cv_rmse_mean": rmse.mean(),
                "cv_rmse_std": rmse.std(ddof=1),
            }
        )

    result = pd.DataFrame(rows)
    result["gradient_1st"] = result["cv_rmse_mean"].diff()
    result["gradient_2nd"] = result["gradient_1st"].diff()
    return result


def choose_best_k(result: pd.DataFrame) -> tuple[int, int]:
    best_k_by_rmse = int(result.loc[result["cv_rmse_mean"].idxmin(), "k"])

    finite_2nd = result["gradient_2nd"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_2nd.empty:
        elbow_k = best_k_by_rmse
    else:
        elbow_index = finite_2nd.abs().idxmax()
        elbow_k = int(result.loc[elbow_index, "k"])

    return best_k_by_rmse, elbow_k


def plot_k_selection(result: pd.DataFrame, best_k: int, elbow_k: int, output_png: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(result["k"], result["cv_rmse_mean"], marker="o", label="CV RMSE")
    axes[0].fill_between(
        result["k"],
        result["cv_rmse_mean"] - result["cv_rmse_std"],
        result["cv_rmse_mean"] + result["cv_rmse_std"],
        alpha=0.2,
        label="±1 std",
    )
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"Best k={best_k}")
    axes[0].axvline(elbow_k, color="green", linestyle=":", label=f"Elbow k={elbow_k}")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("K selection by CV RMSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result["k"], result["gradient_1st"], marker="o", color="orange")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("1st Gradient")
    axes[1].set_title("First gradient of RMSE")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result["k"], result["gradient_2nd"], marker="o", color="purple")
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_ylabel("2nd Gradient")
    axes[2].set_xlabel("k")
    axes[2].set_title("Second gradient of RMSE")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def train_and_report(train_df: pd.DataFrame, best_k: int) -> float:
    x = train_df[["總層數", "構造", "類別"]]
    y = train_df["單價"]
    model = build_pipeline(best_k)
    model.fit(x, y)
    pred = model.predict(x)
    rmse_train = float(np.sqrt(mean_squared_error(y, pred)))
    return rmse_train


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze KNN K-selection gradient for housing unit prices")
    parser.add_argument(
        "--csv",
        type=str,
        default="臺北市房屋構造標準單價表-35層以下(112年7月起適用)-revised.csv",
        help="Path to input CSV",
    )
    parser.add_argument("--k-min", type=int, default=1, help="Min k")
    parser.add_argument("--k-max", type=int, default=20, help="Max k")
    parser.add_argument("--cv", type=int, default=5, help="KFold splits")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="knn_k_selection_results.csv",
        help="Output CSV for k selection result",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default="knn_k_selection_gradient.png",
        help="Output plot image",
    )
    args = parser.parse_args()

    if args.k_min < 1 or args.k_max < args.k_min:
        raise ValueError("Invalid k range. Ensure k-min >= 1 and k-max >= k-min")

    csv_path = Path(args.csv)
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)

    raw_df = load_csv_with_fallback(csv_path)
    long_df = to_long_format(raw_df)
    train_df = long_df.dropna(subset=["單價"]).copy()

    if len(train_df) < args.cv:
        raise ValueError("Not enough non-null rows for cross-validation")

    result = evaluate_k_values(train_df, args.k_min, args.k_max, args.cv)
    best_k, elbow_k = choose_best_k(result)
    train_rmse = train_and_report(train_df, best_k)

    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    plot_k_selection(result, best_k, elbow_k, out_png)

    print("\n=== K Selection Summary ===")
    print(f"Rows for training: {len(train_df)}")
    print(f"Best k by minimum CV RMSE: {best_k}")
    print(f"Elbow k by largest |2nd gradient|: {elbow_k}")
    print(f"Train RMSE with best k={best_k}: {train_rmse:.2f}")
    print(f"Saved result table: {out_csv}")
    print(f"Saved gradient chart: {out_png}")

    print("\nTop 10 k by CV RMSE:")
    show_cols = ["k", "cv_rmse_mean", "cv_rmse_std", "gradient_1st", "gradient_2nd"]
    print(result.sort_values("cv_rmse_mean").head(10)[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
