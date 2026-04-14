import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
    long_df["單價"] = pd.to_numeric(long_df["單價"], errors="coerce")
    long_df = long_df.dropna(subset=["單價"]).copy()
    return long_df


def select_best_k_for_classification(x: pd.DataFrame, y: pd.Series, k_min: int, k_max: int, cv_splits: int) -> tuple[int, pd.DataFrame]:
    rows = []
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for k in range(k_min, k_max + 1):
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance")),
            ]
        )
        scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
        rows.append(
            {
                "k": k,
                "cv_accuracy_mean": scores.mean(),
                "cv_accuracy_std": scores.std(ddof=1),
            }
        )

    result = pd.DataFrame(rows)
    best_k = int(result.loc[result["cv_accuracy_mean"].idxmax(), "k"])
    return best_k, result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use KNN to classify housing type for new input")
    parser.add_argument(
        "--csv",
        type=str,
        default="臺北市房屋構造標準單價表-35層以下(112年7月起適用)-revised.csv",
        help="Path to input CSV",
    )
    parser.add_argument("--floor", type=float, default=None, help="新資料的總層數")
    parser.add_argument("--price", type=float, default=None, help="新資料的單價")
    parser.add_argument("--k-min", type=int, default=1, help="Min k for search")
    parser.add_argument("--k-max", type=int, default=20, help="Max k for search")
    parser.add_argument("--cv", type=int, default=5, help="KFold splits")
    parser.add_argument(
        "--report-csv",
        type=str,
        default="knn_classification_k_report.csv",
        help="Output CSV of k-search report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.k_min < 1 or args.k_max < args.k_min:
        raise ValueError("Invalid k range. Ensure k-min >= 1 and k-max >= k-min")

    csv_path = Path(args.csv)
    raw_df = load_csv_with_fallback(csv_path)
    long_df = to_long_format(raw_df)

    x = long_df[["總層數", "單價"]]
    y = long_df["型別"]

    if len(x) < args.cv:
        raise ValueError("資料筆數不足，無法進行交叉驗證")

    best_k, k_report = select_best_k_for_classification(x, y, args.k_min, args.k_max, args.cv)
    k_report.to_csv(args.report_csv, index=False, encoding="utf-8-sig")

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Some classes only appear once, so stratified split is not feasible.
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        print("Warning: stratified split not possible due to rare classes; fallback to random split.")

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=best_k, weights="distance")),
        ]
    )
    clf.fit(x_train, y_train)

    test_pred = clf.predict(x_test)
    test_acc = accuracy_score(y_test, test_pred)

    floor = args.floor
    price = args.price

    if floor is None:
        floor = float(input("請輸入總層數 (例如 12): "))
    if price is None:
        price = float(input("請輸入單價 (例如 15230): "))

    new_x = pd.DataFrame({"總層數": [floor], "單價": [price]})

    pred_label = clf.predict(new_x)[0]
    pred_prob = clf.predict_proba(new_x)[0]
    classes = clf.named_steps["knn"].classes_

    top_idx = np.argsort(pred_prob)[::-1][:3]

    print("\n=== KNN Classification Summary ===")
    print(f"Training rows: {len(x_train)}, Test rows: {len(x_test)}")
    print(f"Best k by CV accuracy: {best_k}")
    print(f"Holdout test accuracy: {test_acc:.4f}")
    print(f"Saved k-search report: {args.report_csv}")

    print("\n=== New Input Prediction ===")
    print(f"Input: 總層數={floor}, 單價={price}")
    print(f"Predicted 型別: {pred_label}")
    print("Top-3 probabilities:")
    for i in top_idx:
        print(f"  {classes[i]}: {pred_prob[i]:.4f}")


if __name__ == "__main__":
    main()
