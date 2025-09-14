# 1
# python split_dataset.py --input_csv sentiment.csv --output_dir datasets --random_state 42 --tolerance_pct 0.5
# -*- coding: utf-8 -*-
import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split


def read_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = {"text", "score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing column: {missing}")

    df = df.dropna(subset=["text", "score"]).reset_index(drop=True)
 

    df["text"] = df["text"].astype(str)
    try:
        df["score"] = df["score"].astype(int)
    except Exception as e:
        raise ValueError("Cannot convert to int") from e

    valid_set = {0, 1, 2}
    bad_mask = ~df["score"].isin(valid_set)
    if bad_mask.any():
        bad_vals = df.loc[bad_mask, "score"].unique().tolist()
        raise ValueError(f"bad score: {bad_vals}; just valid: {valid_set}.")

    df = df.reset_index(drop=True)
    df["row_id"] = df.index
    return df


def stratified_80_10_10(df: pd.DataFrame, random_state: int = 42):
    y = df["score"]
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=y, random_state=random_state, shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["score"],
        random_state=random_state,
        shuffle=True,
    )

    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, val_df, test_df


def class_distribution(df: pd.DataFrame):
    cnt = df["score"].value_counts().sort_index()
    total = len(df)
    pct = (cnt / total * 100.0).round(3)
    return cnt.to_dict(), pct.to_dict(), total


def pretty_distribution(name: str, df: pd.DataFrame):
    cnt, pct, total = class_distribution(df)
    rows = []
    for k in sorted(cnt.keys()):
        rows.append(f"  Class {k}: {cnt[k]:,} ({pct[k]:.3f}%)")
    return f"{name} → Total: {total:,}\n" + "\n".join(rows)


def verify_no_overlap(original_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    t_ids = set(train_df["row_id"].tolist())
    v_ids = set(val_df["row_id"].tolist())
    te_ids = set(test_df["row_id"].tolist())

    assert t_ids.isdisjoint(v_ids), "Overlap between TRAIN and VAL!"
    assert t_ids.isdisjoint(te_ids), "Overlap between TRAIN and TEST!"
    assert v_ids.isdisjoint(te_ids), "Overlap between VAL and TEST!"

    union = t_ids | v_ids | te_ids
    orig = set(original_df["row_id"].tolist())
    assert union == orig, "The union of the three parts is not equal to the rows of the original dataset!"
    return True


def verify_sizes(original_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    n_total = len(original_df)
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    assert n_total == (n_train + n_val + n_test), "The sum of the sizes does not equal the entire dataset!"
    return n_total, n_train, n_val, n_test


def verify_stratification(global_df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                          tolerance_pct: float = 0.5):
    _, g_pct, _ = class_distribution(global_df)
    _, t_pct, _ = class_distribution(train_df)
    _, v_pct, _ = class_distribution(val_df)
    _, te_pct, _ = class_distribution(test_df)

    problems = []
    for k in sorted(g_pct.keys()):
        gp = g_pct[k]
        for split_name, sp in [("train", t_pct[k]), ("val", v_pct[k]), ("test", te_pct[k])]:
            if abs(sp - gp) > tolerance_pct:
                problems.append(f"Class {k} in {split_name}: difference {abs(sp-gp):.3f}% > {tolerance_pct}%")
    return len(problems) == 0, problems


def ensure_each_class_present(df: pd.DataFrame, name: str):
    missing = sorted({0, 1, 2} - set(df["score"].unique()))
    if missing:
        raise AssertionError(f"{name} It lacks these classes: {missing}")


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: str = "."):
    os.makedirs(out_dir, exist_ok=True)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        to_save = part[["text", "score"]].copy()
        out_path = os.path.join(out_dir, f"{name}.csv")
        to_save.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved File: {out_path}  ({len(to_save):,} Row)")


def main():
    parser = argparse.ArgumentParser(description="Stratified 80/10/10 split for sentiment.csv")
    parser.add_argument("--input_csv", type=str, default="datasets/sentiment.csv")
    parser.add_argument("--output_dir", type=str, default="datasets/bilstm")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--tolerance_pct", type=float, default=0.5)
    args = parser.parse_args()

    print("==> Reading dataset ...")
    df_all = read_dataset(args.input_csv)

    print("==> Global distribution:")
    print(pretty_distribution("ALL", df_all))

    print("\n==> Stratified 80/10/10 split ...")
    train_df, val_df, test_df = stratified_80_10_10(df_all, random_state=args.random_state)

    print("\n==> Verifying sizes & overlaps ...")
    n_total, n_train, n_val, n_test = verify_sizes(df_all, train_df, val_df, test_df)
    verify_no_overlap(df_all, train_df, val_df, test_df)
    print(f"[OK] The sum of the sizes is correct: {n_train:,} + {n_val:,} + {n_test:,} = {n_total:,}")
    ensure_each_class_present(train_df, "TRAIN")
    ensure_each_class_present(val_df, "VAL")
    ensure_each_class_present(test_df, "TEST")
    print("[OK] Each section includes all three classes (0,1,2).")

    print("\n==> Split distributions:")
    print(pretty_distribution("TRAIN", train_df))
    print(pretty_distribution("VAL", val_df))
    print(pretty_distribution("TEST", test_df))

    print("\n==> Checking stratification tolerance ...")
    ok, problems = verify_stratification(df_all, train_df, val_df, test_df, tolerance_pct=args.tolerance_pct)
    if ok:
        print(f"[OK] The percentage of classes in each section is within ±{args.tolerance_pct}% of the overall distribution.")
    else:
        print("[WARN] Deviation exceeding the allowed limits in some sections:")
        for p in problems:
            print("   -", p)

    print("\n==> Saving CSVs ...")
    save_splits(train_df, val_df, test_df, out_dir=args.output_dir)

    cnt, _, _ = class_distribution(train_df)
    total = len(train_df)
    weights = {c: round(total / (3 * cnt[c]), 6) for c in sorted(cnt)}
    print("\nTip) Class weights (1/freq heuristic from TRAIN):", weights)
    print("==> Done.")


if __name__ == "__main__":
    main()
