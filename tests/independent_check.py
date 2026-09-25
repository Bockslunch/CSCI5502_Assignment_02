"""Independent spot-check of the profiler's numbers.

Re-computes a few statistics with a *different code path* (pandas' own CSV type
guessing + NumPy percentiles, no profiler code) and compares them with the
values stored in output/<dataset>/analysis_summary.json.

Usage (from the repository root):
    python tests/independent_check.py data/chicago_energy_benchmarking.csv output/dataset_a
"""

import json
import sys

import numpy as np
import pandas as pd


def main(csv_path, out_dir):
    df = pd.read_csv(csv_path, low_memory=False)  # pandas chooses dtypes itself here
    summary = json.load(open(f"{out_dir}/analysis_summary.json", encoding="utf-8"))
    ok = True

    def check(label, ours, theirs, tol=1e-6):
        nonlocal ok
        same = (ours == theirs) if isinstance(ours, str) else abs(float(ours) - float(theirs)) <= tol * max(1, abs(float(theirs)))
        ok &= same
        print(f"{'OK ' if same else 'XX '} {label}: independent={ours}  profiler={theirs}")

    ov = summary["overview"]
    check("rows", len(df), ov["rows"])
    check("columns", df.shape[1], ov["columns"])
    check("duplicate rows", int(df.duplicated().sum()), summary["data_quality"]["duplicate_rows"])

    for col, st in list(summary["numeric_statistics"].items())[:3]:
        v = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce").dropna().to_numpy(float)
        q1, med, q3 = np.percentile(v, [25, 50, 75])  # NumPy 'linear' = pandas default
        iqr = q3 - q1
        n_out = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum())
        check(f"{col} valid count", len(v), st["valid_count"])
        check(f"{col} mean", v.mean(), st["mean"])
        check(f"{col} median", med, st["median"])
        check(f"{col} std (ddof=1)", v.std(ddof=1), st["std"])
        check(f"{col} IQR", iqr, st["iqr"])
        check(f"{col} outliers (1.5xIQR)", n_out, st["outlier_count"])

    for col, st in list(summary["categorical_statistics"].items())[:2]:
        vc = df[col].dropna().astype(str).str.strip().value_counts()
        check(f"{col} distinct", vc.size, st["unique"])
        check(f"{col} top value", str(vc.index[0]), st["top_values"][0]["value"])
        check(f"{col} top count", int(vc.iloc[0]), st["top_values"][0]["count"])

    for p in summary["relationships"].get("top_pairs_by_abs_r", [])[:2]:
        a = pd.to_numeric(df[p["col_a"]], errors="coerce")
        b = pd.to_numeric(df[p["col_b"]], errors="coerce")
        m = a.notna() & b.notna()
        r = np.corrcoef(a[m], b[m])[0, 1]
        check(f"r({p['col_a']}, {p['col_b']})", round(r, 4), p["pearson_r"], tol=1e-3)

    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS DIFFER - investigate")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
