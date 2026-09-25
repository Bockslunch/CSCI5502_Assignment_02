"""Automated CSV Profiler and Verified Insight Generator (CSCI 5502, Assignment 2).

Architecture:  CSV file -> Python analysis -> verified summary (JSON)
               -> LLM explanation (optional, Ollama) -> Python verification -> generated report

Entry point (Python):
    from profiler import generate_profile
    generate_profile("data/my_file.csv", "output/my_dataset", use_llm=True)

Entry point (command line):
    python src/profiler.py data/my_file.csv --out output/my_dataset
    python src/profiler.py data/my_file.csv --out output/my_dataset --no-llm
"""

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow `python src/profiler.py`

from analysis import (categorical_stats, date_stats, group_comparison, numeric_stats,  # noqa: E402
                      pick_group_column, quality_checks, relationships)
from config import ProfilerConfig  # noqa: E402
from inference import infer_all  # noqa: E402
from llm import (assemble_insights, build_facts, build_prompt, call_ollama, parse_insights,  # noqa: E402
                 template_insights, verify_insights)
from plots import choose_and_draw  # noqa: E402
from report import write_report  # noqa: E402


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return str(o)


def _clean_nan(obj):
    """Replace float NaN/inf with None so the JSON file is valid."""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def load_csv(csv_path, cfg):
    """Read every column as text so type inference is ours, not pandas'. Nothing is dropped."""
    na_values = list(cfg.missing_codes) if cfg.missing_codes else None
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(csv_path, dtype=str, na_values=na_values, keep_default_na=True,
                               encoding=enc, low_memory=False), enc
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def generate_profile(csv_path, output_dir, use_llm=True, config=None):
    """Profile one CSV file and write the full report package to output_dir.

    Returns the analysis summary dictionary."""
    cfg = config or ProfilerConfig()
    csv_path, out = Path(csv_path), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots_dir = out / "plots"
    if plots_dir.exists():  # remove plots from an earlier run so the folder matches this report
        for old in plots_dir.glob("plot_*.png"):
            old.unlink()
    print(f"[1/6] Loading {csv_path} ...")
    df_raw, encoding = load_csv(csv_path, cfg)
    n_rows, n_cols = df_raw.shape
    print(f"      {n_rows:,} rows x {n_cols} columns (encoding {encoding})")

    print("[2/6] Inferring column types and roles ...")
    infos, typed = infer_all(df_raw, cfg)

    print("[3/6] Data quality checks and descriptive statistics ...")
    quality = quality_checks(df_raw, infos, cfg)
    num_stats, cat_stats, dt_stats = {}, {}, {}
    for i in infos:
        c, role = i["column"], i["role"]
        if role == "Numeric measure":
            num_stats[c] = numeric_stats(typed[c], n_rows, cfg)
        elif role in ("Categorical attribute", "Boolean field"):
            cat_stats[c] = {"role": role, **categorical_stats(df_raw[c], n_rows, cfg.top_k)}
        elif role == "Date-like field":
            dt_stats[c] = date_stats(typed[c], n_rows, is_year=i["technical_type"] != "datetime")

    rel, corr = relationships(typed, infos, cfg)
    gcol = pick_group_column(infos, typed)
    measures = [c for c in (rel.get("numeric_columns_used") or list(num_stats)) if typed[c].nunique() > 10] \
        or rel.get("numeric_columns_used") or []
    measures = sorted(measures, key=lambda c: -typed[c].notna().sum())
    gcomp = group_comparison(typed, gcol, measures[0]) if gcol and measures else None

    skipped = []
    if not num_stats:
        skipped.append("numeric statistics and outlier analysis (no numeric measure columns)")
    if not cat_stats:
        skipped.append("categorical frequency analysis (no categorical or Boolean columns)")
    if not dt_stats:
        skipped.append("time-based analysis (no date-like columns)")
    if rel.get("skipped"):
        skipped.append(rel["skipped"])
    if not gcomp:
        skipped.append("numeric-by-category comparison (needs a numeric measure and a categorical column with 2-15 levels)")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {k: v for k, v in vars(cfg).items()},
        "overview": {"filename": csv_path.name, "rows": n_rows, "columns": n_cols, "encoding": encoding,
                     "column_names": list(df_raw.columns),
                     "role_counts": dict(Counter(i["role"] for i in infos).most_common())},
        "columns": infos,
        "data_quality": quality,
        "numeric_statistics": num_stats,
        "categorical_statistics": cat_stats,
        "date_statistics": dt_stats,
        "relationships": rel,
        "group_comparison": gcomp,
        "skipped_analyses": skipped,
        "limitations": [
            "Types and roles are inferred from values and generic name hints; they can be wrong for unusual columns.",
            "Column meanings, units and valid ranges are undocumented in a bare CSV and are not interpreted.",
            "Outliers use the 1.5×IQR rule, which flags many points in skewed distributions; flagged values are not necessarily errors.",
            "Pearson correlation captures only linear association and is sensitive to outliers; no correlation implies causation.",
            "Sensitive-field detection is a heuristic; no warning does not mean the data are safe.",
            "Placeholder values (e.g. 'unknown', -999) are flagged but not converted unless listed in config.missing_codes.",
        ],
    }

    print("[4/6] Choosing and drawing visualizations ...")
    plots, plots_skipped = choose_and_draw(typed, infos, summary, corr, plots_dir, cfg)
    summary["plots"], summary["plots_skipped"] = plots, plots_skipped
    print(f"      {len(plots)} plots saved")

    print("[5/6] Building verified fact list and (optionally) calling the LLM ...")
    facts = build_facts(summary)
    summary["llm_facts"] = facts
    columns = list(df_raw.columns)
    prompt = build_prompt(facts, columns, cfg.min_insights, cfg.max_insights)
    (out / "llm_prompt.txt").write_text(prompt, encoding="utf-8")

    verification = []
    llm_info = {"model": cfg.llm_model, "host": cfg.ollama_host}
    if not use_llm:
        llm_info["status"] = "disabled"
        (out / "llm_response.txt").write_text("LLM step disabled (use_llm=False). No model was called.\n", encoding="utf-8")
    else:
        response, err = call_ollama(prompt, cfg)
        if err:
            llm_info.update(status="unavailable", error=err)
            print(f"      LLM unavailable: {err}")
            (out / "llm_response.txt").write_text(
                f"LLM call failed, so AI-generated narrative insights were skipped.\nReason: {err}\n", encoding="utf-8")
        else:
            (out / "llm_response.txt").write_text(response, encoding="utf-8")
            parsed = parse_insights(response)
            verification = verify_insights(parsed, facts, columns, summary)
            llm_info.update(status="used", returned_count=len(parsed),
                            verified_count=sum(v["status"].startswith("VERIFIED") for v in verification))
            print(f"      model returned {len(parsed)} insights; {llm_info['verified_count']} passed verification")

    # Final insight list: verified LLM insights (max 2 per category), plus Python templates for any
    # required category the model missed, so the report always has 5-8 insights covering every type.
    insights = assemble_insights(verification, template_insights(facts, cfg.min_insights), facts,
                                 cfg.min_insights, cfg.max_insights)
    summary["llm"] = llm_info
    summary["insights"] = insights
    summary["claim_verification"] = verification

    print("[6/6] Writing report, column profile and JSON summary ...")
    _write_column_profile(summary, out / "column_profile.csv")
    (out / "analysis_summary.json").write_text(
        json.dumps(_clean_nan(json.loads(json.dumps(summary, default=_json_default))), indent=2), encoding="utf-8")
    write_report(summary, insights, verification, out / "report.md")
    print(f"Done. Report: {out / 'report.md'}")
    return summary


def _write_column_profile(summary, path):
    rows = []
    for c in summary["columns"]:
        name = c["column"]
        row = {"column": name, "technical_type": c["technical_type"], "probable_role": c["role"],
               "non_missing": c["non_missing"], "missing": c["missing"], "missing_pct": c["missing_pct"],
               "unique": c["unique"], "unique_ratio": c["unique_ratio"],
               "non_conforming_values": c["non_conforming_count"], "notes": "; ".join(c["notes"])}
        s = summary["numeric_statistics"].get(name)
        if s and s.get("valid_count"):
            row.update({k: s.get(k) for k in ("min", "max", "mean", "median", "mode", "std", "q1", "q3", "iqr",
                                               "outlier_count", "outlier_pct", "zero_pct")})
        cs = summary["categorical_statistics"].get(name)
        if cs:
            row["most_frequent"] = " | ".join(cs["most_frequent"][:3])
            row["most_frequent_count"] = cs["most_frequent_count"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Automated CSV profiler with verified LLM insights")
    ap.add_argument("csv_path", help="path to the CSV file to profile")
    ap.add_argument("--out", help="output folder (default: output/<csv file name>)")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM narrative step")
    ap.add_argument("--model", help="Ollama model name (default: %(default)s)", default=ProfilerConfig.llm_model)
    ap.add_argument("--host", help="Ollama host URL", default=ProfilerConfig.ollama_host)
    ap.add_argument("--missing-threshold", type=float, default=ProfilerConfig.high_missing_threshold,
                    help="fraction above which a column is flagged as highly missing (default 0.30)")
    ap.add_argument("--missing-codes", nargs="*", default=[],
                    help="extra strings to treat as missing when loading, e.g. -999 unknown")
    ap.add_argument("--max-plots", type=int, default=ProfilerConfig.max_plots)
    args = ap.parse_args(argv)

    cfg = ProfilerConfig(high_missing_threshold=args.missing_threshold, llm_model=args.model,
                         ollama_host=args.host, missing_codes=args.missing_codes, max_plots=args.max_plots)
    out = args.out or str(Path(__file__).resolve().parent.parent / "output" / Path(args.csv_path).stem)
    generate_profile(args.csv_path, out, use_llm=not args.no_llm, config=cfg)


if __name__ == "__main__":
    main()
