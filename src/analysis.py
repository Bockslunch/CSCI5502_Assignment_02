"""Deterministic analysis: data quality, descriptive statistics, relationships.

Nothing here modifies the data. Every check *flags* possible problems; rows,
missing values and outliers are never deleted or imputed.
"""

# IMPORTS: `re` FOR TEXT PATTERNS, `numpy`/`pandas` FOR THE MATH, AND `name_tokens` TO SPLIT COLUMN NAMES INTO WORDS.
import re

import numpy as np
import pandas as pd

from inference import name_tokens

# ------------------------------------------------------------------------------------
# SENSITIVE-FIELD HEURISTICS (WARNINGS ONLY - ABSENCE OF A WARNING IS NOT PROOF OF SAFETY)
# ------------------------------------------------------------------------------------
# COLUMN-NAME WORDS THAT MAY INDICATE PERSONAL OR SENSITIVE DATA, AND THE REASON SHOWN IN THE REPORT FOR EACH ONE.
SENSITIVE_NAME_HINTS = {
    "name": "person or entity name", "firstname": "person name", "lastname": "person name",
    "email": "email address", "phone": "phone number", "ssn": "government ID",
    "social": "possible social security number", "passport": "government ID",
    "address": "street address", "street": "street address", "dob": "date of birth",
    "birth": "date of birth", "birthdate": "date of birth", "age": "age",
    "gender": "gender", "sex": "sex", "race": "race", "ethnicity": "ethnicity",
    "religion": "religion", "diagnosis": "health information", "medical": "health information",
    "patient": "health information", "salary": "income", "income": "income",
    "latitude": "precise location", "longitude": "precise location", "lat": "precise location",
    "lon": "precise location", "lng": "precise location", "location": "location",
    "zip": "postal code", "zipcode": "postal code", "postal": "postal code",
    "ip": "IP address", "account": "account identifier", "license": "license number",
    "vin": "vehicle identifier", "password": "credential",
}
# VALUE PATTERNS THAT LOOK LIKE SENSITIVE DATA (EMAILS, PHONE NUMBERS, SSN-LIKE NUMBERS, IP ADDRESSES, CARD-LIKE
# NUMBERS, MAP POINTS). A COLUMN IS FLAGGED IF AT LEAST HALF OF ITS SAMPLED VALUES MATCH ONE OF THEM.
VALUE_PATTERNS = {
    "email address": re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$"),
    "US phone number": re.compile(r"^(\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}$"),
    "SSN-like number": re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    "IPv4 address": re.compile(r"^(\d{1,3}\.){3}\d{1,3}$"),
    "card-like number": re.compile(r"^(\d{4}[ -]?){3}\d{4}$"),
    "geographic point": re.compile(r"^POINT\s*\(", re.I),
}
# TEXT VALUES THAT OFTEN MEAN "MISSING" EVEN THOUGH THE CELL IS NOT EMPTY. THEY ARE COUNTED AND FLAGGED, NEVER CONVERTED.
PLACEHOLDER_TOKENS = {"-999", "-99", "-9999", "9999", "99999", "unknown", "unk", "not reported",
                      "not available", "none", "n/a", "na", "null", "?", "-", "--", "missing", "tbd"}


def _pct(part, whole):
    # SMALL HELPER: PART AS A PERCENTAGE OF WHOLE, ROUNDED TO 2 DECIMALS (RETURNS 0 INSTEAD OF DIVIDING BY ZERO).
    return round(100.0 * part / whole, 2) if whole else 0.0


def sensitive_checks(df_raw, infos, sample_size=2000):
    # CHECK EVERY COLUMN FOR SENSITIVE-LOOKING NAMES AND VALUES.
    flags = []
    for info in infos:
        col = info["column"]
        # NAME CHECK: A HINT MATCHES IF IT IS ONE OF THE NAME'S WORDS (OR THE WHOLE NAME, E.G. "FIRSTNAME" -> "FIRSTNAME").
        toks = name_tokens(col)
        joined = "".join(toks)
        reasons = sorted({v for k, v in SENSITIVE_NAME_HINTS.items() if k in toks or joined == k})
        # VALUE CHECK: TEST A RANDOM SAMPLE OF UP TO 2,000 VALUES AGAINST EACH SENSITIVE PATTERN.
        vals = df_raw[col].dropna().astype(str).str.strip()
        if len(vals):
            vals = vals.sample(min(len(vals), sample_size), random_state=0)
            for label, rx in VALUE_PATTERNS.items():
                share = vals.str.match(rx).mean()
                if share >= 0.5:
                    reasons.append(f"values look like {label} ({share:.0%} of sampled values)")
        # ONLY COLUMNS WITH AT LEAST ONE REASON ARE REPORTED.
        if reasons:
            flags.append({"column": col, "reasons": reasons})
    return flags


def quality_checks(df_raw, infos, cfg):
    # DUPLICATE ROWS: A ROW IS A DUPLICATE IF EVERY COLUMN MATCHES AN EARLIER ROW. THE FIRST COPY IS NOT COUNTED.
    n_rows = len(df_raw)
    dup_mask = df_raw.duplicated(keep="first")
    n_dup = int(dup_mask.sum())

    # CONSTANT COLUMNS (ONLY ONE DISTINCT VALUE) AND COMPLETELY EMPTY COLUMNS.
    constant = [i["column"] for i in infos if i["unique"] == 1]
    empty = [i["column"] for i in infos if i["non_missing"] == 0]
    # COLUMNS WHOSE MISSING SHARE IS ABOVE THE THRESHOLD (DEFAULT 30%), SORTED FROM MOST TO LEAST MISSING.
    high_missing = sorted(
        [{"column": i["column"], "missing": i["missing"], "missing_pct": i["missing_pct"]}
         for i in infos if i["missing_pct"] > 100 * cfg.high_missing_threshold],
        key=lambda d: -d["missing_pct"])
    # MIXED / INCONSISTENT TYPES: COLUMNS THAT ARE MOSTLY ONE TYPE BUT CONTAIN SOME VALUES THAT DO NOT FIT (WITH EXAMPLES).
    mixed = [{"column": i["column"], "technical_type": i["technical_type"],
              "non_conforming_count": i["non_conforming_count"],
              "non_conforming_pct": _pct(i["non_conforming_count"], i["non_missing"]),
              "examples": i["non_conforming_examples"]}
             for i in infos if i["technical_type"] == "mixed" or i["non_conforming_count"] > 0]
    # IDENTIFIER-LIKE COLUMNS AND CATEGORICAL COLUMNS WITH MORE THAN 50 DISTINCT VALUES (HIGH CARDINALITY).
    high_card = [{"column": i["column"], "unique": i["unique"], "unique_ratio": i["unique_ratio"],
                  "role": i["role"]}
                 for i in infos
                 if i["role"] == "Identifier-like field"
                 or (i["role"] == "Categorical attribute" and i["unique"] > cfg.high_cardinality_count)]

    # PLACEHOLDER SEARCH: FOR EACH NON-EMPTY COLUMN, COUNT VALUES (CASE-INSENSITIVE) THAT ARE IN THE PLACEHOLDER LIST,
    # AND KEEP A COUNT PER PLACEHOLDER VALUE (E.G. "NOT AVAILABLE": 6).
    placeholders = []
    for i in infos:
        if i["non_missing"] == 0:
            continue
        vals = df_raw[i["column"]].dropna().astype(str).str.strip().str.lower()
        hits = vals[vals.isin(PLACEHOLDER_TOKENS)]
        if len(hits):
            vc = hits.value_counts()
            placeholders.append({"column": i["column"], "count": int(len(hits)),
                                 "pct": _pct(len(hits), n_rows),
                                 "values": {str(k): int(v) for k, v in vc.items()}})

    # OVERALL MISSINGNESS: TOTAL MISSING CELLS ACROSS THE WHOLE TABLE, THEN BUNDLE EVERY CHECK INTO ONE DICTIONARY
    # THAT BECOMES THE "DATA_QUALITY" SECTION OF `analysis_summary.json`.
    total_cells = df_raw.size
    total_missing = int(sum(i["missing"] for i in infos))
    return {
        "duplicate_rows": n_dup,
        "duplicate_rows_pct": _pct(n_dup, n_rows),
        "total_missing_cells": total_missing,
        "total_missing_cells_pct": _pct(total_missing, total_cells),
        "rows_with_any_missing": int(df_raw.isna().any(axis=1).sum()),
        "rows_with_any_missing_pct": _pct(int(df_raw.isna().any(axis=1).sum()), n_rows),
        "high_missing_threshold_pct": 100 * cfg.high_missing_threshold,
        "high_missing_columns": high_missing,
        "constant_columns": constant,
        "empty_columns": empty,
        "mixed_type_columns": mixed,
        "identifier_or_high_cardinality_columns": high_card,
        "possible_placeholder_values": placeholders,
        "sensitive_field_warnings": sensitive_checks(df_raw, infos),
    }


# ------------------------------------------------------------------------------------
# DESCRIPTIVE STATISTICS
# ------------------------------------------------------------------------------------
def numeric_stats(s: pd.Series, n_rows: int, cfg) -> dict:
    # KEEP ONLY THE VALUES THAT ARE REAL NUMBERS; COUNT HOW MANY ARE VALID VS. MISSING.
    v = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    n = int(len(v))
    out = {"valid_count": n, "missing_count": int(n_rows - n), "missing_pct": _pct(n_rows - n, n_rows)}
    # A COLUMN WITH NO VALID NUMBERS ONLY GETS ITS COUNTS.
    if n == 0:
        return out
    # QUARTILES: Q1 = 25TH PERCENTILE, MEDIAN = 50TH, Q3 = 75TH. PANDAS USES LINEAR INTERPOLATION BETWEEN DATA POINTS.
    q1, med, q3 = (float(x) for x in v.quantile([0.25, 0.5, 0.75]))
    # INTERQUARTILE RANGE (IQR) = Q3 - Q1, THE SPREAD OF THE MIDDLE 50% OF VALUES.
    iqr = q3 - q1
    # 1.5 X IQR RULE: THE "FENCES" ARE Q1 - 1.5*IQR AND Q3 + 1.5*IQR. VALUES OUTSIDE THEM ARE POTENTIAL OUTLIERS.
    # EXAMPLE (GROSS FLOOR AREA): Q3 269,400 + 1.5 X 194,510 = 561,165, SO BUILDINGS ABOVE 561,165 SQ FT ARE FLAGGED.
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    # COUNT POTENTIAL OUTLIERS BELOW THE LOWER FENCE AND ABOVE THE UPPER FENCE. THEY ARE COUNTED, NOT REMOVED.
    n_low, n_high = int((v < lo_f).sum()), int((v > hi_f).sum())

    # MODE IS REPORTED ONLY WHEN IT IS MEANINGFUL: THE MOST COMMON VALUE MUST
    # OCCUR MORE THAN ONCE AND THE COLUMN MUST NOT BE (ALMOST) ALL-DISTINCT.
    # MOST COMMON VALUE AND HOW OFTEN IT OCCURS.
    vc = v.value_counts()
    mode_val, mode_cnt = float(vc.index[0]), int(vc.iloc[0])
    meaningful = mode_cnt > 1 and (v.nunique() / n) < 0.5
    # ALL NUMERIC STATISTICS FOR THE COLUMN. STD USES `ddof=1` (SAMPLE STANDARD DEVIATION, DIVIDES BY N - 1).
    # SKEWNESS > 0 MEANS A LONG RIGHT TAIL (A FEW VERY LARGE VALUES). ZERO AND NEGATIVE COUNTS FEED OTHER QUALITY FLAGS.
    out.update({
        "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()), "median": med,
        "mode": mode_val if meaningful else None, "mode_count": mode_cnt if meaningful else None,
        "mode_note": None if meaningful else "not meaningful: values are mostly distinct",
        "std": float(v.std(ddof=1)) if n > 1 else 0.0,
        "q1": q1, "q3": q3, "iqr": iqr, "lower_fence": lo_f, "upper_fence": hi_f,
        "outliers_low": n_low, "outliers_high": n_high, "outlier_count": n_low + n_high,
        "outlier_pct": _pct(n_low + n_high, n),
        "skewness": float(v.skew()) if n > 2 else None,
        "zero_count": int((v == 0).sum()), "zero_pct": _pct(int((v == 0).sum()), n),
        "negative_count": int((v < 0).sum()),
    })
    return out


def categorical_stats(s: pd.Series, n_rows: int, top_k: int) -> dict:
    # KEEP NON-MISSING, NON-BLANK VALUES AND COUNT HOW OFTEN EACH CATEGORY APPEARS (MOST FREQUENT FIRST).
    v = s.dropna().astype(str).str.strip()
    v = v[v != ""]
    n = int(len(v))
    vc = v.value_counts()
    # THE TOP-K CATEGORIES (DEFAULT 10) WITH COUNT, % OF NON-MISSING VALUES, AND % OF ALL ROWS.
    top = [{"value": str(k), "count": int(c), "pct_of_non_missing": _pct(c, n), "pct_of_rows": _pct(c, n_rows)}
           for k, c in vc.head(top_k).items()]
    # THE MOST FREQUENT VALUE(S). IF SEVERAL ARE TIED FOR FIRST, UP TO 5 ARE LISTED.
    modes = [str(k) for k, c in vc.items() if c == vc.iloc[0]][:5] if n else []
    # EVERYTHING BEYOND THE TOP-K IS SUMMARIZED AS ONE "ALL OTHER VALUES" COUNT SO THE REPORT STAYS SHORT.
    return {
        "valid_count": n, "missing_count": int(n_rows - n), "missing_pct": _pct(n_rows - n, n_rows),
        "unique": int(vc.size), "most_frequent": modes,
        "most_frequent_count": int(vc.iloc[0]) if n else 0,
        "top_values": top, "shown": len(top),
        "other_count": int(vc.iloc[top_k:].sum()) if vc.size > top_k else 0,
        "other_pct_of_non_missing": _pct(int(vc.iloc[top_k:].sum()), n) if vc.size > top_k else 0.0,
    }


def date_stats(s: pd.Series, n_rows: int, is_year: bool) -> dict:
    # INTEGER YEAR COLUMNS (E.G. "DATA YEAR"): COUNT ROWS PER YEAR AND REPORT THE EARLIEST AND LATEST YEAR.
    if is_year:
        v = pd.to_numeric(s, errors="coerce").dropna().astype(int)
        per = v.value_counts().sort_index()
        return {"valid_count": int(len(v)), "missing_count": int(n_rows - len(v)),
                "missing_pct": _pct(n_rows - len(v), n_rows),
                "min": int(v.min()) if len(v) else None, "max": int(v.max()) if len(v) else None,
                "granularity": "year", "counts_by_period": {str(k): int(c) for k, c in per.items()}}
    # REAL DATE COLUMNS: CONVERT TO DATES AND COUNT VALID VS. MISSING.
    v = pd.to_datetime(s, errors="coerce").dropna()
    out = {"valid_count": int(len(v)), "missing_count": int(n_rows - len(v)),
           "missing_pct": _pct(n_rows - len(v), n_rows)}
    if not len(v):
        return out
    # PICK A SENSIBLE TIME BUCKET FROM THE DATE RANGE: YEARS IF IT SPANS MORE THAN 3 YEARS, MONTHS IF MORE THAN 60 DAYS,
    # OTHERWISE DAYS. THEN COUNT ROWS PER BUCKET (USED FOR THE TIME PLOT).
    span_days = (v.max() - v.min()).days
    freq, label = ("YS", "year") if span_days > 3 * 365 else (("MS", "month") if span_days > 60 else ("D", "day"))
    per = v.dt.to_period(freq[0]).value_counts().sort_index()
    out.update({"min": str(v.min()), "max": str(v.max()), "span_days": int(span_days),
                "granularity": label,
                "counts_by_period": {str(k): int(c) for k, c in per.items()}})
    return out


# ------------------------------------------------------------------------------------
# RELATIONSHIPS
# ------------------------------------------------------------------------------------
def select_measure_columns(typed, infos, cfg, min_valid=10):
    """Numeric measures suitable for correlation: excludes identifiers, years, Booleans, constants."""
    # ONLY "NUMERIC MEASURE" COLUMNS WITH MORE THAN ONE DISTINCT VALUE, AT LEAST 10 VALUES, AND NOT COORDINATES.
    cols = [i["column"] for i in infos
            if i["role"] == "Numeric measure" and i["unique"] > 1 and i["non_missing"] >= min_valid
            and not i.get("exclude_from_correlation")]
    # IF THERE ARE TOO MANY, KEEP THE MOST COMPLETE ONES (TIES -> MORE DISTINCT VALUES).
    ranked = sorted(cols, key=lambda c: (-typed[c].notna().sum(), -typed[c].nunique()))
    return ranked[: cfg.max_corr_columns], cols


def relationships(typed, infos, cfg):
    measures, all_measures = select_measure_columns(typed, infos, cfg)
    # RECORD WHICH NUMERIC COLUMNS WERE USED AND WHICH NUMERIC-TYPED COLUMNS WERE LEFT OUT (IDS, YEARS, ETC.).
    out = {"numeric_columns_used": measures, "numeric_columns_available": len(all_measures),
           "excluded_from_correlation": sorted(
               i["column"] for i in infos
               if i["technical_type"] in ("integer", "float") and i["column"] not in measures)}
    # A CORRELATION NEEDS AT LEAST TWO COLUMNS; OTHERWISE EXPLAIN WHY THE ANALYSIS WAS SKIPPED.
    if len(measures) < 2:
        out["skipped"] = (f"correlation analysis needs at least 2 numeric measure columns; "
                          f"found {len(measures)}")
        return out, None
    # CORRELATION MATRICES. EACH PAIR USES ONLY ROWS WHERE BOTH COLUMNS HAVE A VALUE ("PAIRWISE COMPLETE"), AND NEEDS AT
    # LEAST 10 SUCH ROWS. PEARSON = LINEAR ASSOCIATION; SPEARMAN = RANK-BASED, LESS AFFECTED BY OUTLIERS AND SKEW.
    X = typed[measures].apply(pd.to_numeric, errors="coerce").astype(float)
    pearson = X.corr(method="pearson", min_periods=10)
    spearman = X.corr(method="spearman", min_periods=10)
    # TURN THE MATRIX INTO A LIST OF COLUMN PAIRS (EACH PAIR ONCE), WITH R, RHO AND THE NUMBER OF ROWS USED.
    pairs = []
    for a_i in range(len(measures)):
        for b_i in range(a_i + 1, len(measures)):
            a, b = measures[a_i], measures[b_i]
            r = pearson.loc[a, b]
            if pd.isna(r):
                continue
            n_pairs = int((X[a].notna() & X[b].notna()).sum())
            pairs.append({"col_a": a, "col_b": b, "pearson_r": round(float(r), 4),
                          "spearman_rho": round(float(spearman.loc[a, b]), 4)
                          if not pd.isna(spearman.loc[a, b]) else None,
                          "n_pairs": n_pairs})
    # SORT BY STRENGTH (LARGEST |R| FIRST) AND SPLIT INTO POSITIVE AND NEGATIVE RELATIONSHIPS.
    pairs.sort(key=lambda p: -abs(p["pearson_r"]))
    pos = [p for p in pairs if p["pearson_r"] > 0]
    neg = [p for p in pairs if p["pearson_r"] < 0]
    # SAVE THE HEADLINE RESULTS: STRONGEST POSITIVE/NEGATIVE PAIRS, TOP 5 OVERALL, AND PAIRS WITH |R| >= 0.95,
    # WHICH ARE LIKELY TWO VERSIONS OF THE SAME MEASUREMENT (E.G. SITE EUI AND WEATHER-NORMALIZED SITE EUI).
    out.update({
        "method": "Pearson r on pairwise-complete rows (Spearman rho shown for robustness to outliers/skew)",
        "strongest_positive": pos[:3], "strongest_negative": sorted(neg, key=lambda p: p["pearson_r"])[:3],
        "top_pairs_by_abs_r": pairs[:5],
        "near_duplicate_pairs": [p for p in pairs if abs(p["pearson_r"]) >= 0.95],
        # STRONGEST PAIRS THAT ARE NOT NEAR-DUPLICATES (|R| < 0.95) - USUALLY THE MORE INFORMATIVE ONES
        "strongest_non_redundant": [p for p in pairs if abs(p["pearson_r"]) < 0.95][:3],
        "note": "Correlation describes linear association only; it does not show causation.",
    })
    return out, pearson


def pick_group_column(infos, typed, max_levels=15):
    """Categorical/Boolean column with 2..max_levels levels and the fewest missing values."""
    # CANDIDATES: CATEGORICAL OR BOOLEAN COLUMNS WITH 2 TO 15 CATEGORIES (FEW ENOUGH TO COMPARE IN ONE CHART).
    cands = [i for i in infos if i["role"] in ("Categorical attribute", "Boolean field")
             and 2 <= i["unique"] <= max_levels]
    if not cands:
        return None
    # CHOOSE THE MOST COMPLETE ONE; IF TIED, THE ONE WITH MORE CATEGORIES.
    return sorted(cands, key=lambda i: (i["missing"], -i["unique"]))[0]["column"]


def group_comparison(typed, group_col, measure, top_k=8):
    # KEEP ROWS WHERE BOTH THE GROUP AND THE MEASURE ARE PRESENT, AND ONLY THE 8 MOST COMMON GROUPS.
    g = typed[[group_col, measure]].copy()
    g[measure] = pd.to_numeric(g[measure], errors="coerce")
    g = g.dropna()
    top_levels = g[group_col].value_counts().head(top_k).index
    g = g[g[group_col].isin(top_levels)]
    # COUNT, MEDIAN AND MEAN OF THE MEASURE WITHIN EACH GROUP, SORTED BY MEDIAN (HIGHEST FIRST).
    agg = g.groupby(group_col)[measure].agg(["count", "median", "mean"]).sort_values("median", ascending=False)
    return {"group_column": group_col, "measure": measure,
            "groups": [{"group": str(k), "count": int(r["count"]), "median": float(r["median"]),
                        "mean": float(r["mean"])} for k, r in agg.iterrows()],
            "note": "Top groups by frequency; medians computed by Python on non-missing rows."}
