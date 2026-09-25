"""Automated tests (optional extension). Run from the repository root:  python -m pytest -q tests

They check the statistics against hand-computable values, the edge cases the
assignment names (no numeric / no categorical columns, empty columns), and that
the LLM verifier rejects invented numbers and causal claims.
"""

# IMPORTS: STANDARD LIBRARY, NUMPY/PANDAS FOR TEST DATA, AND PYTEST (THE TEST RUNNER).
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# LET THE TESTS IMPORT THE MODULES IN `src/`, THEN IMPORT THE FUNCTIONS UNDER TEST.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import numeric_stats  # noqa: E402
from config import ProfilerConfig  # noqa: E402
from inference import infer_column  # noqa: E402
from llm import build_facts, parse_insights, template_insights, verify_insights  # noqa: E402
from profiler import generate_profile  # noqa: E402

# ONE DEFAULT CONFIG SHARED BY THE TESTS.
CFG = ProfilerConfig()


# TEST 1: THE NUMERIC STATISTICS MATCH VALUES WORKED OUT BY HAND FOR A TINY COLUMN [1, 2, 3, 4, 100, MISSING].
def test_numeric_stats_match_hand_calculation():
    # 5 VALID VALUES + 1 MISSING; MEDIAN 3; MEAN (1+2+3+4+100)/5 = 22.
    s = pd.Series([1, 2, 3, 4, 100, np.nan])
    st = numeric_stats(s, n_rows=6, cfg=CFG)
    assert st["valid_count"] == 5 and st["missing_count"] == 1
    assert st["median"] == 3 and st["mean"] == pytest.approx(22.0)
    # PANDAS LINEAR QUARTILES: Q1 = 2, Q3 = 4, IQR = 2, UPPER FENCE = 7 -> 100 IS THE ONLY OUTLIER
    assert (st["q1"], st["q3"], st["iqr"]) == (2, 4, 2)
    # 100 IS ABOVE THE UPPER FENCE OF 7, SO IT IS THE ONLY OUTLIER; STD MUST EQUAL NUMPY'S SAMPLE STD (`ddof=1`).
    assert st["outlier_count"] == 1 and st["outliers_high"] == 1
    assert st["std"] == pytest.approx(np.std([1, 2, 3, 4, 100], ddof=1))


# TEST 2: ROLE INFERENCE - EACH (VALUES, EXPECTED ROLE) PAIR BELOW IS RUN AS ITS OWN TEST CASE:
# NUMBERS -> NUMERIC, Y/N -> BOOLEAN, ISO DATES -> DATE-LIKE, REPEATED LABELS -> CATEGORICAL,
# ALL-DISTINCT CODES -> IDENTIFIER, HALF NUMBERS / HALF TEXT -> MIXED.
@pytest.mark.parametrize("values,role", [
    (["1", "2", "3.5", "4"] * 10, "Numeric measure"),
    (["Y", "N"] * 20, "Boolean field"),
    (["2021-01-05", "2022-03-01"] * 20, "Date-like field"),
    (["red", "blue", "green"] * 20, "Categorical attribute"),
    ([f"A{i:05d}" for i in range(40)], "Identifier-like field"),
    (["1", "2", "x", "y"] * 10, "Unknown or mixed type"),
])
def test_role_inference(values, role):
    assert infer_column("col", pd.Series(values, dtype="object"), CFG)["role"] == role


# TEST 3: A NUMERIC COLUMN NAMED "ZIP CODE" MUST BE AN IDENTIFIER, NOT A NUMERIC MEASURE.
def test_numeric_code_names_are_not_measures():
    info = infer_column("ZIP Code", pd.Series(["80301", "80302", "80303"] * 10), CFG)
    assert info["role"] == "Identifier-like field"


# HELPER: SAVE A SMALL DATAFRAME AS A CSV IN A TEMPORARY FOLDER, RUN THE WHOLE PIPELINE WITHOUT THE LLM, AND CHECK
# THAT ALL FIVE REQUIRED OUTPUT FILES EXIST AND THAT THE JSON FILE IS VALID.
def _run(tmp_path, df, name):
    csv = tmp_path / f"{name}.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "out" / name
    summary = generate_profile(csv, out, use_llm=False)
    for f in ("report.md", "column_profile.csv", "analysis_summary.json", "llm_prompt.txt", "llm_response.txt"):
        assert (out / f).exists(), f
    json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))  # VALID JSON
    return summary, out


# TEST 4: A CSV WITH ONLY TEXT CATEGORIES MUST STILL RUN AND SAY THAT THE NUMERIC ANALYSIS WAS SKIPPED.
def test_no_numeric_columns(tmp_path):
    df = pd.DataFrame({"a": list("xyzxyz") * 5, "b": list("ppqqrr") * 5})
    summary, out = _run(tmp_path, df, "cats_only")
    assert any("numeric" in s for s in summary["skipped_analyses"])
    assert "Skipped" in (out / "report.md").read_text(encoding="utf-8")


# TEST 5: A CSV WITH ONLY NUMBERS AND ONE COMPLETELY EMPTY COLUMN: THE EMPTY COLUMN IS FLAGGED, THE CATEGORICAL
# ANALYSIS IS REPORTED AS SKIPPED, AND AT LEAST 3 PLOTS ARE STILL MADE.
def test_no_categorical_columns_and_empty_column(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50), "empty": [None] * 50})
    summary, _ = _run(tmp_path, df, "nums_only")
    assert "empty" in summary["data_quality"]["empty_columns"]
    assert any("categorical" in s for s in summary["skipped_analyses"])
    assert len(summary["plots"]) >= 3


# TEST 6: GRACEFUL FAILURE - POINT THE PROGRAM AT A PORT WHERE NO OLLAMA IS RUNNING. THE RUN MUST FINISH, MARK THE LLM
# AS UNAVAILABLE, SAY SO IN THE REPORT, AND STILL PRODUCE AT LEAST 5 (PYTHON TEMPLATE) INSIGHTS.
def test_llm_unavailable_is_graceful(tmp_path):
    df = pd.DataFrame({"x": range(30), "g": list("ab") * 15})
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    cfg = ProfilerConfig(ollama_host="http://127.0.0.1:9", llm_timeout_seconds=2)
    summary = generate_profile(csv, tmp_path / "o", use_llm=True, config=cfg)
    assert summary["llm"]["status"] == "unavailable"
    assert "skipped because the model was unavailable" in (tmp_path / "o" / "report.md").read_text(encoding="utf-8")
    assert len(summary["insights"]) >= 5


# TEST 7: THE VERIFIER ACCEPTS A CORRECT CLAIM, REJECTS AN INVENTED NUMBER (900), AND REJECTS CAUSAL WORDING ("CAUSE").
def test_verifier_rejects_invented_numbers_and_causation():
    facts = [{"id": "F1", "category": "data quality", "columns": ["score"],
              "text": "Column `score` is missing 842 of 5,210 values (16.16%), above the 30% threshold."}]
    response = json.dumps({"insights": [
        {"category": "data quality", "text": "`score` is missing 842 of 5,210 values (16.16%).", "fact_ids": ["F1"]},
        {"category": "data quality", "text": "`score` is missing 900 values.", "fact_ids": ["F1"]},
        {"category": "data quality", "text": "Missing `score` values cause bias.", "fact_ids": ["F1"]},
    ]})
    res = verify_insights(parse_insights(response), facts, ["score"])
    assert [r["status"] for r in res] == ["VERIFIED", "REJECTED", "REJECTED"]


# TEST 8: MIRRORS REAL `qwen2.5:3b` MISTAKES - NO FACT IDS (PYTHON SHOULD MATCH THEM), "STRONG" FOR R = -0.387,
# "PERFECT" FOR R = 0.999, AND AN R VALUE TAKEN FROM A FACT ABOUT DIFFERENT COLUMNS.
def test_verifier_auto_matches_missing_fact_ids_and_rejects_overstated_strength():
    # MIRRORS REAL QWEN2.5:3B OUTPUT: NO FACT_IDS, AND "STRONG"/"PERFECT" FOR WEAK / 0.999 CORRELATIONS
    facts = [
        {"id": "F1", "category": "data quality", "columns": ["a"], "text": "Column `a` is missing 5 of 50 values (10%)."},
        {"id": "F2", "category": "relationship", "columns": ["a", "b"],
         "text": "`a` and `b`: Pearson r = -0.387, Spearman rho = -0.390, based on 40 rows with both values."},
        {"id": "F3", "category": "relationship", "columns": ["c", "d"],
         "text": "`c` and `d`: Pearson r = 0.999, Spearman rho = 0.995, based on 40 rows with both values."},
    ]
    response = json.dumps({"insights": [
        {"category": "data quality", "text": "`a` is missing 5 of 50 values (10%)."},
        {"category": "relationship", "text": "`a` and `b` are strongly and negatively correlated (r = -0.39)."},
        {"category": "relationship", "text": "`c` and `d` are perfectly correlated (r = 1.00)."},
        {"category": "relationship", "text": "`c` and `d` have a moderate link, r = -0.39."},
    ]})
    res = verify_insights(parse_insights(response), facts, ["a", "b", "c", "d"])
    assert res[0]["status"].startswith("VERIFIED") and res[0]["auto_fact_ids"] == ["F1"]
    assert res[1]["status"] == "REJECTED"   # 'STRONG' BUT |R| = 0.387
    assert res[2]["status"] == "REJECTED"   # 'PERFECT' BUT R = 0.999
    assert res[3]["status"] == "REJECTED"   # -0.39 BELONGS TO A FACT ABOUT OTHER COLUMNS


# TEST 9: IF THE MODEL RETURNS SIX DATA-QUALITY INSIGHTS, ONLY TWO ARE KEPT AND PYTHON TEMPLATES FILL EVERY OTHER
# REQUIRED CATEGORY, WITH 5-8 INSIGHTS IN TOTAL.
def test_assembly_covers_required_categories_and_caps_per_category():
    from llm import assemble_insights
    facts = [{"id": f"F{i}", "category": c, "columns": [], "text": t} for i, (c, t) in enumerate([
        ("data quality", "Column `a` is missing 5 of 50 values (10%), above the 30% high-missingness threshold."),
        ("distribution", "`a` has 3 potential outliers"), ("categorical", "`g` has 2 distinct values"),
        ("relationship", "`a` and `b`: Pearson r = 0.800"), ("limitation", "Column meanings are undocumented.")], 1)]
    llm_out = [{"category": "data quality", "text": f"dq {i}", "fact_ids": ["F1"], "status": "VERIFIED"} for i in range(6)]
    final = assemble_insights(llm_out, template_insights(facts), facts)
    cats = [i["category"] for i in final]
    assert cats.count("data quality") <= 2 and 5 <= len(final) <= 8
    for c in ("distribution", "categorical", "relationship", "limitation", "follow-up question"):
        assert c in cats


# TEST 10: A NUMBER MUST MATCH THE STATISTIC IT IS ATTACHED TO: "MISSING 196,235" (REALLY THE ZERO COUNT) AND "VIN VALUES
# ARE MISSING" (VIN HAS 0 MISSING) ARE REJECTED; A CORRECT MEAN IS ACCEPTED.
def test_statistic_words_must_match_the_named_columns_value():
    # REAL QWEN2.5:3B MISTAKES: ZEROS REPORTED AS "MISSING", TOTAL ROWS REPORTED AS MISSING, FALSE PREMISE
    summary = {"columns": [{"column": "range", "missing": 26, "missing_pct": 0.01, "unique": 117},
                           {"column": "vin", "missing": 0, "missing_pct": 0.0, "unique": 18376}],
               "numeric_statistics": {"range": {"mean": 36.66, "outlier_count": 43197, "outlier_pct": 14.41}},
               "categorical_statistics": {}}
    facts = [{"id": "F1", "category": "data quality", "columns": ["range"],
              "text": "`range` is exactly 0 in 196,235 rows (65.48% of valid values); n = 299,679, missing 0.01%, mean 36.66."},
             {"id": "F2", "category": "data quality", "columns": [], "text": "The file has 299,705 rows."}]
    response = json.dumps({"insights": [
        {"category": "data quality", "text": "`range` is missing 196,235 of 299,679 values (65.48%).", "fact_ids": ["F1"]},
        {"category": "follow-up question", "text": "Why are `vin` values missing for some vehicles?", "fact_ids": ["F2"]},
        {"category": "distribution", "text": "`range` has a mean of 36.66.", "fact_ids": ["F1"]},
    ]})
    res = verify_insights(parse_insights(response), facts, ["range", "vin"], summary)
    assert [r["status"][:8] for r in res] == ["REJECTED", "REJECTED", "VERIFIED"]


# TEST 11: UNFINISHED TEXT ("...", "<COLUMN>") IS REJECTED; A COMPLETE QUESTION WITH THE RIGHT NUMBER IS ACCEPTED.
def test_placeholder_text_is_rejected():
    facts = [{"id": "F1", "category": "data quality", "columns": ["city"], "text": "Column `city` is missing 11 of 100 values (11%)."}]
    summary = {"columns": [{"column": "city", "missing": 11, "missing_pct": 11.0, "unique": 5}],
               "numeric_statistics": {}, "categorical_statistics": {}}
    response = json.dumps({"insights": [
        {"category": "follow-up question", "text": "Why are `city` values missing for ...?", "fact_ids": ["F1"]},
        {"category": "data quality", "text": "Column `<column>` is missing <count> values.", "fact_ids": ["F1"]},
        {"category": "follow-up question", "text": "Why are 11 `city` values missing?", "fact_ids": ["F1"]}]})
    res = verify_insights(parse_insights(response), facts, ["city"], summary)
    assert [r["status"][:8] for r in res] == ["REJECTED", "REJECTED", "VERIFIED"]


# TEST 12: "ZEROS (65.48%) ... MISSING" IS ACCEPTED (THE % BELONGS TO "ZEROS"), A CATEGORY NAME CONTAINING "DUE TO" IS NOT
# TREATED AS A CAUSAL CLAIM, AND "MAJORITY ... (8.07%)" IS REJECTED.
def test_numbers_bind_to_their_own_statistic_and_quoted_values_are_not_causal():
    summary = {"columns": [{"column": "range", "missing": 26, "missing_pct": 0.01, "unique": 117},
                           {"column": "elig", "missing": 0, "missing_pct": 0.0, "unique": 3}],
               "numeric_statistics": {"range": {"mean": 36.66}}, "categorical_statistics": {"elig": {"unique": 3}}}
    facts = [{"id": "F9", "category": "data quality", "columns": ["range"],
              "text": "`range` is exactly 0 in 196,235 rows (65.48% of valid values); zeros could stand for 'not recorded'."},
             {"id": "F12", "category": "categorical", "columns": ["elig"],
              "text": "`elig` has 3 distinct values; most frequent: Eligible = 79,293 (26.46%); Not eligible due to low battery range = 24,184 (8.07%)."}]
    response = json.dumps({"insights": [
        {"category": "distribution", "text": "`range` has a high proportion of zeros (65.48%), which could indicate missing data.", "fact_ids": ["F9"]},
        {"category": "categorical", "text": "In `elig`, 8.07% are Not eligible due to low battery range.", "fact_ids": ["F12"]},
        {"category": "categorical", "text": "In `elig`, the majority are Not eligible due to low battery range (8.07%).", "fact_ids": ["F12"]},
    ]})
    res = verify_insights(parse_insights(response), facts, ["range", "elig"], summary)
    assert [r["status"][:8] for r in res] == ["VERIFIED", "VERIFIED", "REJECTED"]
