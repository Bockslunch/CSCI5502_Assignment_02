"""LLM explanation layer with Python-side verification.

Flow:
  1. build_facts()      - turn the verified JSON summary into numbered fact
                          statements (F1, F2, ...). Every number in them was
                          computed by Python.
  2. build_prompt()     - the only information the model sees is those facts.
  3. call_ollama()      - local model via Ollama's HTTP API (no API key).
  4. verify_insights()  - every number in every insight must match a number in
                          the facts it cites (within rounding tolerance), cited
                          fact IDs must exist, and causal wording is rejected.
                          Failing insights are removed from the report and
                          listed in a claim-verification table.
  5. template_insights()- deterministic Python insights used when the LLM is
                          unavailable or too few LLM insights pass verification.
"""

# IMPORTS: `json` TO BUILD/READ THE MESSAGES EXCHANGED WITH OLLAMA, `re` FOR TEXT PATTERNS, AND `urllib` (BUILT INTO
# PYTHON) TO SEND THE HTTP REQUEST TO THE LOCAL OLLAMA SERVER, SO NO EXTRA PACKAGE OR API KEY IS NEEDED.
import json
import re
import urllib.error
import urllib.request

# PATTERN THAT FINDS NUMBERS IN TEXT: OPTIONAL SIGN, DIGITS WITH OPTIONAL THOUSANDS COMMAS, OPTIONAL DECIMALS.
# THE LOOK-BEHIND SKIPS DIGITS GLUED TO LETTERS, SO FACT IDS LIKE "F12" OR NAMES LIKE "CO2E" ARE NOT READ AS NUMBERS.
NUM_RX = re.compile(r"(?<![A-Za-z_])[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
# WORDS THAT CLAIM CAUSE AND EFFECT. CORRELATION IS NOT CAUSATION, SO INSIGHTS USING THESE WORDS ARE REJECTED.
CAUSAL_RX = re.compile(
    r"\b(causes?|caused|causing|leads? to|led to|results? in|resulted in|drives?|driven by|due to|"
    r"because of|makes? .* (increase|decrease)|proves?|is responsible for)\b", re.I)
# REFERENCE LIST OF ALL INSIGHT CATEGORIES. THE ACTUAL REQUIRED LIST FOR A DATASET IS BUILT BY `required_categories()`,
# WHICH DROPS "CATEGORICAL" OR "RELATIONSHIP" WHEN THE DATASET HAS NO SUCH FACTS.
REQUIRED_CATEGORIES = ["data quality", "distribution", "categorical", "relationship", "limitation", "follow-up question"]


def fmt(x, digits=4):
    """Consistent, readable number formatting shared by facts, templates and the report.

    Integers get thousands separators from 10,000 up (so years print as 1964); floats keep
    about four significant digits below 1,000 and are rounded to 0-1 decimals above that.
    Scientific notation is never used, so the LLM sees plain numbers it can copy exactly."""
    # MISSING VALUES PRINT AS "N/A".
    if x is None:
        return "n/a"
    # WHOLE NUMBERS: COMMAS ONLY FROM 10,000 UP (SO YEARS SHOW AS 1964, NOT 1,964).
    x = float(x)
    if x.is_integer() and abs(x) < 1e15:
        return f"{int(x):,}" if abs(x) >= 10000 else f"{int(x)}"
    # DECIMALS: BIG NUMBERS ROUNDED (254,897 / 1964.3), SMALLER ONES TO ABOUT 4 SIGNIFICANT DIGITS (36.66, 0.1234),
    # AND ONLY TINY NUMBERS USE SCIENTIFIC NOTATION.
    a = abs(x)
    if a >= 10000:
        return f"{x:,.0f}"
    if a >= 1000:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}g}" if a >= 1e-4 else f"{x:.2e}"


# ---------------------------------------------------------------------------------------
# 1. FACTS
# ---------------------------------------------------------------------------------------
def build_facts(summary, max_numeric=8, max_categorical=6):
    # THE FACT LIST. EACH FACT GETS AN ID (F1, F2, ...), A CATEGORY, A SENTENCE WITH PYTHON-COMPUTED NUMBERS, AND THE
    # COLUMNS IT IS ABOUT. THE VERIFIER LATER USES THE COLUMN NAMES AND NUMBERS IN THESE SENTENCES AS THE "TRUTH".
    facts = []

    def add(category, text, columns=()):
        facts.append({"id": f"F{len(facts) + 1}", "category": category, "text": text, "columns": list(columns)})

    # OVERVIEW FACTS: FILE SIZE AND HOW MANY COLUMNS OF EACH ROLE WERE FOUND.
    ov = summary["overview"]
    q = summary["data_quality"]
    add("overview", f"The file {ov['filename']} has {fmt(ov['rows'])} rows and {fmt(ov['columns'])} columns.")
    roles = ", ".join(f"{k}: {v}" for k, v in ov["role_counts"].items())
    add("overview", f"Inferred column roles (count of columns): {roles}.")

    # DATA QUALITY
    # DATA-QUALITY FACTS: DUPLICATES, OVERALL MISSINGNESS, EACH HIGH-MISSING COLUMN (UP TO 6), CONSTANT COLUMNS,
    # MIXED-TYPE COLUMNS, PLACEHOLDER VALUES, IDENTIFIER COLUMNS, AND SENSITIVE-FIELD WARNINGS.
    add("data quality", f"There are {fmt(q['duplicate_rows'])} fully duplicated rows ({q['duplicate_rows_pct']}% of rows)."
        + (" Duplicates were flagged, not removed." if q["duplicate_rows"] else ""))
    add("data quality", f"{fmt(q['total_missing_cells'])} of all cells are missing ({q['total_missing_cells_pct']}%); "
                        f"{fmt(q['rows_with_any_missing'])} rows ({q['rows_with_any_missing_pct']}%) have at least one missing value.")
    for h in q["high_missing_columns"][:6]:
        add("data quality", f"Column `{h['column']}` is missing {fmt(h['missing'])} of {fmt(ov['rows'])} values "
                            f"({h['missing_pct']}%), above the {q['high_missing_threshold_pct']:.0f}% high-missingness threshold.",
            [h["column"]])
    if q["constant_columns"]:
        add("data quality", "Columns with only one distinct non-missing value: "
            + ", ".join(f"`{c}`" for c in q["constant_columns"][:8]) + ".", q["constant_columns"][:8])
    for m in q["mixed_type_columns"][:4]:
        add("data quality", f"Column `{m['column']}` has {fmt(m['non_conforming_count'])} values ({m['non_conforming_pct']}% of "
                            f"non-missing) that do not match its inferred type ({m['technical_type']}).", [m["column"]])
    for p in q["possible_placeholder_values"][:4]:
        add("data quality", f"Column `{p['column']}` contains {fmt(p['count'])} placeholder-like values "
                            f"({', '.join(list(p['values'])[:4])}) that may mean 'missing'.", [p["column"]])
    if q["identifier_or_high_cardinality_columns"]:
        cols = [h["column"] for h in q["identifier_or_high_cardinality_columns"][:6]]
        add("data quality", "Identifier-like or very high-cardinality columns (excluded from correlations): "
            + ", ".join(f"`{c}`" for c in cols) + ".", cols)
    if q["sensitive_field_warnings"]:
        cols = [s["column"] for s in q["sensitive_field_warnings"][:6]]
        add("limitation", "Heuristic sensitive-field warnings were raised for: " + ", ".join(f"`{c}`" for c in cols)
            + ". This is a name/pattern heuristic, not a privacy review.", cols)

    # NUMERIC DISTRIBUTIONS
    # NUMERIC FACTS FOR UP TO 8 COLUMNS (MOST COMPLETE FIRST, COORDINATES LAST): A SUMMARY LINE, AN OUTLIER LINE,
    # AND A "MANY ZEROS" LINE WHEN AT LEAST 30% OF VALUES ARE EXACTLY 0.
    ns = summary["numeric_statistics"]
    coords = {c["column"] for c in summary["columns"] if c.get("exclude_from_correlation")}
    for col in sorted(ns, key=lambda c: (c in coords, -ns[c]["valid_count"]))[:max_numeric]:
        s = ns[col]
        if not s.get("valid_count"):
            continue
        add("distribution", f"`{col}`: n = {fmt(s['valid_count'])}, missing {s['missing_pct']}%, min {fmt(s['min'])}, "
                            f"Q1 {fmt(s['q1'])}, median {fmt(s['median'])}, mean {fmt(s['mean'])}, Q3 {fmt(s['q3'])}, "
                            f"max {fmt(s['max'])}, std {fmt(s['std'])}, skewness {fmt(s['skewness'], 3)}.", [col])
        add("distribution", f"`{col}` has {fmt(s['outlier_count'])} potential outliers by the 1.5×IQR rule "
                            f"({s['outlier_pct']}% of valid values; {fmt(s['outliers_high'])} above {fmt(s['upper_fence'])}, "
                            f"{fmt(s['outliers_low'])} below {fmt(s['lower_fence'])}).", [col])
        if s["zero_pct"] >= 30:
            add("data quality", f"`{col}` is exactly 0 in {fmt(s['zero_count'])} rows ({s['zero_pct']}% of valid values); "
                                f"zeros could be real or could stand for 'not recorded'.", [col])

    # CATEGORICAL
    # CATEGORICAL FACTS FOR UP TO 6 COLUMNS (SAME RANKING AS THE BAR CHARTS): DISTINCT COUNT AND THE TOP 4 VALUES.
    cs = summary["categorical_statistics"]
    from plots import rank_categoricals
    cat_cols = rank_categoricals(cs, [c for c in cs if cs[c]["role"] == "Categorical attribute" and cs[c]["unique"] >= 2])
    cat_cols = cat_cols[:max_categorical]
    for col in cat_cols:
        s = cs[col]
        top = "; ".join(f"{t['value']} = {fmt(t['count'])} ({t['pct_of_non_missing']}%)" for t in s["top_values"][:4])
        add("categorical", f"`{col}` has {fmt(s['unique'])} distinct values; most frequent: {top}.", [col])

    # DATES
    # DATE FACTS: EARLIEST AND LATEST VALUE OF EACH DATE-LIKE COLUMN.
    for col, s in summary["date_statistics"].items():
        if s.get("valid_count"):
            add("distribution", f"`{col}` ranges from {s.get('min')} to {s.get('max')} "
                                f"({fmt(s['valid_count'])} non-missing values, {s['missing_pct']}% missing).", [col])

    # RELATIONSHIPS
    # RELATIONSHIP FACTS: EITHER WHY CORRELATION WAS SKIPPED, OR THE STRONGEST NON-REDUNDANT PAIRS, THE NEAR-DUPLICATE
    # PAIRS, THE OTHER TOP PAIRS AND THE STRONGEST NEGATIVE PAIR. R IS GIVEN TO 3 DECIMALS SO 0.999 IS NOT SHOWN AS 1.00.
    rel = summary["relationships"]
    if rel.get("skipped"):
        add("relationship", f"Correlation analysis was skipped: {rel['skipped']}.")
    else:
        for p in rel.get("strongest_non_redundant", [])[:2]:
            add("relationship", f"`{p['col_a']}` and `{p['col_b']}`: Pearson r = {p['pearson_r']:.3f}, "
                                f"Spearman rho = {p['spearman_rho']:.3f}, based on {fmt(p['n_pairs'])} rows with both values.",
                [p["col_a"], p["col_b"]])
        nd = rel.get("near_duplicate_pairs", [])
        if nd:
            ex = nd[0]
            add("relationship", f"{len(nd)} pair(s) of numeric columns have |Pearson r| of 0.95 or more, e.g. `{ex['col_a']}` and "
                                f"`{ex['col_b']}` (r = {ex['pearson_r']:.3f}); such columns may be derived from each other.",
                [ex["col_a"], ex["col_b"]])
        for p in [p for p in rel.get("top_pairs_by_abs_r", [])[:3] if p not in rel.get("strongest_non_redundant", [])][:2]:
            add("relationship", f"`{p['col_a']}` and `{p['col_b']}`: Pearson r = {p['pearson_r']:.3f}, "
                                f"Spearman rho = {p['spearman_rho']:.3f}, based on {fmt(p['n_pairs'])} rows with both values.",
                [p["col_a"], p["col_b"]])
        for p in rel.get("strongest_negative", [])[:1]:
            add("relationship", f"Strongest negative pair: `{p['col_a']}` and `{p['col_b']}`, Pearson r = {p['pearson_r']:.3f} "
                                f"({fmt(p['n_pairs'])} rows).", [p["col_a"], p["col_b"]])
    # GROUP-COMPARISON FACT: MEDIAN OF A MEASURE IN EACH OF THE TOP 5 CATEGORIES.
    gc = summary.get("group_comparison")
    if gc:
        parts = "; ".join(f"{g['group']}: median {fmt(g['median'])} (n = {fmt(g['count'])})" for g in gc["groups"][:5])
        add("categorical", f"Median `{gc['measure']}` by `{gc['group_column']}`: {parts}.", [gc["measure"], gc["group_column"]])

    # LIMITATIONS
    # LIMITATION FACTS: MEANING/UNITS ARE UNDOCUMENTED, PLUS ANY ANALYSES THAT WERE SKIPPED.
    add("limitation", "Column meanings, units and valid ranges are not documented in the CSV; the profiler "
                      "infers structure only, not meaning.")
    for s in summary["skipped_analyses"][:4]:
        add("limitation", f"Skipped: {s}")
    return facts


# ---------------------------------------------------------------------------------------
# 2. PROMPT
# ---------------------------------------------------------------------------------------
def build_prompt(facts, columns, n_min=5, n_max=8):
    # FORMAT EACH FACT AS ONE LINE: "F12 [CATEGORY] SENTENCE".
    fact_lines = "\n".join(f"{f['id']} [{f['category']}] {f['text']}" for f in facts)
    # WORK OUT WHICH INSIGHT CATEGORIES TO REQUIRE (SAME LOGIC AS `required_categories()` BELOW).
    cats = {f["category"] for f in facts}
    required = ["data quality", "distribution"]
    required += ["categorical"] if "categorical" in cats else []
    required += ["relationship"] if any(f["category"] == "relationship" and "skipped" not in f["text"] for f in facts) else []
    required += ["limitation", "follow-up question"]
    req = ", ".join(f'"{c}"' for c in required)
    # THE PROMPT SENT TO THE MODEL: RULES (USE ONLY THESE FACTS, COPY NUMBERS EXACTLY, CITE FACT IDS, NO CAUSAL WORDS,
    # HONEST CORRELATION WORDING, 5-8 INSIGHTS, AT MOST 2 PER CATEGORY), THE REQUIRED JSON SHAPE, THE FACTS AND THE
    # VALID COLUMN NAMES. THE JSON EXAMPLE USES <PLACEHOLDERS> BECAUSE THE 3B MODEL COPIED CONCRETE EXAMPLE NUMBERS.
    return f"""You are helping explain an automated data profile. All statistics below were computed by Python and are verified.

RULES
- Use ONLY the facts listed below. Do not calculate new numbers (no new sums, differences, ratios or percentages).
- Copy every number exactly as it appears in the facts you cite.
- EVERY insight MUST include "fact_ids": the IDs (e.g. "F12") of the facts it uses.
- Refer to columns by their exact names in backticks, e.g. `column_name`.
- Do not guess what a column means, its units, or why a pattern exists. If meaning is unclear, say it is undocumented.
- Correlation is not causation. Never use words like "causes", "leads to", "drives", "due to", "because of".
- Describe correlation strength honestly: |r| >= 0.7 strong, 0.4-0.7 moderate, below 0.4 weak. Never say "perfect" unless r is exactly 1.000.
- Write between {n_min} and {n_max} insights, AT MOST 2 per category.
- Include at least one insight in EACH of these categories: {req}.
  A "follow-up question" insight is a question worth investigating next.
- Each insight is one or two sentences and should add interpretation, not just repeat a fact.

Return ONLY valid JSON in exactly this shape:
{{"insights": [
  {{"category": "<one of the categories above>", "text": "<one or two complete sentences using exact numbers copied from the cited facts>", "fact_ids": ["<fact id>", "<fact id>"]}}
]}}
Do not copy the angle-bracket placeholders; replace them with real content. Never write "..." in an insight.

VERIFIED FACTS
{fact_lines}

VALID COLUMN NAMES
{", ".join(columns)}
"""


def required_categories(facts):
    """Insight categories the report must cover, given which kinds of facts exist."""
    # DATA QUALITY, DISTRIBUTION, LIMITATION AND FOLLOW-UP ARE ALWAYS REQUIRED; CATEGORICAL AND RELATIONSHIP ONLY WHEN
    # THE DATASET HAS THOSE KINDS OF FACTS (E.G. DATASET B HAS NO CORRELATION FACTS, SO NO RELATIONSHIP INSIGHT IS REQUIRED).
    cats = {f["category"] for f in facts}
    req = ["data quality", "distribution"]
    if "categorical" in cats:
        req.append("categorical")
    if any(f["category"] == "relationship" and "skipped" not in f["text"] for f in facts):
        req.append("relationship")
    return req + ["limitation", "follow-up question"]


# ---------------------------------------------------------------------------------------
# 3. OLLAMA CALL
# ---------------------------------------------------------------------------------------
def call_ollama(prompt, cfg):
    """Returns (response_text, error_message). Never raises."""
    # THE REQUEST BODY FOR OLLAMA'S /API/CHAT ENDPOINT: WHICH MODEL, THE PROMPT AS A USER MESSAGE, ONE COMPLETE REPLY
    # (NO STREAMING), JSON-ONLY OUTPUT, LOW TEMPERATURE, FIXED SEED, AND AN 8,192-TOKEN CONTEXT WINDOW SO ALL FACTS FIT.
    body = json.dumps({
        "model": cfg.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": cfg.llm_temperature, "seed": cfg.random_seed, "num_ctx": 8192},
    }).encode("utf-8")
    # BUILD THE HTTP POST REQUEST TO HTTP://LOCALHOST:11434/API/CHAT.
    req = urllib.request.Request(f"{cfg.ollama_host.rstrip('/')}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    # SEND IT AND READ THE MODEL'S REPLY TEXT. EVERY FAILURE BELOW RETURNS AN ERROR MESSAGE INSTEAD OF CRASHING, SO THE
    # REST OF THE REPORT IS STILL PRODUCED WHEN OLLAMA IS OFF (THE ASSIGNMENT'S "GRACEFUL FAILURE" REQUIREMENT).
    try:
        with urllib.request.urlopen(req, timeout=cfg.llm_timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", ""), None
    # OLLAMA ANSWERED WITH AN ERROR (MOST OFTEN: THE MODEL HAS NOT BEEN PULLED).
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        return None, f"Ollama returned HTTP {e.code}: {detail} (is the model '{cfg.llm_model}' pulled?)"
    # OLLAMA IS NOT RUNNING, THE ADDRESS IS WRONG, OR THE REQUEST TIMED OUT.
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return None, f"could not reach Ollama at {cfg.ollama_host}: {e}"
    # OLLAMA REPLIED BUT NOT WITH VALID JSON.
    except json.JSONDecodeError as e:
        return None, f"Ollama response was not JSON: {e}"


def parse_insights(text):
    """Extract the insights list from the model's JSON (tolerates extra text around it)."""
    # NO REPLY -> NO INSIGHTS.
    if not text:
        return []
    # FIRST TRY TO READ THE WHOLE REPLY AS JSON; IF THAT FAILS, LOOK FOR THE PART BETWEEN THE FIRST "{" AND THE LAST "}".
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    # ACCEPT EITHER {"INSIGHTS": [...]} OR A BARE LIST, AND KEEP ONLY ENTRIES THAT HAVE TEXT.
    # CATEGORIES ARE LOWER-CASED AND FACT IDS ARE CLEANED UP SO THE VERIFIER CAN COMPARE THEM.
    items = obj.get("insights", []) if isinstance(obj, dict) else obj
    out = []
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and it.get("text"):
            ids = it.get("fact_ids") or []
            ids = [str(i).strip() for i in (ids if isinstance(ids, list) else [ids])]
            out.append({"category": str(it.get("category", "")).strip().lower(), "text": str(it["text"]).strip(),
                        "fact_ids": ids})
    return out


# ---------------------------------------------------------------------------------------
# 4. VERIFICATION
# ---------------------------------------------------------------------------------------
def numbers_in(text):
    # FIND EVERY NUMBER IN A PIECE OF TEXT AND CONVERT IT TO A FLOAT ("28,329" -> 28329.0). TRAILING PERIODS/COMMAS
    # FROM THE END OF A SENTENCE ARE STRIPPED FIRST.
    vals = []
    for tok in NUM_RX.findall(text):
        tok = tok.rstrip(".,")
        try:
            vals.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return vals


def _matches(x, pool):
    """True if x equals a verified number, allowing only for rounding.

    Counts (integers) must match exactly, unless the model visibly rounded a large number
    (e.g. 254,900 for 254,897: ends in 00, within 0.1%). Decimals may differ by rounding
    (0.5% or 0.05, whichever is larger)."""
    # COMPARE X WITH EVERY VERIFIED NUMBER. WHOLE NUMBERS (COUNTS) MUST MATCH EXACTLY, EXCEPT A VISIBLY ROUNDED LARGE
    # NUMBER ENDING IN 00 MAY BE WITHIN 0.1%. DECIMALS MAY DIFFER BY NORMAL ROUNDING (0.5% OR 0.05).
    for y in pool:
        if float(x).is_integer() and float(y).is_integer():
            tol = 0.001 * abs(y) if (abs(x) >= 1000 and x % 100 == 0) else 0.5
        else:
            tol = max(abs(y) * 0.005, 0.051 if abs(y) < 100 else 0.5)
        if abs(x - y) <= tol:
            return True
    return False


# PATTERNS FOR THE VERIFIER: UNFINISHED/PLACEHOLDER TEXT ("...", "<SOMETHING>", `X`), "R = NUMBER",
# AND THE CORRELATION STRENGTH WORDS "STRONG", "WEAK" AND "PERFECT".
PLACEHOLDER_RX = re.compile(r"\.\.\.|…|<[^<>]{1,40}>|`X`|\bcolumn_name\b")
R_RX = re.compile(r"\br\s*=\s*(-?\d*\.?\d+)")
STRONG_RX = re.compile(r"\b(very )?strong(ly)?\b|\bhighly correlated\b", re.I)
WEAK_RX = re.compile(r"\bweak(ly)?\b", re.I)
PERFECT_RX = re.compile(r"\bperfect(ly)?\b", re.I)


def _strength_problems(text, evidence_rs):
    """Reject correlation wording that the r values do not support (thresholds stated in the prompt)."""
    # USE THE PRECISE (3-DECIMAL) R FROM THE EVIDENCE FACTS: THE MODEL MAY HAVE ROUNDED 0.999 TO 1.00.
    # FOR EACH R THE INSIGHT QUOTES, LOOK UP THE CLOSEST PRECISE R IN ITS EVIDENCE FACTS; IF THE INSIGHT QUOTES NO R,
    # USE THE R VALUES FROM THE EVIDENCE DIRECTLY.
    text_rs = [float(x) for x in R_RX.findall(text)]
    if text_rs and evidence_rs:
        rs = [abs(min(evidence_rs, key=lambda e: abs(e - t))) for t in text_rs]
    else:
        rs = [abs(r) for r in (text_rs or evidence_rs)]
    if not rs:
        return []
    # THRESHOLDS (ALSO STATED IN THE PROMPT): "PERFECT" NEEDS |R| = 1, "STRONG" NEEDS |R| >= 0.7, "WEAK" NEEDS |R| < 0.4.
    r = max(rs)
    out = []
    if PERFECT_RX.search(text) and r < 1.0:
        out.append(f"calls a correlation 'perfect' but |r| = {r:.3f} < 1")
    if STRONG_RX.search(text) and r < 0.7:
        out.append(f"calls a correlation 'strong' but |r| = {r:.3f} < 0.7")
    if WEAK_RX.search(text) and r >= 0.4:
        out.append(f"calls a correlation 'weak' but |r| = {r:.3f} >= 0.4")
    return out


# STATISTIC WORDS AND THE VALUE(S) OF THE NAMED COLUMN THEY MUST MATCH. A SMALL MODEL CAN COPY A
# REAL NUMBER INTO THE WRONG CONTEXT ("MISSING 196,235" WHEN 196,235 IS THE COUNT OF ZEROS), WHICH A
# PURE "DOES THIS NUMBER EXIST SOMEWHERE" CHECK CANNOT CATCH.
# EACH ENTRY: THE STATISTIC WORD, WHETHER ITS NUMBER USUALLY COMES AFTER IT ("MEAN OF 36.66") OR BEFORE IT
# ("43,197 POTENTIAL OUTLIERS"), AND WHICH OF THE COLUMN'S REAL VALUES THAT NUMBER IS ALLOWED TO BE.
STAT_PATTERNS = [
    # (REGEX FOR THE WORD, WHERE THE NUMBER SITS, KEYS IN THE COLUMN'S STATS THAT ARE ALLOWED)
    (re.compile(r"\bmissing\b", re.I), "after_or_before", ("missing", "missing_pct", "missing_count")),
    (re.compile(r"\b(distinct|unique) values\b|\bcategories\b", re.I), "before", ("unique",)),
    (re.compile(r"\bmean\b|\baverage\b", re.I), "after", ("mean",)),
    (re.compile(r"\bmedian\b", re.I), "after", ("median",)),
    (re.compile(r"\bstandard deviation\b|\bstd\b", re.I), "after", ("std",)),
    (re.compile(r"\bminimum\b|\bmin\b", re.I), "after", ("min",)),
    (re.compile(r"\bmaximum\b|\bmax\b", re.I), "after", ("max",)),
    (re.compile(r"\boutliers?\b", re.I), "before", ("outlier_count", "outlier_pct")),
]


# OTHER STATISTIC WORDS: IF ONE SITS RIGHT BEFORE A NUMBER, THE NUMBER BELONGS TO THAT WORD (E.G. "ZEROS (65.48%)").
# "MAJORITY" WORDS: THE PERCENTAGE THAT FOLLOWS MUST BE ABOVE 50%.
OTHER_STAT_RX = re.compile(r"\bzeros?\b|\bmean\b|\bmedian\b|\boutliers?\b|\bdistinct\b|\bstandard deviation\b|"
                           r"\bmost frequent\b|\bskewness\b|\bduplicat", re.I)
MAJORITY_RX = re.compile(r"\bmajority\b|\bmost of the\b|\bmore than half\b", re.I)


def _column_stats(summary, col):
    """All Python-computed values for one column, keyed by statistic name."""
    # COLLECT EVERY PYTHON-COMPUTED VALUE FOR ONE COLUMN (MISSING COUNT/%, DISTINCT COUNT, AND ALL NUMERIC STATISTICS)
    # INTO ONE DICTIONARY, E.G. {"MISSING": 26, "MEAN": 36.66, ...}.
    out = {}
    for c in summary.get("columns", []):
        if c["column"] == col:
            out.update(missing=c["missing"], missing_pct=c["missing_pct"], unique=c["unique"])
    out.update({k: v for k, v in summary.get("numeric_statistics", {}).get(col, {}).items() if isinstance(v, (int, float))})
    cs = summary.get("categorical_statistics", {}).get(col)
    if cs:
        out["unique"] = cs["unique"]
    return out


def _nearest_number(text, span, where, window=45):
    """The number closest to a matched word: after it, before it, or after-else-before."""
    # NUMBERS WITHIN 45 CHARACTERS AFTER THE WORD AND WITHIN 45 CHARACTERS BEFORE IT.
    after = [(m.start(), m.group()) for m in NUM_RX.finditer(text[span[1]: span[1] + window])]
    before = [(m.start(), m.group()) for m in NUM_RX.finditer(text[max(0, span[0] - window): span[0]])]
    pick = None
    # PREFER THE FIRST NUMBER AFTER THE WORD WHEN THAT IS WHERE IT NORMALLY SITS ...
    if where in ("after", "after_or_before") and after:
        pick = after[0][1]
    # ... OTHERWISE THE CLOSEST NUMBER BEFORE THE WORD.
    elif where in ("before", "after_or_before") and before:
        start = max(0, span[0] - window)
        num_pos = start + before[-1][0]
        # IF ANOTHER STATISTIC WORD SITS JUST BEFORE THAT NUMBER ("ZEROS (65.48%)"), THE NUMBER BELONGS TO IT
        lead = text[max(0, num_pos - 25): num_pos]
        if OTHER_STAT_RX.search(lead):
            return None
        pick = before[-1][1]
    # NO NUMBER NEAR THE WORD. OTHERWISE CONVERT THE PICKED TEXT ("24,280") TO A NUMBER.
    if pick is None:
        return None
    try:
        return float(pick.rstrip(".,").replace(",", ""))
    except ValueError:
        return None


def statistic_problems(text, mentioned, summary):
    """Bind each statistic word to its nearest number and compare with the column's real value.
    Only applied when exactly one column is named, so the number cannot belong to another column."""
    # ONLY RUN WHEN EXACTLY ONE COLUMN IS NAMED; WITH TWO OR MORE COLUMNS WE CANNOT TELL WHICH COLUMN A NUMBER DESCRIBES.
    if not summary or len(mentioned) != 1:
        return []
    # LOOK UP THAT COLUMN'S REAL VALUES AND REMOVE THE COLUMN NAME FROM THE TEXT (NAMES CAN CONTAIN DIGITS, E.G. "2020 GEOID").
    col = mentioned[0]
    stats = _column_stats(summary, col)
    clean = text.replace(f"`{col}`", " ").replace(col, " ")
    problems = []
    # FOR EACH STATISTIC WORD FOUND IN THE INSIGHT, FIND ITS NUMBER AND CHECK IT AGAINST THE COLUMN'S REAL VALUE.
    # EXAMPLE THAT IS REJECTED: "`Electric Range` IS MISSING 196,235 VALUES" - THE REAL MISSING COUNT IS 26.
    for rx, where, keys in STAT_PATTERNS:
        allowed = [stats[k] for k in keys if stats.get(k) is not None]
        if not allowed:
            continue
        for m in rx.finditer(clean):
            x = _nearest_number(clean, m.span(), where)
            if x is None:
                # A CLAIM LIKE "VALUES ARE MISSING" WITH NO NUMBER: CHECK THE PREMISE
                if keys[0] == "missing" and stats.get("missing") == 0:
                    problems.append(f"says `{col}` has missing values, but it has 0 missing")
                continue
            # THE NUMBER DOES NOT MATCH THE STATISTIC IT IS ATTACHED TO -> RECORD A PROBLEM (THIS REJECTS THE INSIGHT).
            if not _matches(x, allowed):
                shown = ", ".join(fmt(a) for a in allowed)
                problems.append(f"'{m.group()}' is attached to {fmt(x)}, but the {keys[0].replace('_', ' ')} "
                                f"of `{col}` is {shown}")
    return problems


def verify_insights(insights, facts, columns, summary=None):
    """Check every model insight against the verified facts.

    Hard problems (-> REJECTED): a number that is in no fact, a number that only appears in facts
    about *other* columns, unknown fact IDs, causal wording, correlation wording the r value does not
    support, or a quantitative claim that names no column.
    If the model gave no fact IDs, Python matches the insight to the facts that mention the same
    column(s) and contain its numbers, and labels those citations as auto-matched."""
    # LOOK-UP TABLE FROM FACT ID ("F12") TO THE FACT ITSELF. EACH MODEL INSIGHT IS THEN CHECKED ONE AT A TIME.
    by_id = {f["id"]: f for f in facts}
    results = []
    for ins in insights:
        text = ins["text"]
        # COLUMNS NAMED IN BACKTICKS (AS THE PROMPT ASKS); OTHERWISE WHOLE-WORD MENTIONS OF NAMES >= 3 CHARS
        # STEP 1 - WHICH COLUMNS DOES THE INSIGHT NAME?
        mentioned = [c for c in columns if f"`{c}`" in text]
        if not mentioned:
            mentioned = [c for c in columns if len(c) >= 3 and re.search(rf"(?<!\w){re.escape(c)}(?!\w)", text)]
        # DROP NAMES CONTAINED IN LONGER MENTIONED NAMES (E.G. `Site EUI` INSIDE `Weather Normalized Site EUI`)
        mentioned = [c for c in mentioned if not any(c != o and c in o and o in text for o in mentioned)]
        # STEP 2 - PULL OUT THE INSIGHT'S NUMBERS, AFTER REMOVING COLUMN NAMES AND FACT IDS SO THEIR DIGITS ARE NOT COUNTED.
        text_no_cols = text
        for c in sorted(mentioned, key=len, reverse=True):  # COLUMN NAMES CAN CONTAIN DIGITS
            text_no_cols = text_no_cols.replace(c, " ")
        text_no_cols = re.sub(r"\bF\d+\b", " ", text_no_cols)
        insight_nums = numbers_in(text_no_cols)

        # STEP 3 - WHICH FACTS DOES IT CITE? `hard` PROBLEMS REJECT THE INSIGHT; `notes` ARE SHOWN BUT STILL ALLOW IT.
        # CITING A FACT ID THAT DOES NOT EXIST IS A HARD PROBLEM.
        hard, notes = [], []
        cited = [by_id[i] for i in ins["fact_ids"] if i in by_id]
        bad_ids = [i for i in ins["fact_ids"] if i not in by_id]
        if bad_ids:
            hard.append(f"cites unknown fact id(s) {bad_ids}")
        # "RELEVANT" FACTS ARE THE ONES ABOUT THE SAME COLUMN(S) AS THE INSIGHT (OR GENERAL FACTS IF NO COLUMN IS NAMED).
        relevant = [f for f in facts if mentioned and any(f"`{c}`" in f["text"] for c in mentioned)]
        if not mentioned:
            relevant = [f for f in facts if f["category"] in ("overview", "data quality", "limitation")]

        # STEP 4 - CHECK EVERY NUMBER: FIRST AGAINST THE CITED FACTS; THEN AGAINST RELEVANT FACTS (AND RECORD THOSE AS
        # "AUTO-MATCHED" CITATIONS); A NUMBER THAT ONLY APPEARS IN FACTS ABOUT A DIFFERENT COLUMN, OR IN NO FACT AT ALL,
        # IS A HARD PROBLEM.
        auto = []
        unsupported, wrong_column = [], []
        for x in insight_nums:
            if any(_matches(x, numbers_in(f["text"])) for f in cited):
                continue
            hit = next((f for f in relevant if _matches(x, numbers_in(f["text"]))), None)
            if hit:
                if hit not in auto and hit not in cited:
                    auto.append(hit)
            elif any(_matches(x, numbers_in(f["text"])) for f in facts):
                wrong_column.append(x)
            else:
                unsupported.append(x)
        if unsupported:
            hard.append("number(s) not found in any verified fact: " + ", ".join(fmt(u) for u in unsupported))
        if wrong_column:
            hard.append("number(s) come from facts about a different column: " + ", ".join(fmt(u) for u in wrong_column))
        # A NUMBER-FREE LIMITATION OR FOLLOW-UP QUESTION IS LINKED TO THE FIRST RELEVANT FACT AS ITS EVIDENCE.
        if not cited and not auto and relevant and not insight_nums:
            auto = relevant[:1] if ins["category"] in ("limitation", "follow-up question") else []
        if auto:
            notes.append("model gave no/partial fact IDs; Python matched " + ", ".join(f["id"] for f in auto))
        # ANY OTHER INSIGHT WITH NO SUPPORTING FACT AT ALL IS REJECTED.
        if not cited and not auto and ins["category"] not in ("limitation", "follow-up question"):
            hard.append("no supporting fact could be identified")
        # STEP 5 - WORDING CHECKS. CAUSAL WORDS ARE REJECTED UNLESS THE PHRASE IS QUOTED FROM THE DATA ITSELF
        # (E.G. THE CATEGORY NAME "NOT ELIGIBLE DUE TO LOW BATTERY RANGE").
        fact_text = " ".join(f["text"] for f in facts)
        for m in CAUSAL_RX.finditer(text):
            window = text[max(0, m.start() - 20): m.end() + 20]
            if not any(frag in fact_text for frag in (window, text[max(0, m.start() - 12): m.end() + 12])):
                hard.append("uses causal language")
                break
        # "MAJORITY" MUST BE FOLLOWED BY A PERCENTAGE ABOVE 50%.
        for m in MAJORITY_RX.finditer(text):
            pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text[m.end(): m.end() + 120])
            if pct and float(pct.group(1)) <= 50:
                hard.append(f"says '{m.group()}' but the percentage given is {pct.group(1)}%")
        # UNFINISHED TEXT ("...", "<COLUMN>") IS REJECTED.
        if PLACEHOLDER_RX.search(text):
            hard.append("contains placeholder or unfinished text")
        # CORRELATION WORDS ("STRONG", "PERFECT", "WEAK") MUST FIT THE PRECISE R IN THE EVIDENCE.
        evidence = cited + auto
        ev_rs = [float(x) for f in evidence for x in R_RX.findall(f["text"])]
        hard += _strength_problems(text, ev_rs)
        # A CLAIM WITH NUMBERS MUST SAY WHICH COLUMN IT IS ABOUT.
        if insight_nums and not mentioned and ins["category"] not in ("limitation", "follow-up question", "overview"):
            hard.append("quantitative claim does not name a column")
        # STEP 6 - EACH STATISTIC WORD MUST CARRY THAT COLUMN'S REAL VALUE (SEE `statistic_problems()` ABOVE).
        hard += statistic_problems(text, mentioned, summary)
        # FINAL VERDICT: ANY HARD PROBLEM -> REJECTED; ONLY NOTES -> "VERIFIED (WITH NOTE)"; NOTHING -> VERIFIED.
        # SAVE THE VERDICT, THE CITED AND AUTO-MATCHED FACTS, AND THE REASONS (THESE FILL THE CLAIM-VERIFICATION TABLE).
        status = "REJECTED" if hard else ("VERIFIED (with note)" if notes else "VERIFIED")
        results.append({**ins, "fact_ids": [f["id"] for f in cited], "auto_fact_ids": [f["id"] for f in auto],
                        "columns_mentioned": mentioned, "status": status, "problems": hard + notes,
                        "evidence": [f"{f['id']}: {f['text']}" for f in evidence]})
    return results


def assemble_insights(verification, templates, facts, n_min=5, n_max=8, per_category=2):
    """Final report list: verified model insights (max `per_category` each), then Python templates for any
    required category the model missed, keeping the total between n_min and n_max."""
    # PASS 1: TAKE VERIFIED MODEL INSIGHTS IN THE ORDER THE MODEL GAVE THEM, AT MOST 2 PER CATEGORY AND 8 IN TOTAL.
    required = required_categories(facts)
    chosen, counts = [], {}
    for v in verification:
        if v["status"].startswith("VERIFIED") and counts.get(v["category"], 0) < per_category and len(chosen) < n_max:
            chosen.append({**v, "source": "llm"})
            counts[v["category"]] = counts.get(v["category"], 0) + 1
    # PASS 2: FOR EACH REQUIRED CATEGORY THE MODEL DID NOT COVER, ADD THE PYTHON TEMPLATE INSIGHT FOR THAT CATEGORY.
    for cat in required:
        if cat in counts:
            continue
        t = next((t for t in templates if t["category"] == cat), None)
        if not t:
            continue
        if len(chosen) >= n_max:  # MAKE ROOM BY DROPPING THE LAST INSIGHT FROM A CATEGORY THAT HAS TWO
            idx = max((i for i, c in enumerate(chosen) if counts[c["category"]] > 1), default=None)
            if idx is None:
                continue
            counts[chosen[idx]["category"]] -= 1
            chosen.pop(idx)
        chosen.append(t)
        counts[cat] = 1
    # PASS 3: IF THERE ARE STILL FEWER THAN 5 INSIGHTS, TOP UP WITH MORE TEMPLATES.
    for t in templates:
        if len(chosen) >= n_min:
            break
        if t not in chosen:
            chosen.append(t)
    return chosen


# ---------------------------------------------------------------------------------------
# 5. DETERMINISTIC TEMPLATE INSIGHTS (NO LLM)
# ---------------------------------------------------------------------------------------
def template_insights(facts, n_min=5):
    """Pick one fact per required category and phrase it as a Python-generated insight."""
    # WALK THROUGH THE WANTED CATEGORIES IN ORDER AND TAKE THE FIRST UNUSED FACT OF EACH (SKIPPING "SKIPPED" NOTICES).
    # THE TEMPLATE INSIGHT IS SIMPLY THE FACT SENTENCE ITSELF, SO IT IS CORRECT BY CONSTRUCTION.
    out, used = [], set()
    wanted = ["data quality", "distribution", "categorical", "relationship", "data quality", "categorical", "limitation"]
    # PREFER THE MORE INFORMATIVE FACTS WITHIN A CATEGORY
    prefer = ("high-missingness", "potential outliers", "do not match", "placeholder", "exactly 0", "Median `")
    ordered = sorted(facts, key=lambda f: 0 if any(p in f["text"] for p in prefer) else 1)
    for cat in wanted:
        for f in ordered:
            if f["category"] == cat and f["id"] not in used and "skipped" not in f["text"].lower():
                used.add(f["id"])
                out.append({"category": cat, "text": f["text"], "fact_ids": [f["id"]], "source": "python-template"})
                break
    # FOLLOW-UP QUESTION: FIRST RULE BELOW THAT APPLIES (HIGH-MISSING COLUMN, PAIR, ZEROS, GROUP MEDIANS, GENERIC)
    fq = None
    for f in facts:
        if f["category"] == "data quality" and "high-missingness" in f["text"] and f["columns"]:
            fq = (f"Why is `{f['columns'][0]}` missing so often - is it optional, not applicable to some rows, "
                  f"or a collection gap? (see {f['id']})", f["id"])
            break
    # THE CORRELATION PAIR RULE (USED WHEN NO COLUMN IS HIGHLY MISSING).
    # THE "MANY ZEROS" RULE.
    # THE GROUP-MEDIANS RULE.
    # LAST RESORT: A GENERIC QUESTION ABOUT THE PUBLISHER'S DATA DICTIONARY.
    if not fq:
        for f in facts:
            if f["category"] == "relationship" and len(f["columns"]) == 2:
                fq = (f"Does the association between `{f['columns'][0]}` and `{f['columns'][1]}` hold within "
                      f"subgroups, or is it explained by a third variable? (see {f['id']})", f["id"])
                break
    if not fq:
        for f in facts:
            if "exactly 0" in f["text"] and f["columns"]:
                fq = (f"Do the zeros in `{f['columns'][0]}` mean a true value of zero or 'not recorded'? A data "
                      f"dictionary from the publisher is needed to decide. (see {f['id']})", f["id"])
                break
    if not fq:
        for f in facts:
            if f["text"].startswith("Median `") and len(f["columns"]) == 2:
                fq = (f"Are the differences in `{f['columns'][0]}` across `{f['columns'][1]}` groups stable over time "
                      f"and across other subgroups? (see {f['id']})", f["id"])
                break
    if not fq:
        fq = ("Which columns does the publisher's data dictionary define, and do their documented units and "
              "valid ranges match the values profiled here?", "")
    # SMALL OR SIMPLE DATASETS: TOP UP WITH ANY REMAINING FACTS SO THE REPORT STILL HAS N_MIN INSIGHTS
    for f in ordered:
        if len(out) >= n_min - 1:
            break
        if f["id"] not in used:
            used.add(f["id"])
            out.append({"category": f["category"], "text": f["text"], "fact_ids": [f["id"]], "source": "python-template"})
    # THE FOLLOW-UP QUESTION IS ALWAYS ADDED LAST.
    out.append({"category": "follow-up question", "text": fq[0], "fact_ids": [fq[1]] if fq[1] else [],
                "source": "python-template"})
    return out
