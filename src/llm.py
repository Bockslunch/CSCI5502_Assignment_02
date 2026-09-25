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

import json
import re
import urllib.error
import urllib.request

NUM_RX = re.compile(r"(?<![A-Za-z_])[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
CAUSAL_RX = re.compile(
    r"\b(causes?|caused|causing|leads? to|led to|results? in|resulted in|drives?|driven by|due to|"
    r"because of|makes? .* (increase|decrease)|proves?|is responsible for)\b", re.I)
REQUIRED_CATEGORIES = ["data quality", "distribution", "categorical", "relationship", "limitation", "follow-up question"]


def fmt(x, digits=4):
    """Consistent, readable number formatting shared by facts, templates and the report.

    Integers get thousands separators from 10,000 up (so years print as 1964); floats keep
    about four significant digits below 1,000 and are rounded to 0-1 decimals above that.
    Scientific notation is never used, so the LLM sees plain numbers it can copy exactly."""
    if x is None:
        return "n/a"
    x = float(x)
    if x.is_integer() and abs(x) < 1e15:
        return f"{int(x):,}" if abs(x) >= 10000 else f"{int(x)}"
    a = abs(x)
    if a >= 10000:
        return f"{x:,.0f}"
    if a >= 1000:
        return f"{x:.1f}"
    if a >= 1:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}g}" if a >= 1e-4 else f"{x:.2e}"


# ---------------------------------------------------------------------------------------
# 1. Facts
# ---------------------------------------------------------------------------------------
def build_facts(summary, max_numeric=8, max_categorical=6):
    facts = []

    def add(category, text, columns=()):
        facts.append({"id": f"F{len(facts) + 1}", "category": category, "text": text, "columns": list(columns)})

    ov = summary["overview"]
    q = summary["data_quality"]
    add("overview", f"The file {ov['filename']} has {fmt(ov['rows'])} rows and {fmt(ov['columns'])} columns.")
    roles = ", ".join(f"{k}: {v}" for k, v in ov["role_counts"].items())
    add("overview", f"Inferred column roles (count of columns): {roles}.")

    # Data quality
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

    # Numeric distributions
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

    # Categorical
    cs = summary["categorical_statistics"]
    from plots import rank_categoricals
    cat_cols = rank_categoricals(cs, [c for c in cs if cs[c]["role"] == "Categorical attribute" and cs[c]["unique"] >= 2])
    cat_cols = cat_cols[:max_categorical]
    for col in cat_cols:
        s = cs[col]
        top = "; ".join(f"{t['value']} = {fmt(t['count'])} ({t['pct_of_non_missing']}%)" for t in s["top_values"][:4])
        add("categorical", f"`{col}` has {fmt(s['unique'])} distinct values; most frequent: {top}.", [col])

    # Dates
    for col, s in summary["date_statistics"].items():
        if s.get("valid_count"):
            add("distribution", f"`{col}` ranges from {s.get('min')} to {s.get('max')} "
                                f"({fmt(s['valid_count'])} non-missing values, {s['missing_pct']}% missing).", [col])

    # Relationships
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
    gc = summary.get("group_comparison")
    if gc:
        parts = "; ".join(f"{g['group']}: median {fmt(g['median'])} (n = {fmt(g['count'])})" for g in gc["groups"][:5])
        add("categorical", f"Median `{gc['measure']}` by `{gc['group_column']}`: {parts}.", [gc["measure"], gc["group_column"]])

    # Limitations
    add("limitation", "Column meanings, units and valid ranges are not documented in the CSV; the profiler "
                      "infers structure only, not meaning.")
    for s in summary["skipped_analyses"][:4]:
        add("limitation", f"Skipped: {s}")
    return facts


# ---------------------------------------------------------------------------------------
# 2. Prompt
# ---------------------------------------------------------------------------------------
def build_prompt(facts, columns, n_min=5, n_max=8):
    fact_lines = "\n".join(f"{f['id']} [{f['category']}] {f['text']}" for f in facts)
    cats = {f["category"] for f in facts}
    required = ["data quality", "distribution"]
    required += ["categorical"] if "categorical" in cats else []
    required += ["relationship"] if any(f["category"] == "relationship" and "skipped" not in f["text"] for f in facts) else []
    required += ["limitation", "follow-up question"]
    req = ", ".join(f'"{c}"' for c in required)
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
    cats = {f["category"] for f in facts}
    req = ["data quality", "distribution"]
    if "categorical" in cats:
        req.append("categorical")
    if any(f["category"] == "relationship" and "skipped" not in f["text"] for f in facts):
        req.append("relationship")
    return req + ["limitation", "follow-up question"]


# ---------------------------------------------------------------------------------------
# 3. Ollama call
# ---------------------------------------------------------------------------------------
def call_ollama(prompt, cfg):
    """Returns (response_text, error_message). Never raises."""
    body = json.dumps({
        "model": cfg.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": cfg.llm_temperature, "seed": cfg.random_seed, "num_ctx": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(f"{cfg.ollama_host.rstrip('/')}/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=cfg.llm_timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("message", {}).get("content", ""), None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "ignore")[:300]
        return None, f"Ollama returned HTTP {e.code}: {detail} (is the model '{cfg.llm_model}' pulled?)"
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return None, f"could not reach Ollama at {cfg.ollama_host}: {e}"
    except json.JSONDecodeError as e:
        return None, f"Ollama response was not JSON: {e}"


def parse_insights(text):
    """Extract the insights list from the model's JSON (tolerates extra text around it)."""
    if not text:
        return []
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
# 4. Verification
# ---------------------------------------------------------------------------------------
def numbers_in(text):
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
    for y in pool:
        if float(x).is_integer() and float(y).is_integer():
            tol = 0.001 * abs(y) if (abs(x) >= 1000 and x % 100 == 0) else 0.5
        else:
            tol = max(abs(y) * 0.005, 0.051 if abs(y) < 100 else 0.5)
        if abs(x - y) <= tol:
            return True
    return False


PLACEHOLDER_RX = re.compile(r"\.\.\.|…|<[^<>]{1,40}>|`X`|\bcolumn_name\b")
R_RX = re.compile(r"\br\s*=\s*(-?\d*\.?\d+)")
STRONG_RX = re.compile(r"\b(very )?strong(ly)?\b|\bhighly correlated\b", re.I)
WEAK_RX = re.compile(r"\bweak(ly)?\b", re.I)
PERFECT_RX = re.compile(r"\bperfect(ly)?\b", re.I)


def _strength_problems(text, evidence_rs):
    """Reject correlation wording that the r values do not support (thresholds stated in the prompt)."""
    # Use the precise (3-decimal) r from the evidence facts: the model may have rounded 0.999 to 1.00.
    text_rs = [float(x) for x in R_RX.findall(text)]
    if text_rs and evidence_rs:
        rs = [abs(min(evidence_rs, key=lambda e: abs(e - t))) for t in text_rs]
    else:
        rs = [abs(r) for r in (text_rs or evidence_rs)]
    if not rs:
        return []
    r = max(rs)
    out = []
    if PERFECT_RX.search(text) and r < 1.0:
        out.append(f"calls a correlation 'perfect' but |r| = {r:.3f} < 1")
    if STRONG_RX.search(text) and r < 0.7:
        out.append(f"calls a correlation 'strong' but |r| = {r:.3f} < 0.7")
    if WEAK_RX.search(text) and r >= 0.4:
        out.append(f"calls a correlation 'weak' but |r| = {r:.3f} >= 0.4")
    return out


# Statistic words and the value(s) of the named column they must match. A small model can copy a
# real number into the wrong context ("missing 196,235" when 196,235 is the count of zeros), which a
# pure "does this number exist somewhere" check cannot catch.
STAT_PATTERNS = [
    # (regex for the word, where the number sits, keys in the column's stats that are allowed)
    (re.compile(r"\bmissing\b", re.I), "after_or_before", ("missing", "missing_pct", "missing_count")),
    (re.compile(r"\b(distinct|unique) values\b|\bcategories\b", re.I), "before", ("unique",)),
    (re.compile(r"\bmean\b|\baverage\b", re.I), "after", ("mean",)),
    (re.compile(r"\bmedian\b", re.I), "after", ("median",)),
    (re.compile(r"\bstandard deviation\b|\bstd\b", re.I), "after", ("std",)),
    (re.compile(r"\bminimum\b|\bmin\b", re.I), "after", ("min",)),
    (re.compile(r"\bmaximum\b|\bmax\b", re.I), "after", ("max",)),
    (re.compile(r"\boutliers?\b", re.I), "before", ("outlier_count", "outlier_pct")),
]


OTHER_STAT_RX = re.compile(r"\bzeros?\b|\bmean\b|\bmedian\b|\boutliers?\b|\bdistinct\b|\bstandard deviation\b|"
                           r"\bmost frequent\b|\bskewness\b|\bduplicat", re.I)
MAJORITY_RX = re.compile(r"\bmajority\b|\bmost of the\b|\bmore than half\b", re.I)


def _column_stats(summary, col):
    """All Python-computed values for one column, keyed by statistic name."""
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
    after = [(m.start(), m.group()) for m in NUM_RX.finditer(text[span[1]: span[1] + window])]
    before = [(m.start(), m.group()) for m in NUM_RX.finditer(text[max(0, span[0] - window): span[0]])]
    pick = None
    if where in ("after", "after_or_before") and after:
        pick = after[0][1]
    elif where in ("before", "after_or_before") and before:
        start = max(0, span[0] - window)
        num_pos = start + before[-1][0]
        # if another statistic word sits just before that number ("zeros (65.48%)"), the number belongs to it
        lead = text[max(0, num_pos - 25): num_pos]
        if OTHER_STAT_RX.search(lead):
            return None
        pick = before[-1][1]
    if pick is None:
        return None
    try:
        return float(pick.rstrip(".,").replace(",", ""))
    except ValueError:
        return None


def statistic_problems(text, mentioned, summary):
    """Bind each statistic word to its nearest number and compare with the column's real value.
    Only applied when exactly one column is named, so the number cannot belong to another column."""
    if not summary or len(mentioned) != 1:
        return []
    col = mentioned[0]
    stats = _column_stats(summary, col)
    clean = text.replace(f"`{col}`", " ").replace(col, " ")
    problems = []
    for rx, where, keys in STAT_PATTERNS:
        allowed = [stats[k] for k in keys if stats.get(k) is not None]
        if not allowed:
            continue
        for m in rx.finditer(clean):
            x = _nearest_number(clean, m.span(), where)
            if x is None:
                # a claim like "values are missing" with no number: check the premise
                if keys[0] == "missing" and stats.get("missing") == 0:
                    problems.append(f"says `{col}` has missing values, but it has 0 missing")
                continue
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
    by_id = {f["id"]: f for f in facts}
    results = []
    for ins in insights:
        text = ins["text"]
        # columns named in backticks (as the prompt asks); otherwise whole-word mentions of names >= 3 chars
        mentioned = [c for c in columns if f"`{c}`" in text]
        if not mentioned:
            mentioned = [c for c in columns if len(c) >= 3 and re.search(rf"(?<!\w){re.escape(c)}(?!\w)", text)]
        # drop names contained in longer mentioned names (e.g. `Site EUI` inside `Weather Normalized Site EUI`)
        mentioned = [c for c in mentioned if not any(c != o and c in o and o in text for o in mentioned)]
        text_no_cols = text
        for c in sorted(mentioned, key=len, reverse=True):  # column names can contain digits
            text_no_cols = text_no_cols.replace(c, " ")
        text_no_cols = re.sub(r"\bF\d+\b", " ", text_no_cols)
        insight_nums = numbers_in(text_no_cols)

        hard, notes = [], []
        cited = [by_id[i] for i in ins["fact_ids"] if i in by_id]
        bad_ids = [i for i in ins["fact_ids"] if i not in by_id]
        if bad_ids:
            hard.append(f"cites unknown fact id(s) {bad_ids}")
        # facts about the same column(s) as the insight
        relevant = [f for f in facts if mentioned and any(f"`{c}`" in f["text"] for c in mentioned)]
        if not mentioned:
            relevant = [f for f in facts if f["category"] in ("overview", "data quality", "limitation")]

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
        if not cited and not auto and relevant and not insight_nums:
            auto = relevant[:1] if ins["category"] in ("limitation", "follow-up question") else []
        if auto:
            notes.append("model gave no/partial fact IDs; Python matched " + ", ".join(f["id"] for f in auto))
        if not cited and not auto and ins["category"] not in ("limitation", "follow-up question"):
            hard.append("no supporting fact could be identified")
        fact_text = " ".join(f["text"] for f in facts)
        for m in CAUSAL_RX.finditer(text):
            window = text[max(0, m.start() - 20): m.end() + 20]
            if not any(frag in fact_text for frag in (window, text[max(0, m.start() - 12): m.end() + 12])):
                hard.append("uses causal language")
                break
        for m in MAJORITY_RX.finditer(text):
            pct = re.search(r"(\d+(?:\.\d+)?)\s*%", text[m.end(): m.end() + 120])
            if pct and float(pct.group(1)) <= 50:
                hard.append(f"says '{m.group()}' but the percentage given is {pct.group(1)}%")
        if PLACEHOLDER_RX.search(text):
            hard.append("contains placeholder or unfinished text")
        evidence = cited + auto
        ev_rs = [float(x) for f in evidence for x in R_RX.findall(f["text"])]
        hard += _strength_problems(text, ev_rs)
        if insight_nums and not mentioned and ins["category"] not in ("limitation", "follow-up question", "overview"):
            hard.append("quantitative claim does not name a column")
        hard += statistic_problems(text, mentioned, summary)
        status = "REJECTED" if hard else ("VERIFIED (with note)" if notes else "VERIFIED")
        results.append({**ins, "fact_ids": [f["id"] for f in cited], "auto_fact_ids": [f["id"] for f in auto],
                        "columns_mentioned": mentioned, "status": status, "problems": hard + notes,
                        "evidence": [f"{f['id']}: {f['text']}" for f in evidence]})
    return results


def assemble_insights(verification, templates, facts, n_min=5, n_max=8, per_category=2):
    """Final report list: verified model insights (max `per_category` each), then Python templates for any
    required category the model missed, keeping the total between n_min and n_max."""
    required = required_categories(facts)
    chosen, counts = [], {}
    for v in verification:
        if v["status"].startswith("VERIFIED") and counts.get(v["category"], 0) < per_category and len(chosen) < n_max:
            chosen.append({**v, "source": "llm"})
            counts[v["category"]] = counts.get(v["category"], 0) + 1
    for cat in required:
        if cat in counts:
            continue
        t = next((t for t in templates if t["category"] == cat), None)
        if not t:
            continue
        if len(chosen) >= n_max:  # make room by dropping the last insight from a category that has two
            idx = max((i for i, c in enumerate(chosen) if counts[c["category"]] > 1), default=None)
            if idx is None:
                continue
            counts[chosen[idx]["category"]] -= 1
            chosen.pop(idx)
        chosen.append(t)
        counts[cat] = 1
    for t in templates:
        if len(chosen) >= n_min:
            break
        if t not in chosen:
            chosen.append(t)
    return chosen


# ---------------------------------------------------------------------------------------
# 5. Deterministic template insights (no LLM)
# ---------------------------------------------------------------------------------------
def template_insights(facts, n_min=5):
    """Pick one fact per required category and phrase it as a Python-generated insight."""
    out, used = [], set()
    wanted = ["data quality", "distribution", "categorical", "relationship", "data quality", "categorical", "limitation"]
    # prefer the more informative facts within a category
    prefer = ("high-missingness", "potential outliers", "do not match", "placeholder", "exactly 0", "Median `")
    ordered = sorted(facts, key=lambda f: 0 if any(p in f["text"] for p in prefer) else 1)
    for cat in wanted:
        for f in ordered:
            if f["category"] == cat and f["id"] not in used and "skipped" not in f["text"].lower():
                used.add(f["id"])
                out.append({"category": cat, "text": f["text"], "fact_ids": [f["id"]], "source": "python-template"})
                break
    # follow-up question built from the first high-missing column or the strongest pair
    fq = None
    for f in facts:
        if f["category"] == "data quality" and "high-missingness" in f["text"] and f["columns"]:
            fq = (f"Why is `{f['columns'][0]}` missing so often - is it optional, not applicable to some rows, "
                  f"or a collection gap? (see {f['id']})", f["id"])
            break
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
    # Small or simple datasets: top up with any remaining facts so the report still has n_min insights
    for f in ordered:
        if len(out) >= n_min - 1:
            break
        if f["id"] not in used:
            used.add(f["id"])
            out.append({"category": f["category"], "text": f["text"], "fact_ids": [f["id"]], "source": "python-template"})
    out.append({"category": "follow-up question", "text": fq[0], "fact_ids": [fq[1]] if fq[1] else [],
                "source": "python-template"})
    return out
