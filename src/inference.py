"""Column type and role inference.

The CSV is loaded with every column as text (dtype=str) so that *we* decide
each column's type from its values instead of trusting pandas' automatic
guess. For each column we measure what share of the non-missing values parse
as Boolean, number, or date, and assign:

  technical type : boolean | integer | float | datetime | string | mixed | empty
  probable role  : Numeric measure | Categorical attribute | Boolean field |
                   Date-like field | Identifier-like field | Free-text field |
                   Unknown or mixed type

The role is a heuristic based on structure (cardinality, text length, parse
rates) and a small list of generic name hints (e.g. "id", "zip"). It never
tries to explain what a column *means*.
"""

# IMPORTS: `re` = REGULAR EXPRESSIONS (TEXT PATTERN MATCHING), `numpy` = FAST NUMBER ARRAYS, `pandas` = DATAFRAMES/SERIES.
import re

import numpy as np
import pandas as pd

# VOCABULARY FOR BOOLEAN DETECTION: A COLUMN WHOSE ONLY TWO VALUES COME FROM ONE OF THESE SETS IS TREATED AS A YES/NO FLAG.
BOOL_TOKENS = {"true", "false", "t", "f", "yes", "no", "y", "n"}
BOOL_NUMERIC_TOKENS = {"0", "1", "0.0", "1.0"}

# GENERIC NAME FRAGMENTS THAT SUGGEST A CODE / IDENTIFIER RATHER THAN A
# MEASUREMENT. MATCHED AS WHOLE WORDS AFTER SPLITTING SNAKE_CASE/CAMELCASE.
ID_NAME_HINTS = {
    "id", "ids", "uuid", "guid", "key", "vin", "zip", "zipcode", "postal",
    "postcode", "fips", "geoid", "tract", "phone", "ssn", "pin", "parcel", "account",
}
ID_NAME_SUFFIXES = ("number", "num", "no", "code", "geoid", "district", "ward", "precinct",
                   "tract", "fips")  # E.G. "PERMIT NUMBER", "LEGISLATIVE DISTRICT" (CODES, NOT QUANTITIES)
# NAME WORDS THAT MARK A YEAR COLUMN (E.G. "DATA YEAR", "MODEL YEAR") AND A GEOGRAPHIC COORDINATE COLUMN.
YEAR_NAME_HINTS = {"year", "yr", "fy", "fiscal"}
COORD_NAME_HINTS = {"latitude", "longitude", "lat", "lon", "lng", "long", "x_coord", "y_coord"}

# TEXT PATTERNS THAT LOOK LIKE DATES. THEY ARE ONLY A FIRST SCREEN: A COLUMN MUST ALSO ACTUALLY PARSE AS DATES LATER.
# PATTERN 1: 2024-01-31 OR 2024/01/31, OPTIONALLY WITH A TIME. PATTERN 2: 01/31/2024 STYLE.
# PATTERN 3: 31-JAN-2024 STYLE. PATTERN 4: JAN 31, 2024 STYLE.
DATE_REGEXES = [
    re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?(\.\d+)?)?\s*(Z|[+-]\d{2}:?\d{2}|[AaPp][Mm])?$"),
    re.compile(r"^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}([ T]\d{1,2}:\d{2}(:\d{2})?\s*([AaPp][Mm])?)?$"),
    re.compile(r"^\d{1,2}[ -](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ -]\d{2,4}$", re.I),
    re.compile(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2},? \d{4}$", re.I),
]
# EXACT DATE FORMATS TRIED FIRST. PARSING WITH A KNOWN FORMAT IS MUCH FASTER THAN LETTING PANDAS GUESS EACH VALUE.
COMMON_DATE_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f",
    "%m/%d/%Y", "%m/%d/%Y %H:%M", "%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d", "%d-%b-%Y", "%m/%d/%y",
]


def name_tokens(col_name: str) -> list:
    """Split a column name into lower-case word tokens (snake, kebab, camel, spaces)."""
    # INSERT A SPACE BETWEEN A LOWER-CASE AND UPPER-CASE LETTER SO CAMELCASE NAMES SPLIT INTO WORDS ("MODELYEAR" -> "MODEL YEAR").
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(col_name))
    # SPLIT ON ANYTHING THAT IS NOT A LETTER OR DIGIT (SPACES, _, -, PARENTHESES), LOWER-CASE, AND DROP EMPTY PIECES.
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def has_id_name_hint(col_name: str) -> bool:
    # A NAME COUNTS AS AN IDENTIFIER HINT IF ANY WORD IS IN THE ID LIST (E.G. "ZIP CODE" CONTAINS "ZIP") ...
    toks = name_tokens(col_name)
    if any(t in ID_NAME_HINTS for t in toks):
        return True
    # ... OR IF THE LAST WORD IS A CODE-LIKE SUFFIX (E.G. "PERMIT NUMBER", "LEGISLATIVE DISTRICT").
    return bool(toks) and toks[-1] in ID_NAME_SUFFIXES


def has_coord_name_hint(col_name: str) -> bool:
    # TRUE IF ANY WORD OF THE NAME IS A COORDINATE WORD SUCH AS "LATITUDE" OR "LON".
    return any(t in COORD_NAME_HINTS for t in name_tokens(col_name))


def has_year_name_hint(col_name: str) -> bool:
    # TRUE IF ANY WORD OF THE NAME IS A YEAR WORD SUCH AS "YEAR" OR "FY".
    return any(t in YEAR_NAME_HINTS for t in name_tokens(col_name))


def _clean_numeric_text(s: pd.Series) -> pd.Series:
    """Remove thousands separators, surrounding spaces and a leading '$' before numeric parsing."""
    # STRIP SPACES, DROP A LEADING DOLLAR SIGN, AND REMOVE THOUSANDS COMMAS SO "$1,234" CAN BE READ AS 1234.
    return s.str.strip().str.replace(r"^\$", "", regex=True).str.replace(",", "", regex=False)


def parse_numeric(s: pd.Series) -> pd.Series:
    # CONVERT TEXT TO NUMBERS. ERRORS="COERCE" TURNS ANYTHING THAT IS NOT A NUMBER INTO NAN INSTEAD OF CRASHING,
    # WHICH LETS US COUNT HOW MANY VALUES FAILED TO PARSE.
    return pd.to_numeric(_clean_numeric_text(s.astype("string")), errors="coerce")


def parse_datetime(s: pd.Series):
    """Parse a text series to datetimes. Tries fast fixed formats first, then pandas' mixed parser.

    Returns (parsed_series, format_used)."""
    # TAKE A RANDOM SAMPLE OF UP TO 500 VALUES (FIXED SEED SO RESULTS ARE REPEATABLE) TO TEST FORMATS QUICKLY.
    sample = s.dropna().astype(str).str.strip()
    sample = sample.sample(min(len(sample), 500), random_state=0) if len(sample) else sample
    # TRY EACH KNOWN FORMAT ON THE SAMPLE. THE FIRST ONE THAT PARSES AT LEAST 95% OF THE SAMPLE IS USED FOR THE WHOLE COLUMN.
    for fmt in COMMON_DATE_FORMATS:
        ok = pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean() if len(sample) else 0
        if ok >= 0.95:
            return pd.to_datetime(s.astype(str).str.strip(), format=fmt, errors="coerce"), fmt
    # NO SINGLE FORMAT FIT: FALL BACK TO PANDAS' SLOWER "MIXED" PARSER, WHICH GUESSES THE FORMAT VALUE BY VALUE.
    parsed = pd.to_datetime(s.astype(str).str.strip(), format="mixed", errors="coerce")
    return parsed, "mixed"


def looks_like_dates(values: pd.Series) -> float:
    """Share of values that match a date-like text pattern."""
    # AN EMPTY COLUMN CANNOT LOOK LIKE DATES.
    if len(values) == 0:
        return 0.0
    v = values.astype(str).str.strip()
    # START WITH "NO MATCH" FOR EVERY VALUE, THEN MARK A VALUE AS A HIT IF ANY DATE PATTERN MATCHES IT.
    hits = np.zeros(len(v), dtype=bool)
    for rx in DATE_REGEXES:
        hits |= v.str.match(rx).to_numpy(dtype=bool)
    # THE MEAN OF TRUE/FALSE VALUES IS THE SHARE OF VALUES THAT LOOK LIKE DATES (0.0 TO 1.0).
    return float(hits.mean())


def infer_column(name: str, raw: pd.Series, cfg) -> dict:
    """Infer technical type and probable role for one text column.

    Returns a dict describing the decision (and why), plus the typed series
    under the key "_typed" (removed before anything is written to JSON)."""
    # COUNT ROWS, NON-MISSING VALUES AND DISTINCT VALUES. BLANK STRINGS ("" OR SPACES) COUNT AS MISSING.
    n_rows = len(raw)
    non_null = raw.dropna()
    non_null = non_null[non_null.astype(str).str.strip() != ""]
    n_valid = int(len(non_null))
    n_unique = int(non_null.nunique())
    # THE RECORD FOR THIS COLUMN. IT BECOMES ONE ROW OF `column_profile.csv` AND ONE ENTRY IN `analysis_summary.json`.
    # "UNIQUE_RATIO" = DISTINCT VALUES / NON-MISSING VALUES (1.0 MEANS EVERY VALUE IS DIFFERENT, LIKE AN ID).
    info = {
        "column": name,
        "non_missing": n_valid,
        "missing": int(n_rows - n_valid),
        "missing_pct": round(100 * (n_rows - n_valid) / n_rows, 2) if n_rows else 0.0,
        "unique": n_unique,
        "unique_ratio": round(n_unique / n_valid, 4) if n_valid else 0.0,
        "notes": [],
        "non_conforming_count": 0,
        "non_conforming_examples": [],
        "exclude_from_correlation": False,
    }

    # CASE 0 - COMPLETELY EMPTY COLUMN: NOTHING TO INFER, SO REPORT IT AS EMPTY AND STOP HERE.
    if n_valid == 0:
        info.update(technical_type="empty", role="Unknown or mixed type")
        info["notes"].append("column has no non-missing values")
        info["_typed"] = raw
        return info

    # CLEANED TEXT VERSION OF THE VALUES, PLUS (UP TO 10 OF) THE DISTINCT VALUES IN LOWER CASE FOR THE BOOLEAN TEST.
    text = non_null.astype(str).str.strip()
    lower_unique = set(text.str.lower().unique()[:10])

    # 1) BOOLEAN: AT MOST TWO DISTINCT VALUES, ALL FROM A KNOWN YES/NO VOCABULARY.
    if n_unique == 2 and (lower_unique <= BOOL_TOKENS or lower_unique <= BOOL_NUMERIC_TOKENS):
        info.update(technical_type="boolean", role="Boolean field")
        if lower_unique <= BOOL_NUMERIC_TOKENS:
            info["notes"].append("two values 0/1 - treated as a Boolean flag")
        info["_typed"] = raw.where(raw.notna(), None)
        return info

    # 2) NUMERIC
    # TRY TO READ EVERY VALUE AS A NUMBER; `num_rate` = SHARE THAT SUCCEEDED (E.G. 0.99 = 99% NUMERIC).
    num = parse_numeric(non_null)
    num_rate = float(num.notna().mean())
    # 3) DATE-LIKE TEXT (ONLY TESTED IF VALUES ARE NOT PLAIN NUMBERS)
    # ONLY LOOK FOR DATES WHEN THE COLUMN IS NOT CLEARLY NUMERIC (A SAMPLE OF 2,000 VALUES IS ENOUGH TO DECIDE).
    date_rate = 0.0
    if num_rate < cfg.type_parse_threshold:
        sample = text.sample(min(len(text), 2000), random_state=0)
        date_rate = looks_like_dates(sample)

    # AVERAGE CHARACTERS AND WORDS PER VALUE - USED LATER TO TELL FREE TEXT FROM SHORT CATEGORY LABELS.
    avg_len = float(text.str.len().mean())
    avg_words = float(text.str.split().str.len().mean())

    # CASE 2 - NUMERIC COLUMN (AT LEAST 95% OF VALUES ARE NUMBERS).
    # CONVERT THE FULL COLUMN, AND REMEMBER UP TO 5 EXAMPLES OF VALUES THAT ARE NOT NUMBERS (E.G. ZIP+4 CODES "60626-1864").
    if num_rate >= cfg.type_parse_threshold:
        typed_full = parse_numeric(raw)
        bad = non_null[num.isna()]
        info["non_conforming_count"] = int(len(bad))
        info["non_conforming_examples"] = [str(x) for x in bad.unique()[:5]]
        # IF EVERY VALUE IS A WHOLE NUMBER THE TECHNICAL TYPE IS "INTEGER", OTHERWISE "FLOAT".
        valid = typed_full.dropna()
        is_int = bool(np.all(np.isclose(valid, np.round(valid))))
        info["technical_type"] = "integer" if is_int else "float"
        info["_typed"] = typed_full

        # ROLE RULE A: WHOLE NUMBERS BETWEEN 1800 AND 2100 IN A COLUMN NAMED LIKE A YEAR -> DATE-LIKE (A YEAR, NOT A QUANTITY).
        if is_int and has_year_name_hint(name) and valid.between(1800, 2100).all():
            info["role"] = "Date-like field"
            info["notes"].append("integer years (name contains a year word and values are 1800-2100)")
        # ROLE RULE B: WHOLE NUMBERS IN A COLUMN NAMED LIKE A CODE (ZIP, ID, DISTRICT, ...) -> IDENTIFIER, NOT A MEASUREMENT.
        # AVERAGING OR CORRELATING ZIP CODES WOULD BE MEANINGLESS, SO THESE ARE KEPT OUT OF THE STATISTICS.
        elif is_int and has_id_name_hint(name):
            info["role"] = "Identifier-like field"
            info["notes"].append("numeric code: column name suggests an identifier/code, so it is not treated as a measurement")
        # ROLE RULE C: ALMOST-ALL-DISTINCT WHOLE NUMBERS THAT FILL THEIR RANGE LIKE A COUNTER (1, 2, 3, ...) -> ROW NUMBER / ID.
        elif is_int and info["unique_ratio"] >= cfg.id_unique_ratio and n_valid >= 20 and _is_sequential_like(valid):
            info["role"] = "Identifier-like field"
            info["notes"].append("integers that are (nearly) all distinct and evenly spread - looks like a row number or ID")
        # OTHERWISE IT IS A REAL NUMERIC MEASURE. TWO NOTES MAY BE ADDED: FEW DISTINCT VALUES (MAYBE A RATING OR CODE),
        # AND COORDINATES, WHICH ARE SUMMARIZED BUT KEPT OUT OF CORRELATIONS AND HEADLINE PLOTS.
        else:
            info["role"] = "Numeric measure"
            if n_unique <= 10:
                info["notes"].append(f"only {n_unique} distinct values - may be a coded category or rating")
            if has_coord_name_hint(name) and valid.between(-180, 180).all():
                info["exclude_from_correlation"] = True
                info["notes"].append("looks like a geographic coordinate - summarized, but excluded from "
                                     "correlations and headline plots")
        # RECORD HOW MANY NON-NUMERIC VALUES WERE LEFT OUT OF THE NUMERIC STATISTICS (THEY ARE REPORTED, NOT SILENTLY DROPPED).
        if info["non_conforming_count"]:
            info["notes"].append(
                f"{info['non_conforming_count']} non-numeric value(s) excluded from numeric statistics")
        return info

    # CASE 3 - DATE COLUMN: AT LEAST 90% OF SAMPLED VALUES LOOK LIKE DATES AND AT LEAST 95% ACTUALLY PARSE.
    if date_rate >= 0.90:
        parsed, fmt = parse_datetime(raw.where(raw.notna()))
        rate = float(parsed.loc[non_null.index].notna().mean())
        if rate >= cfg.type_parse_threshold:
            info.update(technical_type="datetime", role="Date-like field")
            info["notes"].append(f"parsed as dates (format: {fmt})")
            bad = non_null[parsed.loc[non_null.index].isna()]
            info["non_conforming_count"] = int(len(bad))
            info["non_conforming_examples"] = [str(x) for x in bad.unique()[:5]]
            info["_typed"] = parsed
            return info

    # CASE 4 - MIXED TYPES: BETWEEN 50% AND 95% OF VALUES ARE NUMBERS AND THE REST ARE TEXT. FLAGGED AS INCONSISTENT.
    if cfg.mixed_type_floor <= num_rate < cfg.type_parse_threshold:
        bad = non_null[num.isna()]
        info.update(technical_type="mixed", role="Unknown or mixed type")
        info["non_conforming_count"] = int(len(bad))
        info["non_conforming_examples"] = [str(x) for x in bad.unique()[:5]]
        info["notes"].append(
            f"{num_rate:.1%} of values are numeric and the rest are text - inconsistent types")
        info["_typed"] = raw
        return info

    # REMAINING: TEXT COLUMNS.
    # CASE 5 - TEXT COLUMN. DECIDE BETWEEN FREE TEXT, IDENTIFIER AND CATEGORY.
    info["technical_type"] = "string"
    info["_typed"] = raw
    # LONG TEXT ONLY COUNTS AS FREE TEXT IF IT IS ALSO VARIED; A FEW LONG LABELS THAT REPEAT
    # THOUSANDS OF TIMES ARE CATEGORIES.
    # FREE TEXT: LONG AND VARIED VALUES (E.G. COMMENTS OR DESCRIPTIONS).
    if (avg_len >= cfg.free_text_avg_len and info["unique_ratio"] > 0.2) or (avg_words >= 5 and info["unique_ratio"] > 0.5):
        info["role"] = "Free-text field"
        info["notes"].append(f"average length {avg_len:.0f} characters, {avg_words:.1f} words")
    # IDENTIFIER: NEARLY EVERY VALUE IS DIFFERENT (E.G. A ROW ID), OR THE NAME SAYS ID/CODE AND MOST VALUES ARE DIFFERENT.
    elif (info["unique_ratio"] >= cfg.id_unique_ratio and n_valid >= 20) or (
            has_id_name_hint(name) and info["unique_ratio"] > 0.5):
        info["role"] = "Identifier-like field"
        info["notes"].append("text values that are (nearly) all distinct or named like an identifier")
    # CATEGORICAL: EVERYTHING ELSE - SHORT LABELS THAT REPEAT (E.G. COUNTY, MAKE, PROPERTY TYPE).
    else:
        info["role"] = "Categorical attribute"
        if has_id_name_hint(name):
            info["notes"].append("name suggests a code; treated as categorical because values repeat")
    return info


def _is_sequential_like(valid: pd.Series) -> bool:
    """True if integer values look like a counter (almost all distinct and fill most of their range)."""
    # IF THE VALUES FILL AT LEAST HALF OF THE RANGE BETWEEN THEIR MIN AND MAX, THEY BEHAVE LIKE A COUNTER.
    # (MEASUREMENTS SUCH AS DOLLARS OR SQUARE FEET ARE SPREAD THINLY OVER A HUGE RANGE, SO THEY FAIL THIS TEST.)
    lo, hi = float(valid.min()), float(valid.max())
    span = hi - lo + 1
    return span > 0 and len(valid) / span >= 0.5


def infer_all(df_raw: pd.DataFrame, cfg):
    """Infer every column. Returns (list_of_column_info, typed_dataframe)."""
    # RUN `infer_column()` ON EVERY COLUMN. KEEP THE DESCRIPTION (`infos`) AND THE CONVERTED VALUES (`typed`) SEPARATELY:
    # THE DESCRIPTIONS GO INTO THE REPORT/JSON, THE CONVERTED VALUES ARE USED FOR THE STATISTICS AND PLOTS.
    infos, typed = [], {}
    for col in df_raw.columns:
        info = infer_column(col, df_raw[col], cfg)
        typed[col] = info.pop("_typed")
        infos.append(info)
    return infos, pd.DataFrame(typed, index=df_raw.index)
