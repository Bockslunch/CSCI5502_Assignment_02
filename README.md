# Automated CSV Profiler and Verified Insight Generator

CSCI 5502 · Assignment 2 — *From CSV to Evidence* · Individual submission

```
CSV file → Python analysis → verified summary (JSON) → LLM explanation → Python claim check → generated report
```

**Python computes the evidence. The language model only explains it.** Every number that appears in an
AI-written insight is checked by Python against the verified summary before it is allowed into the report.

## 1. Purpose

Given any reasonably well-formed, single-table CSV, the system:

1. Loads every column as text and infers each column's **technical type** (boolean / integer / float /
   datetime / string / mixed / empty) and **probable role** (numeric measure, categorical attribute,
   Boolean, date-like, identifier-like, free text, unknown/mixed).
2. Runs **data-quality checks**: duplicate rows, constant and empty columns, high missingness
   (default threshold 30%), mixed/inconsistent types, identifier-like / high-cardinality columns,
   placeholder values (`unknown`, `-999`, …), columns that are mostly zeros, and heuristic
   **sensitive-field warnings** (name and value patterns).
3. Computes **descriptive statistics**: for numeric measures — count, missing, min, max, mean, median,
   mode (when meaningful), sample std, Q1, Q3, IQR, 1.5×IQR outlier count/%; for categorical/Boolean —
   distinct count, most frequent value(s), top-10 frequencies and percentages; for dates — range and
   counts per period.
4. Analyzes **relationships**: Pearson (and Spearman) correlation matrix of numeric measures with
   identifiers, codes, years, flags and coordinates excluded; strongest positive/negative pairs;
   near-duplicate pairs (|r| ≥ 0.95); a numeric-by-category comparison.
5. Chooses **visualizations adaptively** from the column roles (missing-value bars, histogram, boxplot,
   category frequencies, correlation heatmap, scatterplot, numeric-by-category boxplot, time plot),
   up to 8 per dataset, and records which plot types were skipped and why.
6. Turns the results into numbered **verified facts** (`F1`, `F2`, …), sends only those facts to a
   **local LLM (Ollama)**, and **verifies** every returned insight: each number must match a cited
   fact (rounding tolerance), cited fact IDs must exist, causal wording is rejected, and quantitative
   claims must name a real column. Rejected insights are shown in a claim-verification table and left
   out of the insight list. If fewer than 5 insights survive, Python-generated template insights fill
   the gap so the report always has 5–8 evidence-based insights.
7. Writes `report.md`, `column_profile.csv`, `analysis_summary.json`, `llm_prompt.txt`,
   `llm_response.txt` and `plots/` into a separate folder per dataset.

Nothing is deleted, imputed or corrected: duplicates, missing values and outliers are **flagged only**.
No dataset-specific column names appear anywhere in the analysis code.

## 2. Environment

| Item | Version used |
|---|---|
| Python | 3.14.7 (developed on 3.11; 3.10+ should work) |
| pandas | 3.0.6 (≥ 2.0 required for `format="mixed"` date parsing) |
| NumPy | 2.5.3 |
| Matplotlib | 3.11.2 |
| Ollama | local server, model `qwen2.5:7b` |

The LLM call uses Python's standard `urllib` against Ollama's HTTP API, so no extra Python package or API
key is needed. `pytest` is optional (only for the automated tests).

## 3. Installation (Windows / Anaconda Prompt shown; macOS/Linux identical except `activate`)

```bash
cd automated_csv_profiler
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# LLM component (optional)
#   1. install Ollama from https://ollama.com and make sure it is running
ollama pull qwen2.5:7b
```

## 4. Running the program

The single entry point is `generate_profile()` in `src/profiler.py`:

```python
from profiler import generate_profile
generate_profile(csv_path, output_dir, use_llm=True)
```

From the command line:

```bash
# Dataset A (development dataset)
python src/profiler.py data/chicago_energy_benchmarking.csv --out output/dataset_a

# Dataset B (new dataset) - same program, only the path changes
python src/profiler.py data/wa_electric_vehicle_population.csv --out output/dataset_b

# or both in one go (add --model <name> to use a different Ollama model)
python src/run_all.py
```

### Selecting a CSV file

Pass any CSV path as the first argument. If `--out` is omitted the output goes to `output/<file name>/`.

### Enabling / disabling the LLM

* LLM on (default): Ollama must be running at `http://localhost:11434`.
* LLM off: add `--no-llm` (or `use_llm=False`).
* If the LLM is on but Ollama is not reachable, the program still produces the full deterministic profile
  and the report states that AI-generated narrative insights were skipped because the model was unavailable.

### Documented configuration options

| Option | Default | Meaning |
|---|---|---|
| `--model` | `qwen2.5:7b` | Ollama model name |
| `--host` | `http://localhost:11434` | Ollama server URL |
| `--missing-threshold` | `0.30` | columns missing more than this share are flagged. 30% is a common rule of thumb: above it, analyses of that column use well under three quarters of the rows |
| `--missing-codes` | none | extra strings to treat as missing on load, e.g. `--missing-codes -999 unknown` (off by default so data are never changed silently) |
| `--max-plots` | `8` | cap on plots per dataset |

Other thresholds (identifier uniqueness ratio 0.95, type-parse threshold 95%, top-k = 10 categories, etc.)
are in `src/config.py` with comments explaining each.

### Tests (optional extension)

```bash
pip install pytest
python -m pytest -q tests                      # unit/edge-case tests incl. LLM-verifier tests
python tests/independent_check.py data/chicago_energy_benchmarking.csv output/dataset_a
python tests/independent_check.py data/wa_electric_vehicle_population.csv output/dataset_b
```

`independent_check.py` recomputes several statistics with a different code path (pandas' own type
guessing + NumPy percentiles) and compares them with `analysis_summary.json`.

## 5. Repository layout

```
automated_csv_profiler/
    README.md  requirements.txt
    src/        profiler.py (entry point + CLI), config.py, inference.py, analysis.py,
                plots.py, llm.py, report.py, run_all.py
    tests/      test_profiler.py, independent_check.py
    data/       README.md (+ the CSVs; see data/README.md)
    output/     dataset_a/, dataset_b/   (generated by the program)
    video/      video_link.txt
    logs/       genai_log.md
```

## 6. Model used

Local **Ollama `qwen2.5:7b`** (7-billion-parameter Qwen 2.5 instruct model), temperature 0.2, fixed seed,
JSON output mode, run on a laptop CPU. `qwen2.5:3b` was tried first: it often left out fact IDs, copied numbers
into the wrong context (e.g. called zeros "missing") and imitated the prompt's example, so most of its insights
were rejected by the verifier. The 7B model cited facts and added reasonable interpretation, so it became the
default. Any Ollama model can be selected with `--model`.

## 7. Datasets

| | Dataset A (development) | Dataset B (new) |
|---|---|---|
| Organization | City of Chicago (Chicago Data Portal) | Washington State Department of Licensing (data.wa.gov) |
| Title | Chicago Energy Benchmarking | Electric Vehicle Population Data |
| Source | https://data.cityofchicago.org/d/xq83-jr8c | https://data.wa.gov/d/f6w7-q2d2 |
| Accessed | 2026-09-23 | 2026-09-23 |
| Shape | 28,329 rows × 30 columns | 299,705 rows × 16 columns |
| Character | numeric-heavy (energy, emissions, floor area), heavy missingness in several columns, year column | categorical-heavy (county, city, make, model, utility), almost no missing values, one real numeric measure |

Both are public, non-sensitive open-government datasets. See `data/README.md` for download steps.

## 8. Known limitations

* Types and roles are **heuristics** (parse rates, cardinality, text length and a short list of generic
  name hints such as `id`, `zip`, `code`, `year`, `latitude`). Unusual columns can be mis-classified —
  e.g. a numeric code without a telling name will be treated as a numeric measure.
* The system does **not** know what columns mean, their units or valid ranges, and says so.
* The 1.5×IQR rule flags many points in heavily skewed data (e.g. 9.9% of `Gross Floor Area` values);
  flagged values are candidates for review, not errors.
* Pearson correlation measures only linear association and is sensitive to outliers (Spearman is shown
  alongside); correlation is never interpreted as causation.
* Sensitive-field detection is a name/pattern heuristic; **no warning does not mean the data are safe**.
* Placeholder values are detected from a fixed token list and are not converted unless requested.
* The claim checker verifies **numbers, fact IDs, column names, causal wording, correlation-strength words**
  (strong/weak/perfect vs. the actual r) and, when one column is named, that the number next to a statistic word
  (missing, distinct values, mean, median, std, min, max, outliers) is that column's actual value. Other qualitative wording (e.g. "most", "relatively low") and an
  insight's category label still need human reading. Small local models often omit fact IDs; the checker then
  matches the insight to facts about the same column and labels those citations "matched by Python".
* Only single-table CSVs that fit in memory are supported (no Excel/JSON/streaming/nested data).
