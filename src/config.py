"""Default, documented configuration options for the profiler.

Changing a dataset should only require changing the CSV path. Everything in
this file is a *general* threshold or option - none of it refers to a specific
dataset or column name.
"""

from dataclasses import dataclass, field


@dataclass
class ProfilerConfig:
    # --- Data quality thresholds -------------------------------------------------
    # A column is flagged as "high missingness" when more than this fraction of
    # its values are missing. 30% is a common rule of thumb: above it, most
    # analyses of that column rest on well under three quarters of the rows.
    high_missing_threshold: float = 0.30

    # A column whose (distinct values / non-missing values) ratio is at least
    # this value is treated as identifier-like / extremely high cardinality.
    id_unique_ratio: float = 0.95

    # Categorical columns with more distinct values than this are flagged as
    # high-cardinality (their frequency tables are truncated to top_k anyway).
    high_cardinality_count: int = 50

    # Numeric columns where more than this share of values are exactly 0 get a
    # note: zeros may be real, or may be a placeholder for "not recorded".
    zero_share_warning: float = 0.30

    # --- Type inference ------------------------------------------------------------
    # Share of non-missing values that must parse as a type for the column to
    # be assigned that type. Below this (but above mixed_type_floor) the column
    # is reported as mixed / inconsistent.
    type_parse_threshold: float = 0.95
    mixed_type_floor: float = 0.50

    # Columns whose average text length is above this are treated as free text.
    free_text_avg_len: int = 50

    # --- Output limits ---------------------------------------------------------------
    top_k: int = 10              # categories shown per categorical column
    max_plots: int = 8           # hard cap on generated plots
    max_corr_columns: int = 15   # max numeric columns in the correlation heatmap
    scatter_sample: int = 5000   # max points drawn in a scatterplot (random, seeded)
    random_seed: int = 42

    # Optional extension: extra strings to treat as missing when loading
    # (e.g. ["-999", "unknown", "not reported"]). Empty by default so the
    # system never changes the data unless the user asks it to.
    missing_codes: list = field(default_factory=list)

    # --- LLM ---------------------------------------------------------------------------
    llm_model: str = "qwen2.5:7b"
    ollama_host: str = "http://localhost:11434"
    llm_timeout_seconds: int = 600
    llm_temperature: float = 0.2
    min_insights: int = 5
    max_insights: int = 8
