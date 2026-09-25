"""Default, documented configuration options for the profiler.

Changing a dataset should only require changing the CSV path. Everything in
this file is a *general* threshold or option - none of it refers to a specific
dataset or column name.
"""

# IMPORTS: `dataclass` BUILDS A SIMPLE SETTINGS CLASS FROM THE FIELD LIST BELOW (IT WRITES THE `__init__` METHOD FOR US).
# `field` IS NEEDED FOR THE ONE SETTING WHOSE DEFAULT IS A LIST (A LIST DEFAULT MUST BE CREATED FRESH FOR EACH CONFIG).
from dataclasses import dataclass, field


# THE CONFIGURATION CLASS. EVERY OTHER MODULE READS ITS THRESHOLDS FROM AN INSTANCE OF THIS CLASS (USUALLY CALLED `cfg`).
# EACH LINE BELOW IS "NAME: TYPE = DEFAULT VALUE". ANY VALUE CAN BE OVERRIDDEN WHEN THE CONFIG IS CREATED,
# E.G. `ProfilerConfig(high_missing_threshold=0.2)`, WHICH IS WHAT THE COMMAND-LINE OPTIONS IN `profiler.py` DO.
@dataclass
class ProfilerConfig:
    # --- DATA QUALITY THRESHOLDS -------------------------------------------------
    # A COLUMN IS FLAGGED AS "HIGH MISSINGNESS" WHEN MORE THAN THIS FRACTION OF
    # ITS VALUES ARE MISSING. 30% IS A COMMON RULE OF THUMB: ABOVE IT, MOST
    # ANALYSES OF THAT COLUMN REST ON WELL UNDER THREE QUARTERS OF THE ROWS.
    high_missing_threshold: float = 0.30

    # A COLUMN WHOSE (DISTINCT VALUES / NON-MISSING VALUES) RATIO IS AT LEAST
    # THIS VALUE IS TREATED AS IDENTIFIER-LIKE / EXTREMELY HIGH CARDINALITY.
    id_unique_ratio: float = 0.95

    # CATEGORICAL COLUMNS WITH MORE DISTINCT VALUES THAN THIS ARE FLAGGED AS
    # HIGH-CARDINALITY (THEIR FREQUENCY TABLES ARE TRUNCATED TO TOP_K ANYWAY).
    high_cardinality_count: int = 50

    # NUMERIC COLUMNS WHERE MORE THAN THIS SHARE OF VALUES ARE EXACTLY 0 GET A
    # NOTE: ZEROS MAY BE REAL, OR MAY BE A PLACEHOLDER FOR "NOT RECORDED".
    zero_share_warning: float = 0.30

    # --- TYPE INFERENCE ------------------------------------------------------------
    # SHARE OF NON-MISSING VALUES THAT MUST PARSE AS A TYPE FOR THE COLUMN TO
    # BE ASSIGNED THAT TYPE. BELOW THIS (BUT ABOVE MIXED_TYPE_FLOOR) THE COLUMN
    # IS REPORTED AS MIXED / INCONSISTENT.
    type_parse_threshold: float = 0.95
    mixed_type_floor: float = 0.50

    # COLUMNS WHOSE AVERAGE TEXT LENGTH IS ABOVE THIS ARE TREATED AS FREE TEXT.
    free_text_avg_len: int = 50

    # --- OUTPUT LIMITS ---------------------------------------------------------------
    top_k: int = 10              # CATEGORIES SHOWN PER CATEGORICAL COLUMN
    max_plots: int = 8           # HARD CAP ON GENERATED PLOTS
    max_corr_columns: int = 15   # MAX NUMERIC COLUMNS IN THE CORRELATION HEATMAP
    scatter_sample: int = 5000   # MAX POINTS DRAWN IN A SCATTERPLOT (RANDOM, SEEDED)
    random_seed: int = 42

    # OPTIONAL EXTENSION: EXTRA STRINGS TO TREAT AS MISSING WHEN LOADING
    # (E.G. ["-999", "UNKNOWN", "NOT REPORTED"]). EMPTY BY DEFAULT SO THE
    # SYSTEM NEVER CHANGES THE DATA UNLESS THE USER ASKS IT TO.
    missing_codes: list = field(default_factory=list)

    # --- LLM ---------------------------------------------------------------------------
    # WHICH OLLAMA MODEL WRITES THE NARRATIVE INSIGHTS (`qwen2.5:3b` WAS TRIED FIRST; SEE THE GENAI LOG FOR WHY 7B IS THE DEFAULT).
    llm_model: str = "qwen2.5:7b"
    # ADDRESS OF THE LOCAL OLLAMA SERVER. 11434 IS OLLAMA'S STANDARD PORT ON YOUR OWN MACHINE.
    ollama_host: str = "http://localhost:11434"
    # HOW LONG TO WAIT FOR THE MODEL BEFORE GIVING UP (10 MINUTES; CPU-ONLY LAPTOPS CAN BE SLOW).
    llm_timeout_seconds: int = 600
    # LOW TEMPERATURE = LESS RANDOM WORDING, SO THE MODEL STAYS CLOSER TO THE FACTS IT IS GIVEN.
    llm_temperature: float = 0.2
    # THE REPORT MUST CONTAIN BETWEEN 5 AND 8 INSIGHTS (ASSIGNMENT REQUIREMENT).
    min_insights: int = 5
    max_insights: int = 8
