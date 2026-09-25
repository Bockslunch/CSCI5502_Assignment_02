"""Adaptive visualizations.

Plots are chosen from the column roles found by inference. Each candidate
plot has a priority; the top `max_plots` are drawn. Plot types that the data
cannot support are recorded with a reason instead of being forced.
"""

# IMPORTS. `textwrap` WRAPS LONG TITLES. MATPLOTLIB IS SWITCHED TO THE "AGG" BACKEND BEFORE PYPLOT IS IMPORTED,
# WHICH DRAWS STRAIGHT TO PNG FILES WITHOUT OPENING ANY WINDOWS.
import textwrap

import matplotlib

matplotlib.use("Agg")  # FILE OUTPUT ONLY, NO DISPLAY NEEDED

# MATPLOTLIB 3.10 REPLACED BOXPLOT(VERT=FALSE) WITH ORIENTATION="HORIZONTAL" (VERT IS REMOVED IN 3.13).
# READ THE INSTALLED MATPLOTLIB VERSION AS (MAJOR, MINOR) AND PICK THE RIGHT KEYWORD FOR HORIZONTAL BOXPLOTS.
_MPL = tuple(int(p) for p in matplotlib.__version__.split(".")[:2])
HORIZONTAL = {"orientation": "horizontal"} if _MPL >= (3, 10) else {"vert": False}
# PLOTTING, NUMBER AND DATAFRAME LIBRARIES, PLUS THE SHARED NUMBER FORMATTER `fmt` FROM `llm.py`.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from llm import fmt  # noqa: E402

# PLAIN, COMMA-SEPARATED TICK LABELS INSTEAD OF SCIENTIFIC OFFSETS LIKE "1E6"
# AXIS TICK FORMATTER: 12,000 INSTEAD OF 1.2E4, AND SMALL NUMBERS WITHOUT TRAILING ZEROS.
COMMA = FuncFormatter(lambda x, _: f"{x:,.0f}" if abs(x) >= 1000 else f"{x:g}")

# COLOR PALETTE: OFF-WHITE BACKGROUND, DARK TEXT, GREY GRID, BLUE FOR DATA, ORANGE TO HIGHLIGHT OUTLIERS.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

# GLOBAL CHART STYLE APPLIED TO EVERY PLOT: COLORS, FONT SIZES, LIGHT GRID DRAWN BEHIND THE DATA, NO TOP/RIGHT BORDERS.
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "font.size": 10,
})


def _short(label, width=32):
    # SHORTEN LONG LABELS WITH "…" SO CATEGORY NAMES AND COLUMN NAMES FIT ON THE AXIS.
    label = str(label)
    return label if len(label) <= width else label[: width - 1] + "…"


def _wrap(title, width=60):
    # SPLIT A LONG TITLE ACROSS SEVERAL LINES (60 CHARACTERS EACH) SO IT IS NOT CUT OFF.
    return "\n".join(textwrap.wrap(title, width))


def _save(fig, path):
    # SAVE A FINISHED FIGURE AS A PNG (120 DPI) AND CLOSE IT TO FREE MEMORY.
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---- INDIVIDUAL PLOT FUNCTIONS ---------------------------------------------------------
def plot_missing(infos, path):
    # MISSING-VALUE BAR CHART: ONLY COLUMNS WITH ANY MISSING VALUES, MOST-MISSING FIRST, AT MOST 25 BARS.
    miss = sorted([(i["column"], i["missing_pct"]) for i in infos if i["missing"] > 0], key=lambda t: -t[1])[:25]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(miss) + 1.2)))
    # REVERSE THE LISTS SO THE MOST-MISSING COLUMN APPEARS AT THE TOP OF THE HORIZONTAL BAR CHART.
    names = [_short(c) for c, _ in miss][::-1]
    vals = [p for _, p in miss][::-1]
    ax.barh(range(len(vals)), vals, color=BLUE, height=0.7)  # NUMERIC POSITIONS: TRUNCATED NAMES MAY REPEAT
    ax.set_yticks(range(len(vals)), names)
    # AXIS LABELS, A FIXED 0-100% SCALE, AND A PERCENTAGE PRINTED AT THE END OF EACH BAR.
    ax.set_xlabel("Missing values (% of rows)")
    ax.set_ylabel("Column")
    ax.set_xlim(0, 100)
    for y, v in enumerate(vals):
        ax.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_title(_wrap("Missing values by column" + (" (25 most-missing shown)" if len(miss) == 25 else "")))
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_hist(s, col, stats, path):
    # HISTOGRAM: USE A LOG-SCALED X AXIS WHEN THE DATA ARE VERY RIGHT-SKEWED (SKEWNESS > 3) AND ALL POSITIVE,
    # OTHERWISE MOST VALUES WOULD BE SQUEEZED INTO ONE BAR. THE AXIS LABEL SAYS "(LOG SCALE)" WHEN THIS HAPPENS.
    v = pd.to_numeric(s, errors="coerce").dropna()
    log = stats.get("skewness") is not None and stats["skewness"] > 3 and v.min() > 0
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # LOG CASE: 40 BINS EVENLY SPACED ON A LOG SCALE BETWEEN THE MIN AND MAX VALUE.
    if log:
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 40)
        ax.hist(v, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(f"{col} (log scale)")
    # NORMAL CASE: 40 EQUAL-WIDTH BINS.
    else:
        ax.hist(v, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
        ax.set_xlabel(col)
        ax.xaxis.set_major_formatter(COMMA)
    # A DASHED VERTICAL LINE MARKS THE MEDIAN; THE TITLE SHOWS HOW MANY VALUES WERE PLOTTED.
    ax.yaxis.set_major_formatter(COMMA)
    ax.axvline(stats["median"], color=INK, linestyle="--", linewidth=1.2, label=f"median = {fmt(stats['median'])}")
    ax.set_ylabel("Number of rows")
    ax.set_title(_wrap(f"Distribution of {col} (n = {len(v):,})"))
    ax.legend(frameon=False)
    _save(fig, path)
    return log


def plot_box(s, col, stats, path):
    # BOXPLOT: THE BOX IS Q1 TO Q3, THE BLACK LINE IS THE MEDIAN, THE WHISKERS REACH TO 1.5 X IQR, AND THE ORANGE
    # POINTS BEYOND THE WHISKERS ARE THE POTENTIAL OUTLIERS COUNTED IN `numeric_stats()`.
    v = pd.to_numeric(s, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.boxplot(v, **HORIZONTAL, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#d6e6f8", edgecolor=BLUE), medianprops=dict(color=INK, linewidth=1.5),
               whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
               flierprops=dict(marker="o", markersize=3, markerfacecolor=ORANGE, markeredgecolor="none", alpha=0.5))
    ax.set_yticks([1], [_short(col, 20)])
    # SAME LOG-AXIS RULE AS THE HISTOGRAM. ONLY THE DISPLAY CHANGES; THE OUTLIER COUNT IN THE TITLE USES THE REAL VALUES.
    log = stats.get("skewness") is not None and stats["skewness"] > 3 and v.min() > 0
    if log:  # FENCES ARE STILL COMPUTED ON THE ORIGINAL SCALE; LOG ONLY SPREADS THE DISPLAY
        ax.set_xscale("log")
    else:
        ax.xaxis.set_major_formatter(COMMA)
    ax.set_xlabel(f"{_short(col, 40)}{' (log scale)' if log else ''}\nwhiskers = 1.5 × IQR; orange = potential outliers")
    ax.set_ylabel("Column")
    ax.set_title(_wrap(f"Boxplot of {col}: {stats['outlier_count']:,} potential outliers ({stats['outlier_pct']}%)"))
    _save(fig, path)


def plot_categorical(cat_stats, col, path):
    # CATEGORY FREQUENCY BAR CHART: THE TOP-K CATEGORIES (ALREADY COUNTED IN `categorical_stats()`), MOST FREQUENT AT TOP.
    top = cat_stats["top_values"]
    labels = [_short(t["value"]) for t in top][::-1]
    counts = [t["count"] for t in top][::-1]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.38 * len(top) + 1.4)))
    # BARS ARE PLACED AT NUMBERED POSITIONS (NOT BY LABEL) BECAUSE TWO LONG LABELS CAN BE IDENTICAL AFTER SHORTENING,
    # WHICH WOULD OTHERWISE MERGE THEM INTO ONE BAR. EACH BAR GETS ITS PERCENTAGE PRINTED AT THE END.
    ax.barh(range(len(counts)), counts, color=BLUE, height=0.7)
    ax.set_yticks(range(len(counts)), labels)
    for y, (c, t) in enumerate(zip(counts, top[::-1])):
        ax.text(c, y, f" {t['pct_of_non_missing']:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("Number of rows")
    ax.xaxis.set_major_formatter(COMMA)
    ax.set_ylabel(_short(col, 40))
    # THE TITLE SAYS WHETHER ALL CATEGORIES ARE SHOWN OR ONLY THE TOP K OF N.
    extra = f"top {len(top)} of {cat_stats['unique']:,} categories" if cat_stats["unique"] > len(top) else f"all {len(top)} categories"
    ax.set_title(_wrap(f"Most frequent values of {col} ({extra})"))
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    _save(fig, path)


def plot_heatmap(corr, path):
    # CORRELATION HEATMAP: RED = POSITIVE, BLUE = NEGATIVE, WHITE = NONE. THE FIGURE GROWS WITH THE NUMBER OF COLUMNS.
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(7, min(1.5 + 0.7 * n, 13)), max(5.5, min(1.5 + 0.6 * n, 11))))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [_short(c, 24) for c in corr.columns]
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.grid(False)
    # PRINT THE R VALUE IN EACH CELL WHEN THE MATRIX IS SMALL ENOUGH TO READ (WHITE TEXT ON DARK CELLS).
    if n <= 12:
        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(val) > 0.6 else INK)
    # COLOR LEGEND, AXIS LABELS AND TITLE.
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)
    ax.set_xlabel("Numeric column")
    ax.set_ylabel("Numeric column")
    ax.set_title("Correlation matrix of numeric measures (Pearson r)")
    _save(fig, path)


def plot_scatter(typed, a, b, pair, cfg, path):
    # SCATTERPLOT: ONLY ROWS WHERE BOTH COLUMNS HAVE VALUES. AT MOST 5,000 RANDOMLY CHOSEN POINTS (FIXED SEED) ARE DRAWN
    # SO LARGE DATASETS STAY READABLE; SEMI-TRANSPARENT DOTS SHOW WHERE POINTS OVERLAP.
    d = typed[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    shown = d.sample(min(len(d), cfg.scatter_sample), random_state=cfg.random_seed)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(shown[a], shown[b], s=10, alpha=0.35, color=BLUE, edgecolors="none")
    # LOG AXES WHEN A COLUMN IS STRONGLY RIGHT-SKEWED AND POSITIVE, SO THE BULK IS VISIBLE
    for axis, col in (("x", a), ("y", b)):
        v = d[col]
        if v.min() > 0 and v.skew() > 3:
            getattr(ax, f"set_{axis}scale")("log")
    # LINEAR AXES GET COMMA-FORMATTED TICKS; LABELS SAY WHEN AN AXIS IS LOG-SCALED; THE TITLE SHOWS R AND N.
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() != "log":
            axis.set_major_formatter(COMMA)
    ax.set_xlabel(a + (" (log scale)" if ax.get_xscale() == "log" else ""))
    ax.set_ylabel(b + (" (log scale)" if ax.get_yscale() == "log" else ""))
    note = f"; {len(shown):,} of {len(d):,} points shown" if len(shown) < len(d) else ""
    ax.set_title(_wrap(f"{b} vs {a} (Pearson r = {pair['pearson_r']:.2f}, n = {pair['n_pairs']:,}{note})"))
    _save(fig, path)


def plot_group_box(typed, group_col, measure, path, top_k=8):
    # NUMERIC-BY-CATEGORY BOXPLOT: ONE BOX PER GROUP FOR THE 8 MOST COMMON GROUPS. OUTLIER POINTS ARE HIDDEN HERE
    # (THEY ARE ALREADY SHOWN IN THE SINGLE-COLUMN BOXPLOT) SO THE BOXES STAY COMPARABLE.
    d = typed[[group_col, measure]].copy()
    d[measure] = pd.to_numeric(d[measure], errors="coerce")
    d = d.dropna()
    levels = d[group_col].value_counts().head(top_k).index.tolist()[::-1]  # MOST FREQUENT DRAWN AT TOP
    data = [d.loc[d[group_col] == lv, measure].values for lv in levels]
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.45 * len(levels) + 1.5)))
    ax.boxplot(data, **HORIZONTAL, widths=0.6, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor="#d6e6f8", edgecolor=BLUE), medianprops=dict(color=INK, linewidth=1.5),
               whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
    ax.set_yticks(range(1, len(levels) + 1), [_short(lv) for lv in levels])
    # LOG X AXIS FOR STRONGLY SKEWED POSITIVE DATA, OTHERWISE COMMA-FORMATTED TICKS.
    v = d[measure]
    if v.min() > 0 and v.skew() > 3:
        ax.set_xscale("log")
    else:
        ax.xaxis.set_major_formatter(COMMA)
    ax.set_xlabel(measure + (" (log scale)" if ax.get_xscale() == "log" else "") + " — outlier points hidden")
    ax.set_ylabel(_short(group_col, 40))
    ax.set_title(_wrap(f"{measure} by {group_col} (top {len(levels)} groups by count)"))
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_time(date_col, dstats, path):
    # TIME PLOT: NUMBER OF ROWS IN EACH PERIOD (YEAR, MONTH OR DAY), AS COUNTED IN `date_stats()`.
    per = dstats["counts_by_period"]
    x = list(per.keys())
    y = list(per.values())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if all(k.lstrip("-").isdigit() for k in x) and len(x) <= 80:
        # INTEGER PERIODS (YEARS): NUMERIC X AXIS SO GAPS BETWEEN YEARS STAY VISIBLE
        xs = [int(k) for k in x]
        ax.bar(xs, y, color=BLUE, width=0.75)
        ax.xaxis.get_major_locator().set_params(integer=True)
    # UP TO 40 PERIODS: ONE BAR PER PERIOD, SHOWING AT MOST ABOUT 20 TICK LABELS.
    elif len(x) <= 40:
        ax.bar(x, y, color=BLUE, width=0.75)
        step = max(1, len(x) // 20)
        ax.set_xticks(range(0, len(x), step), [x[i] for i in range(0, len(x), step)], rotation=45, ha="right")
    # MORE THAN 40 PERIODS: A LINE CHART READS BETTER THAN HUNDREDS OF THIN BARS.
    else:
        ax.plot(range(len(x)), y, color=BLUE, linewidth=2)
        step = max(1, len(x) // 15)
        ax.set_xticks(range(0, len(x), step), [x[i] for i in range(0, len(x), step)], rotation=45, ha="right")
    ax.set_xlabel(f"{date_col} ({dstats['granularity']})")
    ax.set_ylabel("Number of rows")
    ax.yaxis.set_major_formatter(COMMA)
    ax.set_title(_wrap(f"Rows per {dstats['granularity']} of {date_col}"))
    _save(fig, path)


def plot_boolean(cstats, col, path):
    # BOOLEAN COLUMNS USE THE SAME BAR CHART AS CATEGORIES.
    plot_categorical(cstats, col, path)


def rank_categoricals(cat_stats, cats):
    """Most informative categorical columns first: not extremely high-cardinality, not dominated by one
    value (>90%), mostly complete (missing rounded to whole %), readable labels (average <= 30
    characters), then more categories."""
    # SORT KEY: PYTHON SORTS TUPLES LEFT TO RIGHT AND FALSE BEFORE TRUE, SO COLUMNS WITH NONE OF THESE PROBLEMS COME FIRST:
    # (1) MORE THAN 200 CATEGORIES, (2) ONE VALUE IS OVER 90% OF ROWS, (3) MORE MISSING VALUES, (4) LONG LABELS;
    # FINALLY, MORE CATEGORIES RANK HIGHER (UP TO 200).
    def key(c):
        s = cat_stats[c]
        top_share = s["top_values"][0]["pct_of_non_missing"] if s["top_values"] else 100
        long_labels = sum(len(t["value"]) for t in s["top_values"]) / max(len(s["top_values"]), 1) > 30
        return (s["unique"] > 200, top_share > 90, round(s["missing_pct"]), long_labels, -min(s["unique"], 200))
    return sorted(cats, key=key)


# ---- SELECTION LOGIC -----------------------------------------------------------------
def choose_and_draw(typed, infos, summary, corr, plots_dir, cfg):
    """Build a prioritized list of plot candidates, draw up to cfg.max_plots, and log skips."""
    # CREATE THE PLOTS FOLDER AND PULL THE STATISTICS THIS FUNCTION NEEDS OUT OF THE SUMMARY.
    # EACH CANDIDATE IS (PRIORITY, TYPE, TITLE, WHY IT IS USEFUL, FUNCTION THAT DRAWS IT). LOWER PRIORITY = DRAWN FIRST.
    plots_dir.mkdir(parents=True, exist_ok=True)
    num_stats = summary["numeric_statistics"]
    cat_stats = summary["categorical_statistics"]
    rel = summary["relationships"]
    candidates, skipped = [], []

    # 1. MISSING VALUES
    if any(i["missing"] > 0 for i in infos):
        candidates.append((10, "missing_bar", "Missing-value bar chart",
                           "which columns are incomplete", lambda p: plot_missing(infos, p)))
    else:
        skipped.append({"plot": "Missing-value bar chart", "reason": "no column has missing values"})

    # CHOOSE WHICH NUMERIC MEASURE(S) TO PLOT (RANKING RULE JUST BELOW)
    # 2-3. HISTOGRAM AND BOXPLOT OF THE BEST NUMERIC MEASURE (PLUS A SECOND HISTOGRAM IF THERE IS ROOM).
    measures = [c for c in rel.get("numeric_columns_used", []) if c in num_stats]
    if not measures:
        measures = [c for c in num_stats]
    # PREFER COLUMNS WITH MORE THAN 10 DISTINCT VALUES (CONTINUOUS), THEN THE MOST COMPLETE
    ranked = sorted(measures, key=lambda c: (typed[c].nunique() <= 10, -num_stats[c]["valid_count"]))
    # `lambda p, c=c0:` STORES THE COLUMN NAME NOW; THE PLOT IS ONLY DRAWN LATER, WHEN A FILE PATH `p` IS PASSED IN.
    if ranked:
        c0 = ranked[0]
        candidates.append((20, "histogram", f"Histogram of {c0}", "the shape and skew of the most complete continuous measure",
                           lambda p, c=c0: plot_hist(typed[c], c, num_stats[c], p)))
        candidates.append((30, "boxplot", f"Boxplot of {c0}", "the median, IQR and 1.5×IQR potential outliers",
                           lambda p, c=c0: plot_box(typed[c], c, num_stats[c], p)))
        if len(ranked) > 1:
            c1 = ranked[1]
            candidates.append((75, "histogram", f"Histogram of {c1}", "the shape of a second continuous measure",
                               lambda p, c=c1: plot_hist(typed[c], c, num_stats[c], p)))
    else:
        skipped.append({"plot": "Histogram / boxplot", "reason": "no numeric measure columns"})

    # CATEGORICAL FREQUENCY BARS (PREFER COLUMNS WITH A MANAGEABLE NUMBER OF CATEGORIES)
    # 4. FREQUENCY BAR CHARTS FOR THE ONE OR TWO MOST INFORMATIVE CATEGORICAL COLUMNS (SEE `rank_categoricals()`).
    cats = [c for c, s in cat_stats.items() if s["role"] == "Categorical attribute" and s["unique"] >= 2]
    cats = rank_categoricals(cat_stats, cats)
    if cats:
        candidates.append((40, "category_bar", f"Frequency of {cats[0]}", "how rows are spread across categories",
                           lambda p, c=cats[0]: plot_categorical(cat_stats[c], c, p)))
        if len(cats) > 1:
            candidates.append((70, "category_bar", f"Frequency of {cats[1]}", "frequencies for a second categorical column",
                               lambda p, c=cats[1]: plot_categorical(cat_stats[c], c, p)))
    else:
        skipped.append({"plot": "Categorical frequency bar chart",
                        "reason": "no categorical columns with at least 2 categories"})

    # CORRELATION HEATMAP + SCATTER OF STRONGEST PAIR
    # 5. CORRELATION HEATMAP, ONLY WORTHWHILE WITH 3 OR MORE NUMERIC MEASURES. OTHERWISE THE REASON IS RECORDED.
    if corr is not None and len(corr) >= 3:
        candidates.append((50, "corr_heatmap", "Correlation heatmap", "pairwise linear association among numeric measures",
                           lambda p: plot_heatmap(corr, p)))
    elif corr is not None:
        skipped.append({"plot": "Correlation heatmap", "reason": "only 2 numeric measures; the scatterplot shows their relationship"})
    else:
        skipped.append({"plot": "Correlation heatmap", "reason": rel.get("skipped", "fewer than 2 numeric measures")})

    # 6. SCATTERPLOT OF THE STRONGEST PAIR.
    top_pairs = rel.get("top_pairs_by_abs_r", [])
    # PREFER THE STRONGEST PAIR THAT IS NOT A NEAR-DUPLICATE (|R| >= 0.95 IS OFTEN A DERIVED COLUMN)
    non_dup = rel.get("strongest_non_redundant") or top_pairs
    if non_dup:
        pr = non_dup[0]
        candidates.append((55, "scatter", f"Scatterplot {pr['col_a']} vs {pr['col_b']}",
                           "the strongest numeric relationship that is not a near-duplicate pair (|r| < 0.95)",
                           lambda p, pr=pr: plot_scatter(typed, pr["col_a"], pr["col_b"], pr, cfg, p)))
    else:
        skipped.append({"plot": "Scatterplot", "reason": "no pair of numeric measures with enough overlapping values"})

    # NUMERIC DISTRIBUTION BY CATEGORY
    # 7. BOXPLOT OF A NUMERIC MEASURE SPLIT BY A CATEGORICAL COLUMN.
    gc = summary.get("group_comparison")
    if gc:
        candidates.append((60, "group_box", f"{gc['measure']} by {gc['group_column']}", "how a numeric measure differs across categories",
                           lambda p: plot_group_box(typed, gc["group_column"], gc["measure"], p)))
    else:
        skipped.append({"plot": "Numeric distribution by category",
                        "reason": "needs one numeric measure and one categorical column with 2-15 levels"})

    # TIME
    # 8. ROWS OVER TIME FOR THE MOST COMPLETE DATE-LIKE COLUMN WITH AT LEAST TWO PERIODS.
    dstats = summary["date_statistics"]
    dcols = [c for c, s in dstats.items() if len(s.get("counts_by_period", {})) >= 2]
    if dcols:
        dc = sorted(dcols, key=lambda c: -dstats[c]["valid_count"])[0]
        candidates.append((65, "time_series", f"Rows over time ({dc})", "how many rows fall in each time period (temporal coverage)",
                           lambda p, c=dc: plot_time(c, dstats[c], p)))
    else:
        skipped.append({"plot": "Time-series plot", "reason": "no date-like column with at least 2 distinct periods"})

    # DRAW THE CANDIDATES IN PRIORITY ORDER UNTIL THE CAP (DEFAULT 8) IS REACHED. ANYTHING OVER THE CAP IS RECORDED AS SKIPPED.
    # FILES ARE NAMED PLOT_01_..., PLOT_02_... IN THE ORDER THEY ARE DRAWN.
    candidates.sort(key=lambda t: t[0])
    made = []
    for idx, (prio, kind, title, why, fn) in enumerate(candidates):
        if len(made) >= cfg.max_plots:
            skipped.append({"plot": title, "reason": f"plot limit of {cfg.max_plots} reached"})
            continue
        fname = f"plot_{len(made) + 1:02d}_{kind}.png"
        # IF ONE PLOT FAILS (E.G. ODD DATA), RECORD THE ERROR AND KEEP GOING SO THE REST OF THE REPORT IS STILL PRODUCED.
        try:
            fn(plots_dir / fname)
            made.append({"file": f"plots/{fname}", "type": kind, "title": title, "why": why})
        except Exception as exc:  # A PLOT FAILURE MUST NOT STOP THE REPORT
            plt.close("all")
            skipped.append({"plot": title, "reason": f"plot failed: {exc}"})
    # THE ASSIGNMENT ASKS FOR AT LEAST 5 PLOTS WHEN THE DATA SUPPORT THEM; IF FEWER WERE POSSIBLE, SAY SO IN THE REPORT.
    if len(made) < 5:
        skipped.append({"plot": "(overall)", "reason": f"only {len(made)} meaningful plots could be made for this dataset's column types"})
    return made, skipped
