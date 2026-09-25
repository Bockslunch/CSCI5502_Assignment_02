"""Convenience script: profile both assignment datasets with the SAME analysis code.

Only the input path and output folder differ between the two runs - this is the
"documented configuration" allowed by the generalization requirement.

    python src/run_all.py            # with the LLM (Ollama must be running)
    python src/run_all.py --no-llm   # deterministic profile only
    python src/run_all.py --model qwen2.5:7b   # a different Ollama model
"""

# STANDARD-LIBRARY IMPORTS: `sys` TO ADJUST THE IMPORT PATH, `Path` FOR WINDOWS/MAC/LINUX-SAFE FILE PATHS.
import sys
from pathlib import Path

# LET PYTHON FIND THE OTHER MODULES IN `src/` (PROFILER, CONFIG, ...) NO MATTER WHICH FOLDER THE SCRIPT IS RUN FROM.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler import generate_profile  # noqa: E402

# ROOT IS THE REPOSITORY FOLDER (ONE LEVEL ABOVE `src/`). ALL DATA AND OUTPUT PATHS ARE BUILT FROM IT.
# THE TWO ASSIGNMENT DATASETS: OUTPUT FOLDER NAME -> CSV FILE. THIS IS THE ONLY DATASET-SPECIFIC PART OF THE PROJECT,
# AND IT IS JUST FILE PATHS - THE ANALYSIS CODE IS IDENTICAL FOR BOTH.
ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "dataset_a": ROOT / "data" / "chicago_energy_benchmarking.csv",
    "dataset_b": ROOT / "data" / "wa_electric_vehicle_population.csv",
}

# THIS BLOCK RUNS ONLY WHEN THE FILE IS EXECUTED DIRECTLY (`python src/run_all.py`), NOT WHEN IT IS IMPORTED.
if __name__ == "__main__":
    import argparse

    from config import ProfilerConfig

    # READ THE OPTIONAL COMMAND-LINE SWITCHES: `--no-llm` (SKIP THE MODEL) AND `--model` (CHOOSE A DIFFERENT OLLAMA MODEL).
    ap = argparse.ArgumentParser(description="Profile both assignment datasets")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM narrative step")
    ap.add_argument("--model", default=ProfilerConfig.llm_model, help="Ollama model (default: %(default)s)")
    args = ap.parse_args()
    # BUILD ONE CONFIG (DEFAULT THRESHOLDS + THE CHOSEN MODEL) AND USE IT FOR BOTH DATASETS.
    cfg = ProfilerConfig(llm_model=args.model)
    # RUN THE SAME `generate_profile()` ON EACH DATASET; ONLY THE INPUT FILE AND OUTPUT FOLDER CHANGE.
    for name, csv_path in DATASETS.items():
        print(f"\n===== {name}: {csv_path.name} =====")
        generate_profile(csv_path, ROOT / "output" / name, use_llm=not args.no_llm, config=cfg)
