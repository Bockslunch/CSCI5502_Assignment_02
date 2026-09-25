"""Convenience script: profile both assignment datasets with the SAME analysis code.

Only the input path and output folder differ between the two runs - this is the
"documented configuration" allowed by the generalization requirement.

    python src/run_all.py            # with the LLM (Ollama must be running)
    python src/run_all.py --no-llm   # deterministic profile only
    python src/run_all.py --model qwen2.5:7b   # a different Ollama model
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler import generate_profile  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "dataset_a": ROOT / "data" / "chicago_energy_benchmarking.csv",
    "dataset_b": ROOT / "data" / "wa_electric_vehicle_population.csv",
}

if __name__ == "__main__":
    import argparse

    from config import ProfilerConfig

    ap = argparse.ArgumentParser(description="Profile both assignment datasets")
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM narrative step")
    ap.add_argument("--model", default=ProfilerConfig.llm_model, help="Ollama model (default: %(default)s)")
    args = ap.parse_args()
    cfg = ProfilerConfig(llm_model=args.model)
    for name, csv_path in DATASETS.items():
        print(f"\n===== {name}: {csv_path.name} =====")
        generate_profile(csv_path, ROOT / "output" / name, use_llm=not args.no_llm, config=cfg)
