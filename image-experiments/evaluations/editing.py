import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm


METRICS = {
    "IR": "IR",
    "CLIPScore": "CLIPScore",
    "PickScore": "PickScore",
    "LPIPS": "LPIPS_orig_vs_recon",
    "LPIPS Edit": "LPIPS_edit_vs_recon",
}


def load_run(path, max_samples):
    files = sorted(Path(path).glob("*.json"))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise ValueError(f"No JSON result files found in {path}")

    values = {name: [] for name in METRICS}
    for result_path in tqdm(files, desc=str(path)):
        with result_path.open() as result_file:
            result = json.load(result_file)
        for name, key in METRICS.items():
            values[name].append(result[key])

    return {name: float(np.mean(samples)) for name, samples in values.items()}, len(files)


def format_metrics(metrics):
    return " || ".join(f"{name}: {value:.4f}" for name, value in metrics.items())


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="One or more directories of per-image JSON results (one directory per seed)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples per run; use 0 to include every JSON file",
    )
    args = parser.parse_args()

    max_samples = None if args.max_samples == 0 else args.max_samples
    runs = []
    for path in args.paths:
        metrics, count = load_run(path, max_samples)
        runs.append(metrics)
        print(f"{path} (n={count}) || {format_metrics(metrics)}")

    if len(runs) > 1:
        print(f"Across {len(runs)} runs/seeds (mean ± sample std):")
        for name in METRICS:
            samples = np.asarray([run[name] for run in runs])
            print(f"{name}: {samples.mean():.4f} ± {samples.std(ddof=1):.4f}")
