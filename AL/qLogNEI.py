from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from bo_utils import load_numeric, new_records, recommendation, save_csv, visualize

ROOT = Path(__file__).resolve().parent
FIELDS = ["pool_id", "round", "y", "predicted_mean", "predicted_std"]

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=ROOT / "data/feature.csv")
    p.add_argument("--targets", type=Path, default=ROOT / "data/target_rcs.csv")
    p.add_argument("--initial-observations", type=Path, default= ROOT / "data/existing_results.csv",
                   help="Measured starting recipes for experiment mode (CSV with exported columns)")
    p.add_argument("--mode", choices=["replay", "experiment"], default="experiment")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/qLogNEI")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--trials", type=int, default=3, help="Independent replay trials")
    p.add_argument("--init_point", type=int, default=4)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--minimize", action="store_true")
    p.add_argument("--kernel", choices=["default", "rbf", "matern"], default="default")
    p.add_argument("--matern-nu", type=float, choices=[0.5, 1.5, 2.5], default=2.5)
    p.add_argument("--ard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lengthscale-init", type=float, default=0.5,
                   help="Initial lengthscale for custom kernels; learned during fitting")
    p.add_argument("--noise-std", type=float, default=None,
                   help="Known observation SD in original y units; omit to infer noise")
    p.add_argument("--fit-maxiter", type=int, default=200)
    p.add_argument("--mc-samples", type=int, default=256)
    p.add_argument("--pool-batch-size", type=int, default=128,
                   help="Number of pool candidates evaluated at once; memory control")
    args = p.parse_args()
    for key in ["trials", "init_point", "batch_size", "fit_maxiter", "mc_samples",
                "pool_batch_size", "lengthscale_init"]:
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if args.batches < 0 or (args.noise_std is not None and args.noise_std <= 0):
        p.error("batches must be nonnegative and noise-std must be positive")
    if args.init_point < 2:
        p.error("At least two initial measurements are required")
    if args.kernel == "default" and not args.ard:
        p.error("--no-ard requires --kernel rbf or matern")
    return args


def experimental(ids, targets):
    return targets[np.asarray(ids, dtype=int)]


def validate_records(records, names, raw, source="observations.csv"):
    ids = [int(r["pool_id"]) - 1 for r in records]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= len(raw) for i in ids):
        raise ValueError(f"Duplicate or invalid pool_id in {source}")
    for r, i in zip(records, ids):
        if not np.allclose([float(r[n]) for n in names], raw[i], rtol=0, atol=1e-12):
            raise ValueError(f"Recipe features do not match pool_id in {source}")
        if r["y"] != "" and not np.isfinite(float(r["y"])):
            raise ValueError("Measured y must be finite")


def load_initial_observations(path, names, raw):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS + names:
            raise ValueError(f"Expected columns {FIELDS + names} in {path}")
        records = list(reader)
    if len(records) < 2:
        raise ValueError("At least two initial observations are required")
    for record in records:
        record["y"] = record["y"].strip()
        if record["y"] == "":
            raise ValueError(f"All initial observations must contain y: {path}")
        if int(record["round"]) != 0:
            raise ValueError(f"Initial observations must use round 0: {path}")
    validate_records(records, names, raw, str(path))
    return records


def run(args, names, raw, X, targets, trial):
    folder = args.output / (f"trial_{trial + 1:02d}" if targets is not None else "experiment")
    folder.mkdir(parents=True, exist_ok=True)
    state = folder / "observations.csv"
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["features_sha256"] = hashlib.sha256(args.features.read_bytes()).hexdigest()
    if targets is None and args.initial_observations is not None:
        config["initial_observations_sha256"] = hashlib.sha256(
            args.initial_observations.read_bytes()
        ).hexdigest()
    config["trial_seed"] = args.seed + trial
    config["active_features"] = [n for n, active in zip(names, np.ptp(raw, axis=0) > 0) if active]
    config_path = folder / "config.json"
    if state.exists():
        if targets is not None:
            raise ValueError(f"Output already exists: {folder}. Choose a new --output.")
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Experiment configuration changed; reuse the original command")
        with state.open(encoding="utf-8-sig", newline="") as f:
            records = list(csv.DictReader(f))
        validate_records(records, names, raw)
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n") 
        if targets is None and args.initial_observations is not None:
            records = load_initial_observations(args.initial_observations, names, raw)
        else:
            # 没有已有实测数据时，随机派发第 0 轮配方。
            ids = np.random.default_rng(args.seed + trial).choice(
                len(X), args.init_point, replace=False
            )
            records = new_records(ids, 0, names, raw)
        if targets is not None:
            for r, value in zip(records, experimental(ids, targets)):
                r["y"] = float(value)
        elif args.initial_observations is None:
            save_csv(folder / "recommendations_000.csv", FIELDS + names, records)
        save_csv(state, FIELDS + names, records)
    
    while True:
        history = visualize(records, folder, args.minimize)
        pending = [r for r in records if r["y"] == ""]
        if pending:
            print(f"Waiting for {len(pending)} measurements: fill y in {state}", flush=True)
            return history
        round_id = max(int(r["round"]) for r in records) + 1
        if round_id > args.batches or len(records) == len(X):
            print(f"Finished {folder}: {len(records)} observed, best={history[-1]['best_y']:.6g}")
            return history
        #core
        ids, mean, std = recommendation(X, records, args, args.seed + trial + 1000 * round_id)
        batch = new_records(ids, round_id, names, raw, mean, std)
        # Export the proposal before accessing its measurements.
        save_csv(folder / f"recommendations_{round_id:03d}.csv", FIELDS + names, batch)
        if targets is not None:
            for r, value in zip(batch, experimental(ids, targets)):
                r["y"] = float(value)
        records.extend(batch)
        save_csv(state, FIELDS + names, records)
        print(f"trial {trial + 1}, round {round_id}: selected pool_id {[i + 1 for i in ids]}", flush=True)


def main():
    args = parse_args()
    torch.set_num_threads(4)
    #input setting
    names, raw = load_numeric(args.features)
    
    # 去掉整个候选池中不变化的特征
    active = np.ptp(raw, axis=0) > 0
    # 将剩余特征归一化到 [0, 1]
    values = raw[:, active]
    X = torch.tensor((values - values.min(0)) / np.ptp(values, axis=0), dtype=torch.double)
    targets = None
    if args.mode == "replay":
        _, target_matrix = load_numeric(args.targets)
        targets = target_matrix[:, 0]
        #run
        histories = [run(args, names, raw, X, targets, trial) for trial in range(args.trials)]
        #plot
        '''
        fig, ax = plt.subplots(figsize=(7, 4))
        curves = np.array([[h["best_y"] for h in history] for history in histories])
        counts = [h["n_observed"] for h in histories[0]]
        mean, std = curves.mean(0), curves.std(0)
        ax.plot(counts, mean, marker="o", label="qLogNEI: trial mean")
        ax.fill_between(counts, mean - std, mean + std, alpha=0.2, label="±1 trial SD")
        ax.axhline(targets.min() if args.minimize else targets.max(), color="black",
                    linestyle="--", label="Pool optimum (evaluation only)")
        ax.set(xlabel="Number of measured recipes", ylabel="Best observed y")
        ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
        fig.savefig(args.output / "optimization.png", dpi=180)
        plt.close(fig)
        '''

    # 真实实验不读取 targets；run 会在每批推荐后等待人工回填 y。
    if args.mode == "experiment":
        run(args, names, raw, X, targets=None, trial=0)


if __name__ == "__main__":
    main()
