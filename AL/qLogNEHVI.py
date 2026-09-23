"""Two-objective pool optimization with replay and real-experiment modes."""
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
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from bo_utils import load_numeric, multi_recommendation, save_csv

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, default=ROOT / "data/feature.csv")
    p.add_argument("--targets", type=Path, default=ROOT / "data/multi_target.csv")
    p.add_argument("--initial-observations", type=Path,
                   default=ROOT / "data/existing_multi_results.csv",
                   help="Measured starting recipes for experiment mode")
    p.add_argument("--target-names", nargs=2, default=["De", "Sigma_e"],
                   help="Objective column names used in experiment mode")
    p.add_argument("--mode", choices=["replay", "experiment"], default="experiment")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/qLogNEHVI")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--init_point", "--init-point", type=int, default=4)
    p.add_argument("--batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--directions", nargs=2, choices=["max", "min"], default=["max", "max"],
                   help="Optimization directions in objective-column order")
    p.add_argument("--ref-point", nargs=2, type=float, default=None,
                   help="Fixed reference in original objective units; default: round-0 data")
    p.add_argument("--kernel", choices=["default", "rbf", "matern"], default="default")
    p.add_argument("--matern-nu", type=float, choices=[0.5, 1.5, 2.5], default=2.5)
    p.add_argument("--ard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lengthscale-init", type=float, default=0.5)
    p.add_argument("--noise-std", nargs=2, type=float, default=None,
                   help="Known observation SD per objective in original units")
    p.add_argument("--fit-maxiter", type=int, default=200)
    p.add_argument("--mc-samples", type=int, default=256)
    p.add_argument("--pool-batch-size", type=int, default=128)
    args = p.parse_args()
    for key in ["trials", "init_point", "batch_size", "fit_maxiter", "mc_samples",
                "pool_batch_size", "lengthscale_init"]:
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"{key} must be finite and positive")
    if args.batches < 0 or args.init_point < 2:
        p.error("batches must be nonnegative; at least two initial points are required")
    if args.noise_std is not None and any(not np.isfinite(v) or v <= 0 for v in args.noise_std):
        p.error("noise-std values must be finite and positive")
    if args.ref_point is not None and not np.isfinite(args.ref_point).all():
        p.error("ref-point must be finite")
    if len(set(args.target_names)) != 2 or not all(args.target_names):
        p.error("target-names must contain two distinct nonempty names")
    if args.kernel == "default" and not args.ard:
        p.error("--no-ard requires --kernel rbf or matern")
    return args


def multi_fields(names, target_names):
    fields = ["pool_id", "round"]
    for name in target_names:
        fields.extend([name, f"predicted_mean_{name}", f"predicted_std_{name}"])
    return fields + names


def new_multi_records(ids, round_id, names, raw, target_names, mean=None, std=None):
    rows = []
    for j, i in enumerate(ids):
        row = dict(pool_id=int(i) + 1, round=round_id)
        for k, name in enumerate(target_names):
            row[name] = ""
            row[f"predicted_mean_{name}"] = "" if mean is None else float(mean[j, k])
            row[f"predicted_std_{name}"] = "" if std is None else float(std[j, k])
        row.update(zip(names, raw[i]))
        rows.append(row)
    return rows


def reveal(rows, targets, target_names):
    for row in rows:
        for name, value in zip(target_names, targets[int(row["pool_id"]) - 1]):
            row[name] = float(value)


def validate_records(records, names, raw, target_names, source="observations.csv"):
    ids = [int(r["pool_id"]) - 1 for r in records]
    if len(set(ids)) != len(ids) or any(i < 0 or i >= len(raw) for i in ids):
        raise ValueError(f"Duplicate or invalid pool_id in {source}")
    for record, i in zip(records, ids):
        if not np.allclose([float(record[n]) for n in names], raw[i], rtol=0, atol=1e-12):
            raise ValueError(f"Recipe features do not match pool_id in {source}")
        for name in target_names:
            record[name] = record[name].strip() if isinstance(record[name], str) else record[name]
            if record[name] != "" and not np.isfinite(float(record[name])):
                raise ValueError(f"Measured {name} must be finite in {source}")


def load_initial_observations(path, names, raw, target_names):
    fields = multi_fields(names, target_names)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError(f"Expected columns {fields} in {path}")
        records = list(reader)
    if len(records) < 2:
        raise ValueError("At least two initial observations are required")
    validate_records(records, names, raw, target_names, str(path))
    for record in records:
        if int(record["round"]) != 0:
            raise ValueError(f"Initial observations must use round 0: {path}")
        if any(record[name] == "" for name in target_names):
            raise ValueError(f"All initial observations must contain every objective: {path}")
    return records


def completed_records(records, target_names):
    return [r for r in records if all(r[name] != "" for name in target_names)]


def observed_values(records, target_names):
    return np.array([[float(r[name]) for name in target_names] for r in records])


def initial_reference(values, signs):
    """Place the reference ten percent below the round-0 worst signed value."""
    signed = values * signs
    scale = np.where(np.ptp(signed, axis=0) > 0, np.ptp(signed, axis=0),
                     np.maximum(np.abs(signed).max(0), 1.0))
    return (signed.min(0) - 0.1 * scale) * signs


def visualize(records, folder, names, target_names, signs, ref_point, pool_targets=None):
    completed = completed_records(records, target_names)
    if not completed:
        return []
    hv = Hypervolume(torch.tensor(ref_point * signs, dtype=torch.double))
    history = []
    for round_id in sorted({int(r["round"]) for r in completed}):
        current = [r for r in completed if int(r["round"]) <= round_id]
        Y = torch.tensor(observed_values(current, target_names) * signs, dtype=torch.double)
        mask = is_non_dominated(Y)
        history.append(dict(round=round_id, n_observed=len(current),
                            hypervolume=hv.compute(Y[mask]), n_pareto=int(mask.sum())))
    save_csv(folder / "history.csv", list(history[0]), history)
    Y = torch.tensor(observed_values(completed, target_names) * signs, dtype=torch.double)
    mask = is_non_dominated(Y)
    save_csv(folder / "pareto.csv", multi_fields(names, target_names),
             [r for r, keep in zip(completed, mask.tolist()) if keep])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([h["n_observed"] for h in history],
                 [h["hypervolume"] for h in history], marker="o")
    if pool_targets is not None:
        pool_y = torch.tensor(pool_targets * signs, dtype=torch.double)
        pool_hv = hv.compute(pool_y[is_non_dominated(pool_y)])
        axes[0].axhline(pool_hv, color="black", linestyle="--",
                        label="Full pool (evaluation only)")
        axes[0].legend()
    axes[0].set(xlabel="Number of measured recipes", ylabel="Observed hypervolume")
    observed = observed_values(completed, target_names)
    pareto = observed[mask.numpy()]
    if pool_targets is not None:
        axes[1].scatter(pool_targets[:, 0], pool_targets[:, 1], color="lightgray",
                        label="Pool (evaluation only)")
    axes[1].scatter(observed[:, 0], observed[:, 1], label="Observed")
    axes[1].scatter(pareto[:, 0], pareto[:, 1], marker="x", color="red",
                    label="Observed Pareto")
    axes[1].set(xlabel=target_names[0], ylabel=target_names[1])
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "optimization.png", dpi=180)
    plt.close(fig)
    return history


def run(args, names, raw, X, targets, target_names, trial):
    folder = args.output / (f"trial_{trial + 1:02d}" if targets is not None else "experiment")
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / "observations.csv"
    fields = multi_fields(names, target_names)
    signs = np.array([-1 if d == "min" else 1 for d in args.directions])

    if state_path.exists():
        if targets is not None:
            raise ValueError(f"Output already exists: {folder}. Choose a new --output.")
        with state_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != fields:
                raise ValueError(f"Keep the original CSV columns: {state_path}")
            records = list(reader)
        validate_records(records, names, raw, target_names)
    elif targets is None:
        records = load_initial_observations(args.initial_observations, names, raw, target_names)
    else:
        ids = np.random.default_rng(args.seed + trial).choice(
            len(X), args.init_point, replace=False
        )
        records = new_multi_records(ids, 0, names, raw, target_names)
        save_csv(folder / "recommendations_000.csv", fields, records)
        reveal(records, targets, target_names)

    round_zero = [r for r in completed_records(records, target_names) if int(r["round"]) == 0]
    if args.ref_point is None:
        if not round_zero:
            raise ValueError("A reference point requires completed round-0 observations")
        ref_point = initial_reference(observed_values(round_zero, target_names), signs)
    else:
        ref_point = np.asarray(args.ref_point)

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(
        trial_seed=args.seed + trial,
        target_names=target_names,
        resolved_ref_point=ref_point.tolist(),
        reference_source="round-0 observations" if args.ref_point is None else "user",
        active_features=[n for n, active in zip(names, np.ptp(raw, axis=0) > 0) if active],
        features_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest(),
    )
    if targets is None:
        config["initial_observations_sha256"] = hashlib.sha256(
            args.initial_observations.read_bytes()
        ).hexdigest()
    else:
        config["targets_sha256"] = hashlib.sha256(args.targets.read_bytes()).hexdigest()
    config_path = folder / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Experiment configuration changed; reuse the original command")
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        save_csv(state_path, fields, records)

    while True:
        history = visualize(records, folder, names, target_names, signs, ref_point, targets)
        pending = [r for r in records if any(r[name] == "" for name in target_names)]
        if pending:
            print(f"Waiting for {len(pending)} measurements: fill objectives in {state_path}",
                  flush=True)
            return history
        round_id = max(int(r["round"]) for r in records) + 1
        if round_id > args.batches or len(records) == len(X):
            print(f"Finished {folder}: {len(records)} observed, "
                  f"HV={history[-1]['hypervolume']:.6g}")
            return history
        ids, mean, std = multi_recommendation(
            X, records, args, args.seed + trial + 1000 * round_id, target_names, ref_point
        )
        batch = new_multi_records(ids, round_id, names, raw, target_names, mean, std)
        save_csv(folder / f"recommendations_{round_id:03d}.csv", fields, batch)
        if targets is not None:
            reveal(batch, targets, target_names)
        records.extend(batch)
        save_csv(state_path, fields, records)
        print(f"trial {trial + 1}, round {round_id}: selected pool_id {[i + 1 for i in ids]}",
              flush=True)


def main():
    args = parse_args()
    torch.set_num_threads(4)
    names, raw = load_numeric(args.features)
    if len(np.unique(raw, axis=0)) != len(raw):
        raise ValueError("Duplicate feature vectors in the pool")
    active = np.ptp(raw, axis=0) > 0
    if not active.any():
        raise ValueError("Pool has no varying features")
    values = raw[:, active]
    X = torch.tensor((values - values.min(0)) / np.ptp(values, axis=0), dtype=torch.double)

    if args.mode == "replay":
        target_names, targets = load_numeric(args.targets)
        if len(target_names) != 2 or len(targets) != len(raw):
            raise ValueError("Targets must have two columns and align with feature rows")
        for trial in range(args.trials):
            run(args, names, raw, X, targets, target_names, trial)
    else:
        run(args, names, raw, X, targets=None, target_names=args.target_names, trial=0)


if __name__ == "__main__":
    main()
