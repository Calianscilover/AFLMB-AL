"""Fit a single-objective GP from measured electrolyte experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bo_utils import (existing_recommendation_path, fit_model_data,
                      load_experiment_data, predict_candidates,
                      recommendation, recommendation_path, record_training,
                      save_candidate_predictions, save_recommendations,
                      sync_observations)

ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path,
                        default=ROOT / "pool/converted/experiment.csv")
    parser.add_argument("--pool", type=Path,
                        default=ROOT / "pool/converted/pool_catalog.csv")
    parser.add_argument("--target", default="Conductivity")
    parser.add_argument("--feature-columns", nargs="+", default=None,
                        help="Explicit numeric input columns in both CSVs")
    parser.add_argument("--feature-basis", choices=["mass", "mole"], default="mass",
                        help="Select *_ratio_N columns when --feature-columns is omitted")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/single_training")
    parser.add_argument("--minimize", action="store_true")
    parser.add_argument("--kernel", choices=["default", "rbf", "matern"], default="default")
    parser.add_argument("--matern-nu", type=float, choices=[0.5, 1.5, 2.5], default=2.5)
    parser.add_argument("--ard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lengthscale-init", type=float, default=0.5)
    parser.add_argument("--noise-std", type=float, default=None)
    parser.add_argument("--fit-maxiter", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--mc-samples", type=int, default=256)
    parser.add_argument("--pool-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not np.isfinite(args.lengthscale_init) or args.lengthscale_init <= 0:
        parser.error("lengthscale-init must be finite and positive")
    if args.fit_maxiter <= 0:
        parser.error("fit-maxiter must be positive")
    if min(args.batch_size, args.mc_samples, args.pool_batch_size) <= 0:
        parser.error("batch-size, mc-samples and pool-batch-size must be positive")
    if args.noise_std is not None and (not np.isfinite(args.noise_std)
                                       or args.noise_std <= 0):
        parser.error("noise-std must be finite and positive")
    if args.kernel == "default" and not args.ard:
        parser.error("--no-ard requires --kernel rbf or matern")
    return args


def run(args):
    observations, round_id, pending = sync_observations(
        args.experiment, args.pool, args.output, [args.target]
    )
    if pending:
        print(f"Waiting for {pending} target values in "
              f"{existing_recommendation_path(args.output, round_id - 1)}")
        return None
    data = load_experiment_data(
        observations, args.pool, label="single", target_names=[args.target],
        feature_columns=args.feature_columns, feature_basis=args.feature_basis,
    )
    model = fit_model_data(data.train_X, data.train_Y, args)
    summary = record_training(args.output, data, model, args, "single",
                              observation_path=observations, round_id=round_id)
    if not data.candidate_indices:
        print(f"All candidate recipes have been measured. Saved {observations}")
        return data, model
    directions = ["min" if args.minimize else "max"]
    means, stds = predict_candidates(data, model, directions=directions,
                                      batch_size=args.pool_batch_size)
    prediction_name = ("candidate_predictions.csv" if round_id == 1 else
                       f"candidate_predictions_{round_id}.csv")
    save_candidate_predictions(args.output / prediction_name, data, means, stds)
    indices = recommendation(data, model, args)
    save_recommendations(recommendation_path(args.output, round_id), args.pool,
                         data, indices)
    print(f"Fitted {summary['n_observed']} measurements with "
          f"{summary['input_dim']} input features; "
          f"{summary['n_candidates']} candidates remain; round {round_id} "
          f"recommended {len(indices)}. "
          f"Saved to {args.output}")
    return data, model


if __name__ == "__main__":
    torch.set_num_threads(4)
    run(parse_args())
