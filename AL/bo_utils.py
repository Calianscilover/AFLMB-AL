"""Shared CSV, GP fitting, single/multi-objective recommendation and plotting utilities."""
from __future__ import annotations

import csv
from copy import copy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


def load_numeric(path):
    """Read a header and a numeric matrix from a clean CSV."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        names = [name.strip() for name in next(csv.reader(f))]
        data = np.loadtxt(f, delimiter=",", ndmin=2)
    if not all(names) or len(set(names)) != len(names):
        raise ValueError(f"Column names must be nonempty and unique: {path}")
    if data.size == 0 or data.shape[1] != len(names):
        raise ValueError(f"Expected a nonempty matrix matching the CSV header: {path}")
    if not np.isfinite(data).all():
        raise ValueError(f"Nonfinite values in {path}")
    return names, data


def fit_model(X, records, args):
    ids = [int(r["pool_id"]) - 1 for r in records]
    sign = -1 if args.minimize else 1
    Y = torch.tensor([[sign * float(r["y"])] for r in records], dtype=torch.double)
    covariance = None
    if args.kernel != "default":
        kw = {"ard_num_dims": X.shape[-1] if args.ard else None}
        base = RBFKernel(**kw) if args.kernel == "rbf" else MaternKernel(nu=args.matern_nu, **kw)
        base.initialize(lengthscale=args.lengthscale_init)
        covariance = ScaleKernel(base)
    variance = None if args.noise_std is None else torch.full_like(Y, args.noise_std**2)
    model = SingleTaskGP(X[ids], Y, train_Yvar=variance,
                         covar_module=covariance, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model),
                    optimizer_kwargs={"options": {"maxiter": args.fit_maxiter}})
    return model


def recommendation(X, records, args, seed):
    torch.manual_seed(seed)
    used = {int(r["pool_id"]) - 1 for r in records}
    remaining = [i for i in range(len(X)) if i not in used]
    model = fit_model(X, records, args)
    acq = qLogNoisyExpectedImprovement(
        model=model, X_baseline=X[sorted(used)],
        sampler=SobolQMCNormalSampler(torch.Size([args.mc_samples]), seed=seed),
    )
    selected, _ = optimize_acqf_discrete(
        acq, q=min(args.batch_size, len(remaining)), choices=X[remaining],
        unique=True, max_batch_size=args.pool_batch_size,
    )
    ids = [remaining[int(torch.argmin((X[remaining] - point).square().sum(-1)))]
           for point in selected]
    with torch.no_grad():
        posterior = model.posterior(selected)
    mean = posterior.mean.squeeze(-1).numpy() * (-1 if args.minimize else 1)
    std = posterior.variance.sqrt().squeeze(-1).numpy()
    return ids, mean, std


def save_csv(path, fields, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fit_multi_model(X, records, args, target):
    models = []
    for j, name in enumerate(target):
        objective_args = copy(args)
        objective_args.minimize = args.directions[j] == "min"
        objective_args.noise_std = None if args.noise_std is None else args.noise_std[j]
        objective_records = [dict(pool_id=r["pool_id"], y=r[name]) for r in records]
        models.append(fit_model(X, objective_records, objective_args))
    return ModelListGP(*models)


def multi_recommendation(X, records, args, seed, target_names, ref_point):
    torch.manual_seed(seed)
    used = {int(r["pool_id"]) - 1 for r in records}
    remaining = [i for i in range(len(X)) if i not in used]
    if not remaining:
        raise ValueError("No unmeasured candidates remain")
    signs = X.new_tensor([-1 if d == "min" else 1 for d in args.directions])
    model = fit_multi_model(X, records, args, target_names)
    acq = qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=X.new_tensor(ref_point) * signs,
        X_baseline=X[sorted(used)],
        sampler=SobolQMCNormalSampler(torch.Size([args.mc_samples]), seed=seed),
    )
    selected, _ = optimize_acqf_discrete(
        acq, q=min(args.batch_size, len(remaining)), choices=X[remaining],
        unique=True, max_batch_size=args.pool_batch_size,
    )
    # Remove matched row IDs so equal feature vectors cannot map to one ID twice.
    available = remaining.copy()
    ids = []
    for point in selected:
        match = int(torch.argmin((X[available] - point).square().sum(-1)))
        ids.append(available.pop(match))
    with torch.no_grad():
        posterior = model.posterior(selected)
    mean = (posterior.mean * signs).detach().cpu().numpy()
    std = posterior.variance.clamp_min(0).sqrt().detach().cpu().numpy()
    return ids, mean, std


def new_records(ids, round_id, names, raw, mean=None, std=None):
    return [dict(pool_id=i + 1, round=round_id, y="",
                 predicted_mean="" if mean is None else float(mean[j]),
                 predicted_std="" if std is None else float(std[j]),
                 **dict(zip(names, raw[i]))) for j, i in enumerate(ids)]


def visualize(records, folder, minimize):
    completed = [r for r in records if r["y"] != ""]
    if not completed:
        return []
    history = []
    for round_id in sorted({int(r["round"]) for r in completed}):
        y = [float(r["y"]) for r in completed if int(r["round"]) <= round_id]
        history.append(dict(round=round_id, n_observed=len(y),
                            best_y=float(min(y) if minimize else max(y))))
    save_csv(folder / "history.csv", ["round", "n_observed", "best_y"], history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter([int(r["round"]) for r in completed], [float(r["y"]) for r in completed],
                    label="Measured y", alpha=0.65)
    axes[0].plot([h["round"] for h in history], [h["best_y"] for h in history],
                 color="black", label="Best observed", marker="o")
    axes[0].set(xlabel="BO round (0 = initial design)", ylabel="Objective y")
    axes[0].legend()
    predicted = [r for r in completed if r["predicted_mean"] != ""]
    if predicted:
        axes[1].errorbar([float(r["y"]) for r in predicted],
                         [float(r["predicted_mean"]) for r in predicted],
                         yerr=[1.96 * float(r["predicted_std"]) for r in predicted],
                         fmt="o", alpha=0.65)
        lo, hi = axes[1].get_xlim()
        axes[1].plot([lo, hi], [lo, hi], "k--")
    axes[1].set(xlabel="Measured y", ylabel="Prediction BEFORE experiment",
                title="GP latent mean ± 1.96 posterior SD")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "optimization.png", dpi=180)
    plt.close(fig)
    return history
