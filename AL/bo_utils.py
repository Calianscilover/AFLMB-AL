from __future__ import annotations

import csv
import hashlib
import json
import re
from copy import copy
from dataclasses import dataclass
from pathlib import Path

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


@dataclass
class ExperimentData:
    train_X: torch.Tensor
    train_Y: torch.Tensor
    pool_X: torch.Tensor
    feature_names: list[str]
    target_names: list[str]
    observed_indices: list[int]
    candidate_indices: list[int]
    sample_ids: list[str]

    @property
    def input_dim(self):
        return self.pool_X.shape[-1]

#读取带表头的 CSV，并检查空表、重复列名和损坏的行
def _read_table(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or not all(fields) or len(fields) != len(set(fields)):
            raise ValueError(f"CSV columns must be nonempty and unique: {path}")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"CSV is empty or has malformed rows: {path}")
    return fields, rows

#  按 sample_id 对齐实验表与候选池；从实验表取目标值，从候选池取特征；
def load_experiment_data(experiment_path, pool_path, *, label,
                         target_names=None, feature_columns=None, feature_basis="mass"):
    if label not in {"single", "multi"}:
        raise ValueError("label must be 'single' or 'multi'")
    targets = list(target_names or (["Conductivity"] if label == "single"
                                    else ["logCE", "Conductivity"]))
    expected = 1 if label == "single" else 2
    if len(targets) != expected or len(set(targets)) != expected:
        raise ValueError(f"{label} requires {expected} distinct target column(s)")
    experiment_fields, experiments = _read_table(experiment_path)
    pool_fields, pool = _read_table(pool_path)
    if "sample_id" not in experiment_fields or "sample_id" not in pool_fields:
        raise ValueError("Both input files require sample_id")
    if any(name not in experiment_fields for name in targets):
        raise ValueError(f"Missing target columns in {experiment_path}: {targets}")
    if feature_columns is None:
        pattern = re.compile(rf"{re.escape(feature_basis)}_ratio_(\d+)$")
        indexed = sorted((int(match.group(1)), name) for name in pool_fields
                         if (match := pattern.fullmatch(name)))
        if [index for index, _ in indexed] != list(range(len(indexed))):
            raise ValueError("Ratio feature slots must be contiguous from zero")
        features = [name for _, name in indexed]
    else:
        features = list(feature_columns)
    if not features or len(set(features)) != len(features):
        raise ValueError("Select one or more distinct feature columns")
    if set(features) & (set(targets) | {"logCE", "Conductivity"}):
        raise ValueError("Performance columns cannot be used as input features")
    if any(name not in pool_fields or name not in experiment_fields for name in features):
        raise ValueError("Feature columns must exist in both input files")

    sample_ids = [row["sample_id"] for row in pool]
    experiment_ids = [row["sample_id"] for row in experiments]
    if (any(not value for value in sample_ids + experiment_ids)
            or len(set(sample_ids)) != len(sample_ids)
            or len(set(experiment_ids)) != len(experiment_ids)):
        raise ValueError("sample_id values must be nonempty and unique in each file")
    index = {sample_id: i for i, sample_id in enumerate(sample_ids)}
    if any(sample_id not in index for sample_id in experiment_ids):
        raise ValueError("An experimental sample_id is absent from the candidate pool")

    raw = np.array([[float(row[name]) for name in features] for row in pool])
    #去掉恒定特征并按整个池的范围归一化。
    span = np.ptp(raw, axis=0)
    active = span > 0
    if not active.any():
        raise ValueError("Pool has no varying features")
    if len(np.unique(raw[:, active], axis=0)) != len(pool):
        raise ValueError("Duplicate feature vectors in the pool")
    pool_X = torch.as_tensor((raw[:, active] - raw[:, active].min(0)) / span[active],
                             dtype=torch.double)

    #只有目标齐全的行进入训练，未测或部分已测行仍在候选集中
    observed = []
    values = []
    for row in experiments:
        i = index[row["sample_id"]]
        try:
            row_features = np.array([float(row[name]) for name in features])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Non-numeric experimental features: {row['sample_id']}") from error
        if not np.allclose(row_features, raw[i], rtol=0, atol=1e-12):
            raise ValueError(f"Experimental features differ from pool: {row['sample_id']}")
        measurements = [row[name].strip() for name in targets]
        for value in measurements:
            if value:
                try:
                    finite = np.isfinite(float(value))
                except ValueError as error:
                    raise ValueError(f"Non-numeric measured target: {row['sample_id']}") from error
                if not finite:
                    raise ValueError(f"Nonfinite measured target: {row['sample_id']}")
        if all(measurements):
            observed.append(i)
            values.append([float(value) for value in measurements])
    if len(observed) < 2:
        raise ValueError("At least two complete experimental measurements are required")
    observed_set = set(observed)
    return ExperimentData(
        train_X=pool_X[observed],
        train_Y=torch.as_tensor(values, dtype=torch.double),
        pool_X=pool_X,
        feature_names=[name for name, keep in zip(features, active) if keep],
        target_names=targets,
        observed_indices=observed,
        candidate_indices=[i for i in range(len(pool)) if i not in observed_set],
        sample_ids=sample_ids,
    )

#保存当前轮的 training_summary.json 和 model.pt，记录数据指纹、参数、已测数量及参考点等；这些文件目前会被下一轮更新。
def record_training(output, data, model, args, label, ref_point=None,
                    observation_path=None, round_id=None):
    """Save the fitted state and enough context to inspect this training run."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": label,
        "experiment": str(Path(args.experiment).resolve()),
        "pool": str(Path(args.pool).resolve()),
        "experiment_sha256": hashlib.sha256(Path(args.experiment).read_bytes()).hexdigest(),
        "pool_sha256": hashlib.sha256(Path(args.pool).read_bytes()).hexdigest(),
        "feature_names": data.feature_names,
        "target_names": data.target_names,
        "input_dim": data.input_dim,
        "n_observed": len(data.observed_indices),
        "n_candidates": len(data.candidate_indices),
        "observed_sample_ids": [data.sample_ids[i] for i in data.observed_indices],
        "directions": (args.directions if label == "multi"
                       else ["min" if args.minimize else "max"]),
        "kernel": args.kernel,
        "matern_nu": args.matern_nu,
        "ard": args.ard,
        "lengthscale_init": args.lengthscale_init,
        "noise_std": args.noise_std,
        "fit_maxiter": args.fit_maxiter,
        "batch_size": args.batch_size,
        "mc_samples": args.mc_samples,
        "pool_batch_size": args.pool_batch_size,
        "seed": args.seed,
    }
    if ref_point is not None:
        summary["ref_point"] = np.asarray(ref_point).tolist()
    if observation_path is not None:
        summary["observations"] = str(Path(observation_path).resolve())
        summary["observations_sha256"] = hashlib.sha256(
            Path(observation_path).read_bytes()
        ).hexdigest()
    if round_id is not None:
        summary["round"] = round_id
    temporary = output / "training_summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "training_summary.json")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "train_X": data.train_X,
        "train_Y": data.train_Y,
        "feature_names": data.feature_names,
        "target_names": data.target_names,
    }
    temporary = output / "model.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(output / "model.pt")
    return summary

#单目标对应的 qLogNEI 选择候选池中最有前景的配方，返回其索引、预测均值和标准差。
def fit_model_data(train_X, train_Y, args):
    sign = -1 if args.minimize else 1
    Y = sign * train_Y.reshape(-1, 1)
    covariance = None
    if args.kernel != "default":
        kw = {"ard_num_dims": train_X.shape[-1] if args.ard else None}
        base = RBFKernel(**kw) if args.kernel == "rbf" else MaternKernel(nu=args.matern_nu, **kw)
        base.initialize(lengthscale=args.lengthscale_init)
        covariance = ScaleKernel(base)
    variance = None if args.noise_std is None else torch.full_like(Y, args.noise_std**2)
    model = SingleTaskGP(train_X, Y, train_Yvar=variance,
                         covar_module=covariance, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model),
                    optimizer_kwargs={"options": {"maxiter": args.fit_maxiter}})
    return model

def fit_model(X, records, args):
    ids = [int(r["pool_id"]) - 1 for r in records]
    Y = torch.tensor([[float(r["y"])] for r in records], dtype=torch.double)
    return fit_model_data(X[ids], Y, args)


def legacy_recommendation(X, records, args, seed):
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

#分批预测所有未测候选，返回与 candidate_indices 顺序对应的均值和后验标准差
def predict_candidates(data, model, *, directions, batch_size=128):
    """Predict every unmeasured recipe in original target units."""
    signs = np.array([-1 if direction == "min" else 1 for direction in directions])
    means, stds = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data.candidate_indices), batch_size):
            ids = data.candidate_indices[start:start + batch_size]
            posterior = model.posterior(data.pool_X[ids])
            means.append(posterior.mean.detach().cpu().numpy() * signs)
            stds.append(posterior.variance.clamp_min(0).sqrt().detach().cpu().numpy())
    shape = (0, len(data.target_names))
    return (np.concatenate(means) if means else np.empty(shape),
            np.concatenate(stds) if stds else np.empty(shape))

#将采集优化器返回的归一化特征点映射回候选池行号，并防止同一行在一批中重复。
def _selected_indices(selected, data):
    available = data.candidate_indices.copy()
    ids = []
    for point in selected:
        distances = (data.pool_X[available] - point).square().sum(-1)
        match = int(torch.argmin(distances))
        if distances[match] > 1e-12:
            raise RuntimeError("Acquisition optimizer returned a point outside the pool")
        ids.append(available.pop(match))
    return ids

#以已测输入为 baseline，用 qLogNEI 在未测池中做离散批量选点，返回池行号。
def recommendation(data, model, args):
    if not data.candidate_indices:
        return []
    torch.manual_seed(args.seed)
    acquisition = qLogNoisyExpectedImprovement(
        model=model, X_baseline=data.train_X,
        sampler=SobolQMCNormalSampler(torch.Size([args.mc_samples]), seed=args.seed),
    )
    selected, _ = optimize_acqf_discrete(
        acquisition, q=min(args.batch_size, len(data.candidate_indices)),
        choices=data.pool_X[data.candidate_indices], unique=True,
        max_batch_size=args.pool_batch_size,
    )
    return _selected_indices(selected, data)

#根据初始已测双目标值生成超体积参考点，兼容最大化和最小化方向。
def initial_reference(values, directions):
    signs = np.array([-1 if direction == "min" else 1 for direction in directions])
    signed = np.asarray(values) * signs
    span = np.ptp(signed, axis=0)
    scale = np.where(span > 0, span, np.maximum(np.abs(signed).max(0), 1.0))
    return (signed.min(0) - 0.1 * scale) * signs

#用 qLogNEHVI 和参考点进行双目标离散批量选点。
def multi_recommendation(data, model, args, ref_point):
    if not data.candidate_indices:
        return []
    torch.manual_seed(args.seed)
    signs = data.pool_X.new_tensor([-1 if d == "min" else 1 for d in args.directions])
    acquisition = qLogNoisyExpectedHypervolumeImprovement(
        model=model, ref_point=data.pool_X.new_tensor(ref_point) * signs,
        X_baseline=data.train_X,
        sampler=SobolQMCNormalSampler(torch.Size([args.mc_samples]), seed=args.seed),
    )
    selected, _ = optimize_acqf_discrete(
        acquisition, q=min(args.batch_size, len(data.candidate_indices)),
        choices=data.pool_X[data.candidate_indices], unique=True,
        max_batch_size=args.pool_batch_size,
    )
    return _selected_indices(selected, data)

#将全部未测候选的 sample_id、预测均值和标准差写入预测 CSV。
def save_candidate_predictions(path, data, means, stds):
    fields = ["sample_id"]
    for name in data.target_names:
        fields.extend([f"predicted_mean_{name}", f"predicted_std_{name}"])
    rows = []
    for j, i in enumerate(data.candidate_indices):
        row = {"sample_id": data.sample_ids[i]}
        for k, name in enumerate(data.target_names):
            row[f"predicted_mean_{name}"] = float(means[j, k])
            row[f"predicted_std_{name}"] = float(stds[j, k])
        rows.append(row)
    save_csv(path, fields, rows)

#从候选池复制选中行的原始配方信息，清空性能列后输出推荐 CSV
def save_recommendations(path, pool_path, data, indices):
    path = Path(path)
    fields, pool = _read_table(pool_path)
    if [row["sample_id"] for row in pool] != data.sample_ids:
        raise ValueError("Candidate pool changed since model fitting")
    target_fields = list(dict.fromkeys([*data.target_names, "logCE", "Conductivity"]))
    fields = fields + [name for name in data.target_names if name not in fields]
    rows = []
    for i in indices:
        row = dict(pool[i])
        for name in target_fields:
            if name in fields:
                row[name] = ""
        rows.append(row)
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields, existing_rows = reader.fieldnames, list(reader)
        if existing_fields != fields or existing_rows != rows:
            raise ValueError(f"Existing recommendations have changed; preserve them: {path}")
        return
    save_csv(path, fields, rows)


def recommendation_path(output, round_id):
    """Name an issued batch by its one-based round number."""
    return Path(output) / f"recommendation_{round_id}.csv"


def existing_recommendation_path(output, round_id):
    """Find an issued batch, including files from the previous naming scheme."""
    current = recommendation_path(output, round_id)
    if current.exists():
        return current
    legacy = Path(output) / ("recommendations.csv" if round_id == 1
                             else f"recommendations_{round_id}.csv")
    return legacy if legacy.exists() else current


def sync_observations(experiment_path, pool_path, output, target_names):
    """Merge returned recommendations into cumulative measured observations.

    Returns (observation_path, next_round, pending_count). A pending batch must
    be completed before another recommendation is issued.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    experiment_fields, experiments = _read_table(experiment_path)
    pool_fields, pool = _read_table(pool_path)
    if "sample_id" not in experiment_fields or "sample_id" not in pool_fields:
        raise ValueError("Both input files require sample_id")
    if any(name not in experiment_fields for name in target_names):
        raise ValueError("Selected targets are missing from experiment data")
    pool_by_id = {row["sample_id"]: row for row in pool}
    if len(pool_by_id) != len(pool):
        raise ValueError("Duplicate sample_id in candidate pool")
    performance = set(target_names) | {"logCE", "Conductivity"}
    observed_targets = [name for name in experiment_fields
                        if name in performance and name not in pool_fields]
    fields = pool_fields + observed_targets
    recommendation_fields = pool_fields + [name for name in target_names
                                           if name not in pool_fields]
    state_path = output / "observation.csv"
    if state_path.exists():
        state_fields, records = _read_table(state_path)
        if state_fields != fields:
            raise ValueError(f"Observation columns changed: {state_path}")
    else:
        records = []
        for measured in experiments:
            sample_id = measured["sample_id"]
            if sample_id not in pool_by_id:
                raise ValueError(f"Experimental sample_id absent from pool: {sample_id}")
            row = dict(pool_by_id[sample_id])
            for name in performance & set(fields):
                row[name] = measured.get(name, "").strip()
            records.append(row)
        save_csv(state_path, fields, records)
    by_id = {row["sample_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate sample_id in observation.csv")

    issued = {}
    paths = list(output.glob("recommendation_*.csv"))
    paths += list(output.glob("recommendations_*.csv"))
    legacy_first = output / "recommendations.csv"
    if legacy_first.exists():
        paths.append(legacy_first)
    for path in paths:
        if path.name == "recommendations.csv":
            round_id = 1
        else:
            match = re.fullmatch(r"recommendations?_(\d+)\.csv", path.name)
            if not match or int(match.group(1)) < 1:
                raise ValueError(f"Invalid recommendation file name: {path}")
            round_id = int(match.group(1))
        if round_id in issued:
            raise ValueError(f"Two recommendation files for round {round_id}")
        issued[round_id] = path
    if sorted(issued) != list(range(1, len(issued) + 1)):
        raise ValueError("Recommendation rounds must be consecutive")

    changed = False
    seen = set()
    pending = 0
    for round_id, path in sorted(issued.items()):
        rec_fields, batch = _read_table(path)
        if rec_fields != recommendation_fields:
            raise ValueError(f"Recommendation columns changed: {path}")
        for returned in batch:
            sample_id = returned["sample_id"]
            if sample_id in seen or sample_id not in pool_by_id:
                raise ValueError(f"Duplicate or unknown recommendation: {sample_id}")
            seen.add(sample_id)
            original = pool_by_id[sample_id]
            if any(returned[name] != original[name]
                   for name in pool_fields if name not in performance):
                raise ValueError(f"Recipe fields changed in {path}: {sample_id}")
            new_row = sample_id not in by_id
            if new_row:
                row = dict(original)
                for name in performance & set(fields):
                    row[name] = ""
            else:
                row = by_id[sample_id]
            for name in target_names:
                value = returned[name].strip()
                if value:
                    try:
                        finite = np.isfinite(float(value))
                    except ValueError as error:
                        raise ValueError(f"Invalid returned {name}: {sample_id}") from error
                    if not finite:
                        raise ValueError(f"Invalid returned {name}: {sample_id}")
                    if row[name] and not np.isclose(float(row[name]), float(value),
                                                    rtol=0, atol=1e-10):
                        raise ValueError(f"Returned {name} conflicts with observations: {sample_id}")
                    if not row[name]:
                        row[name] = value
                        changed = True
                if not row[name]:
                    pending += 1
            if new_row and any(row[name] for name in target_names):
                records.append(row)
                by_id[sample_id] = row
                changed = True
        if pending:
            if round_id != max(issued):
                raise ValueError("A later recommendation exists before earlier feedback completed")
            break
    if changed:
        save_csv(state_path, fields, records)
    return state_path, len(issued) + 1, pending


def save_csv(path, fields, rows):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fit_multi_model_data(train_X, train_Y, args):
    """Fit independent GPs for two measured objectives."""
    models = []
    for j in range(train_Y.shape[-1]):
        objective_args = copy(args)
        objective_args.minimize = args.directions[j] == "min"
        objective_args.noise_std = None if args.noise_std is None else args.noise_std[j]
        models.append(fit_model_data(train_X, train_Y[:, j:j + 1], objective_args))
    return ModelListGP(*models)


def fit_multi_model(X, records, args, target):
    ids = [int(r["pool_id"]) - 1 for r in records]
    Y = torch.tensor([[float(r[name]) for name in target] for r in records],
                     dtype=torch.double)
    return fit_multi_model_data(X[ids], Y, args)


def new_records(ids, round_id, names, raw, mean=None, std=None):
    return [dict(pool_id=i + 1, round=round_id, y="",
                 predicted_mean="" if mean is None else float(mean[j]),
                 predicted_std="" if std is None else float(std[j]),
                 **dict(zip(names, raw[i]))) for j, i in enumerate(ids)]
