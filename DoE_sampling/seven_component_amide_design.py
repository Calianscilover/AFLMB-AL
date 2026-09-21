"""Generate the seven-component amide-electrolyte DoE candidate space."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc


NAMES = ["LiDFOB", "NDFA", "TTE", "FEC", "LiNO3", "VC", "TMSP"]
POOL = ["NDFA", "TTE", "FEC", "VC", "TMSP"]
OPTIONAL = ["TTE", "FEC", "LiNO3", "VC", "TMSP"]


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def largest_remainder_round(raw_mass_g, step_g, total_mass_g):
    units = raw_mass_g / step_g
    integers = np.floor(units).astype(int)
    order = np.argsort(-(units - integers), axis=1, kind="stable")
    target_units = int(round(total_mass_g / step_g))
    for row, missing in enumerate(target_units - integers.sum(axis=1)):
        integers[row, order[row, :missing]] += 1
    return integers * step_g


def generate_candidates(components, config):
    """Sample all 32 optional-component patterns and continuous variables."""
    per_pattern_power = int(config["candidate_power"]) - len(OPTIONAL)
    uniforms = np.vstack([
        qmc.Sobol(6, scramble=True, seed=config["seed"] + pattern).random_base2(per_pattern_power)
        for pattern in range(2 ** len(OPTIONAL))
    ])
    patterns = np.repeat(np.arange(2 ** len(OPTIONAL)), 2 ** per_pattern_power)
    presence = ((patterns[:, None] >> np.arange(len(OPTIONAL))) & 1).astype(float)
    mw = components.set_index("component")["molar_mass_g_mol"]

    ratio_low, ratio_high = config["ndfa_per_lidfob_molar_ratio_bounds"]
    ratio = ratio_low + uniforms[:, 0] * (ratio_high - ratio_low)
    pool_fraction = {}
    for column, name in enumerate(["TTE", "FEC", "VC", "TMSP"], start=1):
        low, high = config["solvent_pool_mass_fraction_bounds"][name]
        pool_fraction[name] = (
            low + uniforms[:, column] * (high - low)
        ) * presence[:, OPTIONAL.index(name)]
    pool_fraction["NDFA"] = 1 - sum(pool_fraction.values())

    lino3_low, lino3_high = config["lino3_final_mass_fraction_bounds"]
    lino3_fraction = (
        lino3_low + uniforms[:, 5] * (lino3_high - lino3_low)
    ) * presence[:, OPTIONAL.index("LiNO3")]

    raw = np.zeros((len(uniforms), len(NAMES)))
    for name in POOL:
        raw[:, NAMES.index(name)] = pool_fraction[name]
    ndfa_mol = raw[:, NAMES.index("NDFA")] / mw["NDFA"]
    raw[:, NAMES.index("LiDFOB")] = ndfa_mol / ratio * mw["LiDFOB"]
    base_mass = raw.sum(axis=1)
    raw[:, NAMES.index("LiNO3")] = lino3_fraction / (1 - lino3_fraction) * base_mass
    target = raw / raw.sum(axis=1, keepdims=True)

    result = pd.DataFrame({
        "NDFA_per_LiDFOB_molar_ratio_target": ratio,
        "LiNO3_final_mass_fraction_target": lino3_fraction,
    })
    for name in POOL:
        result[f"{name}_solvent_pool_mass_fraction_target"] = pool_fraction[name]
    for column, name in enumerate(NAMES):
        result[f"{name}_mass_fraction_target"] = target[:, column]
    return result


def evaluate_candidates(candidates, components, config):
    """Round candidates to weighable masses and annotate every constraint."""
    total_g = float(config["batch_mass_g"])
    target = candidates[[f"{name}_mass_fraction_target" for name in NAMES]].to_numpy()
    masses = largest_remainder_round(target * total_g, config["mass_step_g"], total_g)
    fractions = masses / total_g
    mw = components["molar_mass_g_mol"].to_numpy()
    density = components["density_g_mL"].to_numpy()
    idx = {name: NAMES.index(name) for name in NAMES}

    pool_mass = masses[:, [idx[name] for name in POOL]].sum(axis=1)
    pool_fractions = {name: masses[:, idx[name]] / pool_mass for name in POOL}
    lidfob_mol = masses[:, idx["LiDFOB"]] / mw[idx["LiDFOB"]]
    lino3_mol = masses[:, idx["LiNO3"]] / mw[idx["LiNO3"]]
    ratio = (masses[:, idx["NDFA"]] / mw[idx["NDFA"]]) / lidfob_mol
    volume_mL = (masses / density).sum(axis=1)
    lidfob_molarity = 1000 * lidfob_mol / volume_mL
    total_salt_molarity = 1000 * (lidfob_mol + lino3_mol) / volume_mL

    ratio_low, ratio_high = config["ndfa_per_lidfob_molar_ratio_bounds"]
    molarity_low, molarity_high = config["lidfob_molarity_bounds_mol_L"]
    _, lino3_high = config["lino3_final_mass_fraction_bounds"]
    minimum = components["minimum_nonzero_mass_g"].to_numpy()
    checks = {
        "total_mass": np.isclose(masses.sum(axis=1), total_g),
        "minimum_nonzero_mass": ((masses == 0) | (masses >= minimum)).all(axis=1),
        "NDFA_LiDFOB_molar_ratio": (ratio >= ratio_low) & (ratio <= ratio_high),
        "LiNO3_final_mass_fraction": fractions[:, idx["LiNO3"]] <= lino3_high,
        "LiDFOB_molarity": (lidfob_molarity >= molarity_low) & (lidfob_molarity <= molarity_high),
    }
    for name, (low, high) in config["solvent_pool_mass_fraction_bounds"].items():
        checks[f"{name}_solvent_pool_fraction"] = (
            (pool_fractions[name] >= low - 1e-12) & (pool_fractions[name] <= high + 1e-12)
        )

    result = candidates.copy()
    result["temperature_C"] = config["temperature_C"]
    result["target_total_mass_g"] = total_g
    result["NDFA_per_LiDFOB_molar_ratio"] = ratio
    result["LiNO3_final_mass_fraction"] = fractions[:, idx["LiNO3"]]
    result["estimated_final_volume_mL"] = volume_mL
    result["estimated_LiDFOB_molarity_mol_L"] = lidfob_molarity
    result["estimated_total_lithium_salt_molarity_mol_L"] = total_salt_molarity
    for name in POOL:
        result[f"{name}_solvent_pool_mass_fraction"] = pool_fractions[name]
    for name, passed in checks.items():
        result[f"constraint_{name}_pass"] = passed
    result["constraint_all_pass"] = np.logical_and.reduce(list(checks.values()))
    result["presence_pattern"] = [
        "+".join(name for name in OPTIONAL if row[idx[name]] > 0) or "core_only"
        for row in masses
    ]
    total_moles = (masses / mw).sum(axis=1)
    for column, name in enumerate(NAMES):
        result[f"{name}_mass_g"] = masses[:, column]
        result[f"{name}_mass_fraction"] = fractions[:, column]
        result[f"{name}_mole_fraction"] = (masses[:, column] / mw[column]) / total_moles
    return result


def select_space_filling(feasible, config):
    ratio_low, ratio_high = config["ndfa_per_lidfob_molar_ratio_bounds"]
    molarity_low, molarity_high = config["lidfob_molarity_bounds_mol_L"]
    features = [
        (feasible["NDFA_per_LiDFOB_molar_ratio"].to_numpy() - ratio_low) / (ratio_high - ratio_low),
        (feasible["estimated_LiDFOB_molarity_mol_L"].to_numpy() - molarity_low) / (molarity_high - molarity_low),
    ]
    for name, (_, high) in config["solvent_pool_mass_fraction_bounds"].items():
        features.append(feasible[f"{name}_solvent_pool_mass_fraction"].to_numpy() / high)
    features.append(feasible["LiNO3_final_mass_fraction"].to_numpy() / config["lino3_final_mass_fraction_bounds"][1])
    for name in OPTIONAL:
        features.append((feasible[f"{name}_mass_g"].to_numpy() > 0).astype(float) * 0.5)
    features = np.column_stack(features)

    nearest = np.linalg.norm(features - features.mean(axis=0), axis=1)
    chosen = []
    for _ in range(int(config["selection_count"])):
        row = int(np.argmax(nearest))
        chosen.append(row)
        nearest = np.minimum(nearest, np.linalg.norm(features - features[row], axis=1))
        nearest[chosen] = -np.inf
    return feasible.iloc[chosen].reset_index(drop=True).copy()


def make_run_order(selected, config):
    rng = np.random.default_rng(config["seed"] + 1)
    rows = []
    for block, indices in enumerate(np.array_split(rng.permutation(len(selected)), config["blocks"]), 1):
        for order, row_index in enumerate(indices, 1):
            row = selected.iloc[row_index].to_dict()
            row.update({"preparation_id": f"P{len(rows) + 1:03d}", "block": block,
                        "order_in_block": order, "status": "planned_unverified"})
            rows.append(row)
    return pd.DataFrame(rows)


def main(config_path, output_dir):
    config = load_config(config_path)
    components = pd.DataFrame(config["components"]).rename(columns={"name": "component"})
    evaluated = evaluate_candidates(generate_candidates(components, config), components, config)
    mass_columns = [f"{name}_mass_g" for name in NAMES]
    feasible = evaluated.loc[evaluated["constraint_all_pass"]].drop_duplicates(mass_columns).reset_index(drop=True)
    selected = select_space_filling(feasible, config)
    selected.insert(0, "formulation_id", [f"A{row:03d}" for row in range(1, len(selected) + 1)])
    selected.insert(1, "formulation_type", "space_filling")
    runs = make_run_order(selected, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_dir / "components.csv", index=False)
    feasible.to_csv(output_dir / "feasible_candidates.csv", index=False, float_format="%.10g")
    selected.to_csv(output_dir / "selected_formulations.csv", index=False, float_format="%.10g")
    runs.to_csv(output_dir / "experiment_information_table.csv", index=False, float_format="%.10g")
    pd.DataFrame([
        {"constraint": column.removeprefix("constraint_").removesuffix("_pass"),
         "passed_candidates": int(evaluated[column].sum()),
         "failed_candidates": int((~evaluated[column]).sum())}
        for column in evaluated if column.startswith("constraint_") and column != "constraint_all_pass"
    ]).to_csv(output_dir / "constraint_screening_summary.csv", index=False)
    with (output_dir / "config_snapshot.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
    print(f"Candidates {len(evaluated)}; feasible {len(feasible)}; selected {len(selected)}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("seven_component_config.json"))
    parser.add_argument("--output", type=Path, default=root / "data/DoE/seven_component")
    args = parser.parse_args()
    main(args.config, args.output)
