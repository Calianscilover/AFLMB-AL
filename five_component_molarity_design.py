"""Five-component, 5 g electrolyte design screened by estimated total-salt molarity."""
import argparse
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from design_utils import (
    _largest_remainder_round,
    farthest_point_space_filling,
    generate_candidates,
    randomize_experiments,
    validate_components,
)

NAMES = ("LiFSI", "LiDFOB", "DME", "NDFA", "TTE")


def load_components(config):
    if tuple(record["name"] for record in config["components"]) != NAMES:
        raise ValueError("This design is specific to LiFSI, LiDFOB, DME, NDFA and TTE")
    table = validate_components(config["components"])
    table["density_g_mL"] = [float(item["density_g_mL"]) for item in config["components"]]
    table["density_source_url"] = [item["density_source_url"] for item in config["components"]]
    if not np.isfinite(table["density_g_mL"]).all() or (table["density_g_mL"] <= 0).any():
        raise ValueError("All component densities must be positive and finite")
    return table


def screen_candidates(candidates, components, config, require_all=True, enforce_salt_fraction=True):
    """Apply weighing limits and an ideal-additive-volume molarity estimate."""
    total_g = float(config["batch_mass_g"])
    step_g = float(config["mass_step_g"])
    if total_g <= 0 or step_g <= 0 or not np.isclose(total_g / step_g, round(total_g / step_g)):
        raise ValueError("Batch mass must be a positive integer multiple of the weighing step")
    target_cols = [f"{name}_mass_fraction_target" for name in NAMES]
    target = candidates[target_cols].to_numpy(float)
    if np.any(target < 0) or not np.allclose(target.sum(axis=1), 1):
        raise ValueError("Target mass fractions must be nonnegative and sum to one")
    masses = _largest_remainder_round(target * total_g, step_g, total_g)
    fractions = masses / total_g
    salt = np.array([True, True, False, False, False])
    mw = components["molar_mass_g_mol"].to_numpy(float)
    density = components["density_g_mL"].to_numpy(float)
    salt_mol = (masses[:, salt] / mw[salt]).sum(axis=1)
    volume_mL = (masses / density).sum(axis=1)
    molarity = 1000 * salt_mol / volume_mL
    salt_fraction = fractions[:, salt].sum(axis=1)
    low_M, high_M = map(float, config["total_salt_molarity_bounds_mol_L"])
    if not 0 < low_M < high_M:
        raise ValueError("Molarity bounds must be positive and ordered")
    keep = np.isclose(masses.sum(axis=1), total_g)
    minimum = components["minimum_nonzero_mass_g"].to_numpy(float)
    maximum = components["maximum_mass_fraction"].to_numpy(float)
    keep &= ((masses == 0) | (masses >= minimum)).all(axis=1)
    keep &= (fractions <= maximum + 1e-12).all(axis=1)
    keep &= (salt_mol > 0) & (masses[:, ~salt].sum(axis=1) > 0)
    keep &= (molarity >= low_M) & (molarity <= high_M)
    if require_all:
        keep &= (masses > 0).all(axis=1)
    if enforce_salt_fraction:
        low_w, high_w = config["total_salt_mass_fraction_bounds"]
        keep &= (salt_fraction >= low_w) & (salt_fraction <= high_w)

    result = candidates.copy()
    result["target_total_mass_g"] = total_g
    result["total_salt_mass_fraction"] = salt_fraction
    result["total_salt_amount_mol"] = salt_mol
    result["estimated_final_volume_mL"] = volume_mL
    result["estimated_total_salt_molarity_mol_L"] = molarity
    total_moles = (masses / mw).sum(axis=1)
    for i, name in enumerate(NAMES):
        result[f"{name}_mass_g"] = masses[:, i]
        result[f"{name}_mass_fraction"] = fractions[:, i]
        result[f"{name}_mole_fraction"] = (masses[:, i] / mw[i]) / total_moles
    mass_cols = [f"{name}_mass_g" for name in NAMES]
    return result.loc[keep].drop_duplicates(mass_cols).reset_index(drop=True)


def fixed_formulations(components, config):
    rows, metadata = [], []
    mw = components["molar_mass_g_mol"].to_numpy(float)
    for item in config["fixed_formulations"]:
        ratios = np.array([item["components"].get(name, 0) for name in NAMES], dtype=float)
        if np.any(ratios < 0) or ratios.sum() <= 0:
            raise ValueError(f"Invalid fixed formulation {item['formulation_id']}")
        if item["basis"] == "mole_ratio":
            ratios *= mw
        elif item["basis"] != "mass_ratio":
            raise ValueError("Fixed basis must be mole_ratio or mass_ratio")
        fractions = ratios / ratios.sum()
        rows.append({f"{name}_mass_fraction_target": fractions[i] for i, name in enumerate(NAMES)})
        metadata.append({"formulation_id": item["formulation_id"],
                         "formulation_type": item["formulation_type"],
                         "literature_solution_density_g_mL": item["literature_solution_density_g_mL"],
                         "literature_density_source_url": item["literature_density_source_url"]})
    fixed = screen_candidates(pd.DataFrame(rows), components, config,
                              require_all=False, enforce_salt_fraction=False)
    if len(fixed) != len(rows):
        raise ValueError("A baseline violates weighing or molarity constraints")
    fixed = pd.concat([pd.DataFrame(metadata), fixed], axis=1)
    fixed["literature_density_based_molarity_mol_L"] = (
        1000 * fixed["total_salt_amount_mol"] * fixed["literature_solution_density_g_mL"]
        / fixed["target_total_mass_g"]
    )
    return fixed


def main(config_path, output_dir):
    config = runpy.run_path(str(config_path))["CONFIG"]
    components = load_components(config)
    candidates = generate_candidates(components, config)
    feasible = screen_candidates(candidates, components, config)
    fixed = fixed_formulations(components, config)
    new_count = int(config["selection_count"]) - len(fixed)
    selected = farthest_point_space_filling(feasible, components, config, new_count, fixed)
    selected.insert(0, "formulation_id", [f"F{i:03d}" for i in range(1, new_count + 1)])
    selected.insert(1, "formulation_type", "space_filling")
    formulations = pd.concat([fixed, selected], ignore_index=True)
    runs = randomize_experiments(formulations, config)
    mass_cols = [f"{name}_mass_g" for name in NAMES]
    fraction_cols = [f"{name}_mass_fraction" for name in NAMES]
    report_cols = ["formulation_id", "formulation_type", "target_total_mass_g",
                   "total_salt_mass_fraction", "total_salt_amount_mol",
                   "estimated_final_volume_mL", "estimated_total_salt_molarity_mol_L",
                   "literature_solution_density_g_mL", "literature_density_based_molarity_mol_L",
                   "literature_density_source_url",
                   *mass_cols, *fraction_cols]
    run_cols = ["preparation_id", "formulation_id", "preparation_replicate", "block",
                "order_in_block", "status", *report_cols[2:]]
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in {
        "components.csv": components,
        "feasible_candidates.csv": feasible,
        "selected_formulations.csv": formulations,
        "formulation_test_set.csv": formulations[report_cols],
        "experimental_run_order.csv": runs[run_cols],
    }.items():
        table.to_csv(output_dir / filename, index=False, float_format="%.10g")
    print(f"Feasible: {len(feasible)}; selected: {len(formulations)}; runs: {len(runs)}; "
          f"estimated molarity: {formulations['estimated_total_salt_molarity_mol_L'].min():.3f}"
          f"–{formulations['estimated_total_salt_molarity_mol_L'].max():.3f} M")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.py"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    main(args.config, args.output)
