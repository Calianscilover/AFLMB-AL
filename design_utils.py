"""Shared sampling and space-filling utilities for the five-component design."""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import beta, qmc

ROLES = {"lithium_salt", "solvent"}


def validate_components(records, minimum_count=5, maximum_count=10):
    """Validate component identities and split them into salt and solvent groups."""
    if not minimum_count <= len(records) <= maximum_count:
        raise ValueError(f"Component count must be between {minimum_count} and {maximum_count}")
    if len({record["name"] for record in records}) != len(records):
        raise ValueError("Component names must be unique")
    rows = []
    for record in records:
        if record["role"] not in ROLES:
            raise ValueError(f"Invalid role for {record['name']}: {record['role']}")
        molecule = Chem.MolFromSmiles(record["smiles"])
        if molecule is None:
            raise ValueError(f"Invalid SMILES for {record['name']}")
        formula = rdMolDescriptors.CalcMolFormula(molecule)
        if formula != record["formula"] or Chem.GetFormalCharge(molecule) != 0:
            raise ValueError(f"Formula or net-charge mismatch for {record['name']}")
        rows.append({
            "component": record["name"],
            "role": record["role"],
            "cas": record.get("cas", ""),
            "canonical_smiles": Chem.MolToSmiles(molecule),
            "formula": formula,
            "molar_mass_g_mol": Descriptors.MolWt(molecule),
            "minimum_nonzero_mass_g": float(record["minimum_nonzero_mass_g"]),
            "maximum_mass_fraction": float(record.get("maximum_mass_fraction", 1.0)),
        })
    table = pd.DataFrame(rows)
    if not ROLES.issubset(set(table["role"])):
        raise ValueError("At least one lithium salt and one solvent are required")
    return table


def _simplex_from_uniforms(uniforms, size):
    """Transform size-1 uniforms into a uniform point on a size-simplex."""
    if size == 1:
        return np.ones((len(uniforms), 1))
    remainder = np.ones(len(uniforms))
    output = np.empty((len(uniforms), size))
    for index in range(size - 1):
        fraction = beta.ppf(np.clip(uniforms[:, index], 1e-12, 1 - 1e-12), 1, size - index - 1)
        output[:, index] = remainder * fraction
        remainder -= output[:, index]
    output[:, -1] = remainder
    return output


def generate_candidates(components, config):
    """Generate scrambled-Sobol candidates in the K-1 dimensional design space."""
    salt_mask = components["role"].eq("lithium_salt").to_numpy()
    salt_count = int(salt_mask.sum())
    solvent_count = len(components) - salt_count
    dimension = 1 + max(salt_count - 1, 0) + max(solvent_count - 1, 0)
    uniforms = qmc.Sobol(dimension, scramble=True, seed=config["seed"]).random_base2(config["candidate_power"])
    salt_low, salt_high = config["total_salt_mass_fraction_bounds"]
    total_salt = salt_low + uniforms[:, 0] * (salt_high - salt_low)
    cursor = 1
    salt_composition = _simplex_from_uniforms(uniforms[:, cursor:cursor + salt_count - 1], salt_count)
    cursor += salt_count - 1
    solvent_composition = _simplex_from_uniforms(uniforms[:, cursor:cursor + solvent_count - 1], solvent_count)

    fractions = np.empty((len(uniforms), len(components)))
    fractions[:, salt_mask] = total_salt[:, None] * salt_composition
    fractions[:, ~salt_mask] = (1 - total_salt)[:, None] * solvent_composition
    result = pd.DataFrame({"total_salt_mass_fraction_target": total_salt})
    for index, name in enumerate(components["component"]):
        result[f"{name}_mass_fraction_target"] = fractions[:, index]
    return result


def _largest_remainder_round(raw_mass_g, step_g, total_mass_g):
    units = raw_mass_g / step_g
    integers = np.floor(units).astype(int)
    ranking = np.argsort(-(units - integers), axis=1, kind="stable")
    target_units = int(round(total_mass_g / step_g))
    for row_index, missing in enumerate(target_units - integers.sum(axis=1)):
        integers[row_index, ranking[row_index, :missing]] += 1
    return integers * step_g


def _distance_features(table, components, config):
    names = components["component"].tolist()
    fractions = table[[f"{name}_mass_fraction" for name in names]].to_numpy(float)
    salt_mask = components["role"].eq("lithium_salt").to_numpy()
    salt_total = fractions[:, salt_mask].sum(axis=1)
    salt_low, salt_high = config["total_salt_mass_fraction_bounds"]
    loading = ((salt_total - salt_low) / (salt_high - salt_low))[:, None]
    salt_composition = fractions[:, salt_mask] / salt_total[:, None]
    solvent_total = fractions[:, ~salt_mask].sum(axis=1)
    solvent_composition = fractions[:, ~salt_mask] / solvent_total[:, None]
    return np.column_stack([
        loading,
        np.sqrt(salt_composition) / np.sqrt(2),
        np.sqrt(solvent_composition) / np.sqrt(2),
    ])


def farthest_point_space_filling(candidates, components, config, count, fixed=None):
    """Select dispersed formulations in full space and, optionally, a fixed 2D PCA view."""
    if count < 1 or count > len(candidates):
        raise ValueError("Selection count must fit the feasible candidate pool")
    features = _distance_features(candidates, components, config)
    fixed_features = np.empty((0, features.shape[1]))
    if fixed is not None and len(fixed):
        fixed_features = _distance_features(fixed, components, config)
    minimum_2d = float(config.get("minimum_projected_distance", 0))
    minimum_full = float(config.get("minimum_full_distance", 0))
    if minimum_2d <= 0:
        if len(fixed_features):
            nearest = np.linalg.norm(features[:, None, :] - fixed_features[None, :, :], axis=2).min(axis=1)
        else:
            center = np.mean(features, axis=0)
            nearest = np.linalg.norm(features - center, axis=1)
        chosen = []
        for _ in range(count):
            index = int(np.argmax(nearest))
            chosen.append(index)
            nearest = np.minimum(nearest, np.linalg.norm(features - features[index], axis=1))
            nearest[chosen] = -np.inf
        return candidates.iloc[chosen].reset_index(drop=True).copy()

    if minimum_full <= 0:
        raise ValueError("minimum_full_distance must be positive when 2D separation is requested")
    center = features.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(features - center, full_matrices=False)
    projection = (features - center) @ right_vectors[:2].T
    fixed_projection = (fixed_features - center) @ right_vectors[:2].T
    if len(fixed_features):
        nearest_full = np.linalg.norm(features[:, None, :] - fixed_features[None, :, :], axis=2).min(axis=1)
        nearest_2d = np.linalg.norm(projection[:, None, :] - fixed_projection[None, :, :], axis=2).min(axis=1)
    else:
        nearest_full = np.full(len(features), np.inf)
        nearest_2d = np.full(len(features), np.inf)

    chosen = []
    for _ in range(count):
        eligible = (nearest_full >= minimum_full) & (nearest_2d >= minimum_2d)
        eligible[chosen] = False
        if not np.any(eligible):
            raise ValueError("Distance constraints cannot fit the requested number of formulations")
        # Keep full-space separation as a hard floor; spread the visible 2D map.
        index = int(np.argmax(np.where(eligible, nearest_2d, -np.inf)))
        chosen.append(index)
        nearest_full = np.minimum(nearest_full, np.linalg.norm(features - features[index], axis=1))
        nearest_2d = np.minimum(nearest_2d, np.linalg.norm(projection - projection[index], axis=1))

    def minimum_pair_distance(indices, points):
        selected_points = np.vstack([points[0], points[1][indices]])
        if len(selected_points) < 2:
            return np.inf
        differences = selected_points[:, None, :] - selected_points[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        np.fill_diagonal(distances, np.inf)
        return float(distances.min())

    # Exchange selected points to improve actual high-dimensional maximin distance.
    for _ in range(int(config.get("exchange_passes", 4))):
        improved = False
        for position in range(count):
            others = chosen[:position] + chosen[position + 1:]
            full_references = np.vstack([fixed_features, features[others]])
            projected_references = np.vstack([fixed_projection, projection[others]])
            full_distance = np.linalg.norm(features[:, None, :] - full_references[None, :, :], axis=2).min(axis=1)
            projected_distance = np.linalg.norm(
                projection[:, None, :] - projected_references[None, :, :], axis=2
            ).min(axis=1)
            eligible = (full_distance >= minimum_full) & (projected_distance >= minimum_2d)
            eligible[others] = False
            if not np.any(eligible):
                continue
            replacement = int(np.argmax(np.where(eligible, full_distance, -np.inf)))
            current_minimum = minimum_pair_distance(chosen, (fixed_features, features))
            proposal = others + [replacement]
            proposed_minimum = minimum_pair_distance(proposal, (fixed_features, features))
            if proposed_minimum > current_minimum + 1e-8:
                chosen[position] = replacement
                improved = True
        if not improved:
            break

    if minimum_pair_distance(chosen, (fixed_features, features)) < minimum_full - 1e-10:
        raise AssertionError("Selected formulations violate the full-space minimum distance")
    if minimum_pair_distance(chosen, (fixed_projection, projection)) < minimum_2d - 1e-10:
        raise AssertionError("Selected formulations violate the projected minimum distance")
    return candidates.iloc[chosen].reset_index(drop=True).copy()


def randomize_experiments(formulations, config):
    """Create independent repeats, balance blocks, and randomize within blocks."""
    repeat_ids = config.get("independent_repeat_formulation_ids", [])
    if not set(repeat_ids).issubset(set(formulations["formulation_id"])):
        raise ValueError("An independent-repeat formulation ID is missing")
    records = []
    for _, row in formulations.iterrows():
        records.append((row, 1))
        if row["formulation_id"] in repeat_ids:
            records.append((row, 2))
    rng = np.random.default_rng(config["seed"] + 1)
    block_count = int(config.get("blocks", 1))
    groups = [[] for _ in range(block_count)]
    for formulation_id in repeat_ids:
        repeats = [item for item in records if item[0]["formulation_id"] == formulation_id]
        for index, item in enumerate(repeats):
            groups[index % block_count].append(item)
    ordinary = [item for item in records if item[0]["formulation_id"] not in repeat_ids]
    for index in rng.permutation(len(ordinary)):
        groups[min(range(block_count), key=lambda item: len(groups[item]))].append(ordinary[index])
    rows = []
    for block, group in enumerate(groups, 1):
        for order, index in enumerate(rng.permutation(len(group)), 1):
            formulation, replicate = group[index]
            row = formulation.to_dict()
            row.update({
                "preparation_id": f"P{len(rows) + 1:03d}",
                "preparation_replicate": replicate,
                "block": block,
                "order_in_block": order,
                "status": "planned_unverified",
            })
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["block", "order_in_block"]).reset_index(drop=True)
