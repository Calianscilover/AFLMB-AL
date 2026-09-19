"""Plot composition distances without assuming any measured properties."""
import argparse
import runpy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from design_utils import _distance_features


def composition_features(formulations, components, config):
    """Match the loading + within-group composition metric used by design_utils."""
    names = components["component"].tolist()
    mass_columns = [f"{name}_mass_g" for name in names]
    masses = formulations[mass_columns].to_numpy(float)
    if not np.isfinite(masses).all() or np.any(masses < 0) or np.any(masses.sum(axis=1) <= 0):
        raise ValueError("Component masses must be finite and nonnegative")
    fraction = masses / masses.sum(axis=1, keepdims=True)
    salt_mask = components["role"].eq("lithium_salt").to_numpy()
    salt_total = fraction[:, salt_mask].sum(axis=1)
    solvent_total = fraction[:, ~salt_mask].sum(axis=1)
    if np.any(salt_total <= 0) or np.any(solvent_total <= 0):
        raise ValueError("Every formulation needs salt and solvent")
    fraction_table = formulations.copy()
    for index, name in enumerate(names):
        fraction_table[f"{name}_mass_fraction"] = fraction[:, index]
    return _distance_features(fraction_table, components, config), salt_total


def classical_mds(distances):
    """Return a deterministic metric 2D embedding and its distance correlation."""
    count = len(distances)
    centered = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centered @ (distances ** 2) @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    coordinates = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    # Fix eigenvector sign for reproducible visual orientation.
    for dimension in range(2):
        if coordinates[np.argmax(np.abs(coordinates[:, dimension])), dimension] < 0:
            coordinates[:, dimension] *= -1
    original = squareform(distances)
    projected = pdist(coordinates)
    correlation = float(np.corrcoef(original, projected)[0, 1])
    variance = float(np.maximum(eigenvalues[:2], 0).sum() / np.maximum(eigenvalues, 0).sum())
    return coordinates, correlation, variance


def plot(input_csv, components_csv, feasible_csv, config_path, output_directory):
    formulations = pd.read_csv(input_csv)
    components = pd.read_csv(components_csv)
    feasible = pd.read_csv(feasible_csv)
    config = runpy.run_path(str(config_path))["CONFIG"]
    names = components["component"].tolist()
    expected = [f"{name}_mass_g" for name in names]
    missing = [column for column in ["formulation_id", "formulation_type", *expected] if column not in formulations]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    features, salt_total = composition_features(formulations, components, config)
    distances = squareform(pdist(features, metric="euclidean"))
    pool_features = _distance_features(feasible, components, config)
    center = pool_features.mean(axis=0)
    _, singular_values, right_vectors = np.linalg.svd(pool_features - center, full_matrices=False)
    coordinates = (features - center) @ right_vectors[:2].T
    correlation = float(np.corrcoef(squareform(distances), pdist(coordinates))[0, 1])
    variance = float((singular_values[:2] ** 2).sum() / (singular_values ** 2).sum())
    minimum_full = float(pdist(features).min())
    minimum_2d = float(pdist(coordinates).min())
    identifiers = formulations["formulation_id"].tolist()
    baseline = formulations["formulation_type"].str.contains("baseline", case=False).to_numpy()

    output_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(distances, index=identifiers, columns=identifiers).rename_axis("formulation_id").to_csv(
        output_directory / "composition_distance_matrix.csv", float_format="%.8f"
    )
    pd.DataFrame({
        "formulation_id": identifiers,
        "pca_1": coordinates[:, 0],
        "pca_2": coordinates[:, 1],
        "total_salt_mass_fraction": salt_total,
        "formulation_type": formulations["formulation_type"],
    }).to_csv(output_directory / "composition_pca_coordinates.csv", index=False, float_format="%.8f")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, (map_axis, matrix_axis) = plt.subplots(1, 2, figsize=(14, 6.8), layout="constrained")
    figure.suptitle("Electrolyte formulation distance", fontsize=16, fontweight="semibold")

    palette = plt.get_cmap("viridis")
    salt_low, salt_high = config["total_salt_mass_fraction_bounds"]
    normalization = matplotlib.colors.Normalize(vmin=salt_low, vmax=salt_high)
    map_axis.scatter(
        coordinates[~baseline, 0], coordinates[~baseline, 1],
        c=salt_total[~baseline], cmap=palette, norm=normalization,
        s=100, linewidth=0.7, edgecolor="black", zorder=3,
    )
    map_axis.scatter(
        coordinates[baseline, 0], coordinates[baseline, 1],
        c=salt_total[baseline], cmap=palette, norm=normalization,
        s=230, marker="*", linewidth=0.8, edgecolor="black", zorder=4,
    )
    for index, identifier in enumerate(identifiers):
        map_axis.annotate(
            identifier, coordinates[index], xytext=(6, 5),
            textcoords="offset points", fontsize=9,
        )
    map_axis.set_title(f"2D design projection  |  min gap {minimum_2d:.2f}")
    map_axis.set_xlabel("PCA 1 (composition-distance units)")
    map_axis.set_ylabel("PCA 2 (composition-distance units)")
    map_axis.set_aspect("equal", adjustable="datalim")
    map_axis.grid(alpha=0.15)
    map_axis.margins(0.18)
    colorbar = figure.colorbar(matplotlib.cm.ScalarMappable(norm=normalization, cmap=palette), ax=map_axis, shrink=0.76)
    colorbar.set_label("Total salt mass fraction")

    image = matrix_axis.imshow(distances, cmap="magma", vmin=0, vmax=np.max(distances), interpolation="nearest")
    matrix_axis.set_title("Exact pairwise composition distance")
    matrix_axis.set_xticks(range(len(identifiers)), identifiers, rotation=90)
    matrix_axis.set_yticks(range(len(identifiers)), identifiers)
    matrix_axis.tick_params(length=0, labelsize=8)
    matrix_axis.set_xlabel("Formulation ID")
    matrix_axis.set_ylabel("Formulation ID")
    colorbar_matrix = figure.colorbar(image, ax=matrix_axis, shrink=0.76)
    colorbar_matrix.set_label("Distance (dimensionless)")

    figure.savefig(output_directory / "formulation_distances.png", dpi=220, bbox_inches="tight")
    figure.savefig(output_directory / "formulation_distances.pdf", bbox_inches="tight")
    plt.close(figure)
    print(
        f"Plotted {len(formulations)} formulations; full minimum distance = {minimum_full:.3f}; "
        f"2D minimum distance = {minimum_2d:.3f}; 2D distance correlation = {correlation:.3f}; "
        f"PCA variance = {variance:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/formulation_test_set.csv"))
    parser.add_argument("--components", type=Path, default=Path("outputs/components.csv"))
    parser.add_argument("--feasible", type=Path, default=Path("outputs/feasible_candidates.csv"))
    parser.add_argument("--config", type=Path, default=Path("config.py"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    arguments = parser.parse_args()
    plot(arguments.input, arguments.components, arguments.feasible, arguments.config, arguments.output)
