"""Convert the experimental run table into fixed-slot molecular ML inputs."""

import argparse
import sys
from pathlib import Path

import pandas as pd


SMILES = {
    "LiDFOB": "F[B-]1(OC(C(O1)=O)=O)F.[Li+]",
    "LiNO3": "[Li+].[O-][N+](=O)[O-]",
    "NDFA": "CN(C)C(=O)C(F)(F)F",
    "TTE": "FC(F)(OCC(F)(F)C(F)F)C(F)F",
    "FEC": "FC1COC(=O)O1",
    "VC": "O=C1OC=CO1",
    "TMSP": "C[Si](C)(C)OP(O[Si](C)(C)C)O[Si](C)(C)C",
}
SOLVENT_SLOTS = ["NDFA", "TTE", "FEC", "VC", "TMSP"]
LINO3_MOLAR_MASS_G_MOL = 68.945


def convert(table):
    output = table[[
        "preparation_id", "formulation_id", "formulation_type", "block",
        "order_in_block", "status", "temperature_C", "target_total_mass_g",
        "presence_pattern", "NDFA_per_LiDFOB_molar_ratio",
        "estimated_final_volume_mL", "estimated_total_lithium_salt_molarity_mol_L",
    ]].copy()
    output["salt_smiles"] = SMILES["LiDFOB"]
    output["salt_conc"] = table["estimated_LiDFOB_molarity_mol_L"]
    output["salt_mass_g"] = table["LiDFOB_mass_g"]
    output["salt_mass_fraction"] = table["LiDFOB_mass_fraction"]
    output["salt_2_smiles"] = SMILES["LiNO3"]
    output["salt_2_conc"] = (
        1000 * table["LiNO3_mass_g"]
        / LINO3_MOLAR_MASS_G_MOL
        / table["estimated_final_volume_mL"]
    )
    output["salt_2_mass_g"] = table["LiNO3_mass_g"]
    output["salt_2_mass_fraction"] = table["LiNO3_mass_fraction"]
    output["solvent_ratio_basis"] = "salt_free_liquid_pool_mass_fraction"
    for slot, name in enumerate(SOLVENT_SLOTS, 1):
        output[f"solv_{slot}_name"] = name
        output[f"solv_{slot}_smiles"] = SMILES[name]
        output[f"solv_{slot}_ratio"] = table[f"{name}_solvent_pool_mass_fraction"]
        output[f"solv_{slot}_mass_g"] = table[f"{name}_mass_g"]
        output[f"solv_{slot}_mass_fraction"] = table[f"{name}_mass_fraction"]
    return output


def main(input_path, output_path):
    source = sys.stdin if str(input_path) == "-" else input_path
    result = convert(pd.read_csv(source))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, float_format="%.10g")
    print(f"Converted {len(result)} experimental rows to {output_path}")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=root / "data/DoE/seven_component/experiment_information_table.csv"
    )
    parser.add_argument(
        "--output", type=Path,
        default=root / "data/DoE/seven_component/experiment_information_ml.csv",
    )
    arguments = parser.parse_args()
    main(arguments.input, arguments.output)
