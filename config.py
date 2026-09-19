"""Editable inputs for the five-component, 5 g molarity-screened DoE."""

CONFIG = {
  "seed": 20260920,
  "candidate_power": 14,
  "selection_count": 16,
  "batch_mass_g": 5.0,
  "mass_step_g": 0.0001,
  "total_salt_mass_fraction_bounds": [0.10, 0.45],
  "total_salt_molarity_bounds_mol_L": [0.5, 2.5],
  "minimum_full_distance": 0.40,
  "minimum_projected_distance": 0.18,
  "exchange_passes": 4,
  "blocks": 2,
  "independent_repeat_formulation_ids": ["B001", "B002"],
  "fixed_formulations": [
    {"formulation_id": "B001", "formulation_type": "BAFF_baseline", "basis": "mole_ratio", "components": {"LiDFOB": 1.0, "NDFA": 5.0}, "literature_solution_density_g_mL": 1.36, "literature_density_source_url": "https://assets-eu.researchsquare.com/files/rs-5047161/v1/256b3495-d635-4d2f-bcb8-579d0eea3251.pdf"},
    {"formulation_id": "B002", "formulation_type": "LHCE_baseline", "basis": "mole_ratio", "components": {"LiFSI": 1.0, "DME": 1.2, "TTE": 3.0}, "literature_solution_density_g_mL": 1.48, "literature_density_source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7936379/"}
  ],
  "components": [
    {"name": "LiFSI", "role": "lithium_salt", "cas": "171611-11-3", "smiles": "FS([N-]S(F)(=O)=O)(=O)=O.[Li+]", "formula": "F2LiNO4S2", "density_g_mL": 2.32, "density_source_url": "https://www.kishida.co.jp/product/catalog/detail/id/19757", "minimum_nonzero_mass_g": 0.005, "maximum_mass_fraction": 0.45},
    {"name": "LiDFOB", "role": "lithium_salt", "cas": "409071-16-5", "smiles": "F[B-]1(OC(C(O1)=O)=O)F.[Li+]", "formula": "C2BF2LiO4", "density_g_mL": 2.12, "density_source_url": "https://www.scbt.com/p/lithium-difluoro-oxalato-borate-409071-16-5", "minimum_nonzero_mass_g": 0.005, "maximum_mass_fraction": 0.45},
    {"name": "DME", "role": "solvent", "cas": "110-71-4", "smiles": "COCCOC", "formula": "C4H10O2", "density_g_mL": 0.867, "density_source_url": "https://www.sigmaaldrich.com/JO/en/product/sial/259527", "minimum_nonzero_mass_g": 0.020, "maximum_mass_fraction": 0.90},
    {"name": "NDFA", "role": "solvent", "cas": "1547-87-1", "smiles": "CN(C)C(=O)C(F)(F)F", "formula": "C4H6F3NO", "density_g_mL": 1.26, "density_source_url": "https://www.tcichemicals.com/MX/en/p/T3262", "minimum_nonzero_mass_g": 0.020, "maximum_mass_fraction": 0.90},
    {"name": "TTE", "role": "solvent", "cas": "16627-68-2", "smiles": "FC(F)(OCC(F)(F)C(F)F)C(F)F", "formula": "C5H4F8O", "density_g_mL": 1.54, "density_source_url": "https://www.merckmillipore.com/AE/en/product/aldrich/933961", "minimum_nonzero_mass_g": 0.020, "maximum_mass_fraction": 0.90}
  ]
}
