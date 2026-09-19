import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from five_component_molarity_design import load_components, main  # noqa: E402


class FiveComponentDesignTests(unittest.TestCase):
    def test_five_gram_design_and_molarity(self):
        config_path = ROOT / "config.py"
        config = runpy.run_path(str(config_path))["CONFIG"]
        components = load_components(config)
        with TemporaryDirectory() as directory:
            main(config_path, Path(directory))
            table = pd.read_csv(Path(directory) / "formulation_test_set.csv")
            runs = pd.read_csv(Path(directory) / "experimental_run_order.csv")
            self.assertEqual(len(table), 16)
            self.assertEqual(len(runs), 18)
            self.assertEqual(runs.groupby("block").size().to_dict(), {1: 9, 2: 9})
            self.assertTrue(table["estimated_total_salt_molarity_mol_L"].between(0.5, 2.5).all())

            names = components["component"].tolist()
            masses = table[[f"{name}_mass_g" for name in names]].to_numpy(float)
            np.testing.assert_allclose(masses.sum(axis=1), config["batch_mass_g"])
            np.testing.assert_allclose(masses / config["mass_step_g"],
                                       np.rint(masses / config["mass_step_g"]))
            fraction = table[[f"{name}_mass_fraction" for name in names]].to_numpy(float)
            np.testing.assert_allclose(fraction, masses / config["batch_mass_g"])

            density = components["density_g_mL"].to_numpy(float)
            mw = components["molar_mass_g_mol"].to_numpy(float)
            volume_mL = (masses / density).sum(axis=1)
            salt_mol = (masses[:, :2] / mw[:2]).sum(axis=1)
            np.testing.assert_allclose(table["estimated_final_volume_mL"], volume_mL)
            np.testing.assert_allclose(table["estimated_total_salt_molarity_mol_L"],
                                       1000 * salt_mol / volume_mL)
            self.assertTrue(table.loc[2:, "literature_solution_density_g_mL"].isna().all())
            self.assertAlmostEqual(table.loc[0, "literature_density_based_molarity_mol_L"],
                                   1.6, delta=0.02)
            self.assertAlmostEqual(table.loc[1, "literature_density_based_molarity_mol_L"],
                                   1.49, delta=0.02)


if __name__ == "__main__":
    unittest.main()
