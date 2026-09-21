import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from DoE_sampling.seven_component_amide_design import main


ROOT = Path(__file__).resolve().parents[1]


class SevenComponentDesignTests(unittest.TestCase):
    def test_generated_space_obeys_constraints(self):
        config_path = ROOT / "DoE_sampling/seven_component_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        with TemporaryDirectory() as directory:
            main(config_path, Path(directory))
            feasible = pd.read_csv(Path(directory) / "feasible_candidates.csv")
            selected = pd.read_csv(Path(directory) / "selected_formulations.csv")
            self.assertEqual(feasible["presence_pattern"].nunique(), 32)
            self.assertEqual(len(selected), config["selection_count"])
            self.assertTrue(selected["constraint_all_pass"].all())
            self.assertTrue(selected["NDFA_per_LiDFOB_molar_ratio"].between(1, 5).all())
            self.assertTrue(selected["estimated_LiDFOB_molarity_mol_L"].between(1, 2.5).all())
            mass_columns = [f"{item['name']}_mass_g" for item in config["components"]]
            np.testing.assert_allclose(selected[mass_columns].sum(axis=1), 5.0)


if __name__ == "__main__":
    unittest.main()
