# Seven-component DoE sampling

This module generates the LiDFOB/NDFA/TTE/FEC/LiNO3/VC/TMSP formulation space.

```bash
python -m DoE_sampling.seven_component_amide_design
```

The default output directory is `data/DoE/seven_component/`. The generator explicitly samples all 32 presence/absence patterns for TTE, FEC, LiNO3, VC, and TMSP, then applies the configured molar-ratio, composition, weighing, and LiDFOB-molarity constraints.
