# AFLMB-AL

Design-of-experiments sampling and formulation data for anode-free lithium-metal battery electrolytes.

## Repository structure

- `DoE/`: sampling code and editable JSON configuration.
- `data/DoE/seven_component/`: generated seven-component candidate space, selected formulations, experiment table, component metadata, constraint summary, and configuration snapshot.

## Run

```bash
python -m DoE.seven_component_amide_design
```

The default workflow samples LiDFOB, NDFA, TTE, FEC, LiNO3, VC, and TMSP formulations and writes results to `data/DoE/seven_component/`.