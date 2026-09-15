# HomiCSx--Abaqus finite-strain comparison

This compact dataset records nine outcome-blind, geometry-matched comparisons at macroscopic deformation gradient `diag(1.5, 1.0, 1.0)`. It covers three particle counts and three clearance-to-radius levels. Abaqus is an independent comparison implementation, not ground truth.

The retained primary metrics are macroscopic energy, macroscopic first Piola--Kirchhoff stress `P11`, reference-area-weighted matrix energy Q99, and reference-area-weighted matrix maximum-principal-Biot-stress Q99. The acceptance limits were fixed before the selected Abaqus responses were inspected: 1% for macroscopic metrics and 3% for Q99 metrics.

Run the comparison without an Abaqus installation:

```console
python -m validation.abaqus.compare_reference
```

The command recomputes every relative difference from `reference_results.csv`; stored pass flags are deliberately not used. Expected maximum absolute differences are 0.106% for macroscopic energy, 0.137% for `P11`, 0.516% for energy Q99, and 2.052% for principal-Biot Q99.

Raw Abaqus ODB files and dense field exports are intentionally excluded because they are large, proprietary-format, and unnecessary for rerunning the numerical comparison. `metadata.json` records the material, loading, mesh, selection, and source provenance needed to interpret this extract. This suite demonstrates agreement for the selected cases and metrics; it does not establish pointwise field identity, general mesh independence, or validity for every HomiCSx feature.
