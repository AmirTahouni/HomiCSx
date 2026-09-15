# HomiCSx--Abaqus finite-strain comparison

This compact dataset records nine outcome-blind, geometry-matched comparisons at macroscopic deformation gradient `diag(1.5, 1.0, 1.0)`. It covers three particle counts and three clearance-to-radius levels. Abaqus is an independent comparison implementation, not ground truth.

The retained verification quantities are macroscopic energy, the normal components `P11`, `P22`, and `P33` of macroscopic first Piola--Kirchhoff stress, and mean deformation Jacobian `J`. The acceptance limit is 1% for macroscopic energy and stresses. The mean-J comparison uses a tighter 0.000001% gate because both solvers impose the same macroscopic deformation with determinant 1.5.

Run the comparison without an Abaqus installation:

```console
python -m validation.abaqus.compare_reference
```

The command recomputes every relative difference from `reference_results.csv`; stored pass flags are deliberately not used. Expected maximum absolute differences are 0.106% for macroscopic energy, 0.137% for `P11`, 0.0051% for `P22`, 0.111% for `P33`, and approximately 1.42e-8% for mean `J`.

Raw Abaqus ODB files, dense field exports, and research-specific localization statistics are intentionally excluded. `metadata.json` records the material, loading, mesh, selection, and source provenance needed to interpret this extract. This suite demonstrates macroscopic agreement for the selected cases; it does not establish pointwise field identity, general mesh independence, or validity for every HomiCSx feature.
