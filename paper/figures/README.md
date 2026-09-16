# Reproducing the manuscript figures

Run the Python generators from the repository root in the HomiCSx development
environment:

```bash
python paper/figures/generate_architecture_figure.py
python paper/figures/generate_periodic_mesh_figure.py
python paper/figures/generate_validation_figure.py
python paper/figures/generate_paraview_data.py
```

The last command writes XDMF/HDF5 output and a final-state VTU file to the
ignored `paper/figures/paraview_data` directory. Render that VTU file with
ParaView's Python executable:

```bash
pvpython paper/figures/render_paraview_fields.py
```

The checked-in ParaView image was rendered with ParaView 6.2.0. Its field data
come from a deterministic, periodic-conforming two-phase Neo-Hookean problem at
the final simple-shear state, not from the application paper's localization
statistics.
