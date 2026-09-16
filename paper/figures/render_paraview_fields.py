"""Render the manuscript field figure with ParaView's Python runtime.

Generate the XDMF input first with ``generate_paraview_data.py``, then run::

    pvpython paper/figures/render_paraview_fields.py
"""

from pathlib import Path

from paraview.simple import (  # type: ignore[import-not-found]
    ColorBy,
    CreateLayout,
    CreateView,
    GetAnimationScene,
    GetColorTransferFunction,
    GetOpacityTransferFunction,
    GetScalarBar,
    SaveScreenshot,
    SetActiveView,
    Show,
    Text,
    XMLUnstructuredGridReader,
)


HERE = Path(__file__).resolve().parent
reader = XMLUnstructuredGridReader(
    FileName=[str(HERE / "paraview_data" / "simple_shear_final.vtu")]
)
scene = GetAnimationScene()

layout = CreateLayout(name="Field comparison")
layout.SplitHorizontal(0, 0.5)
views = []
for index, (field, title) in enumerate(
    (("von_Mises", "(a) von Mises stress"), ("energy_density", "(b) strain-energy density"))
):
    view = CreateView("RenderView")
    view.UseColorPaletteForBackground = 0
    view.Background = [1.0, 1.0, 1.0]
    view.OrientationAxesVisibility = 0
    view.InteractionMode = "2D"
    layout.AssignView(index + 1, view)
    SetActiveView(view)
    display = Show(reader, view)
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.25, 0.25, 0.25]
    display.LineWidth = 0.35
    ColorBy(display, ("CELLS", field))
    display.RescaleTransferFunctionToDataRange(True, False)
    lut = GetColorTransferFunction(field)
    lut.ApplyPreset("Viridis", True)
    opacity = GetOpacityTransferFunction(field)
    opacity.RescaleTransferFunction(lut.RGBPoints[0], lut.RGBPoints[-4])
    display.SetScalarBarVisibility(view, True)
    scalar_bar = GetScalarBar(lut, view)
    scalar_bar.Title = "von Mises" if field == "von_Mises" else "Energy density"
    scalar_bar.ComponentTitle = ""
    scalar_bar.TitleColor = [0.1, 0.1, 0.1]
    scalar_bar.LabelColor = [0.1, 0.1, 0.1]
    scalar_bar.Orientation = "Horizontal"
    scalar_bar.WindowLocation = "Lower Center"
    scalar_bar.ScalarBarLength = 0.42
    scalar_bar.ScalarBarThickness = 10
    scalar_bar.TitleFontSize = 10
    scalar_bar.LabelFontSize = 8
    text = Text(registrationName=f"Title {index}")
    text.Text = title
    text_display = Show(text, view)
    text_display.WindowLocation = "Upper Center"
    text_display.Color = [0.05, 0.05, 0.05]
    text_display.FontSize = 10
    view.ResetCamera()
    view.CameraParallelProjection = 1
    views.append(view)

SaveScreenshot(
    str(HERE / "paraview_fields.png"),
    layout,
    ImageResolution=[2100, 1050],
    TransparentBackground=0,
)
print(f"Wrote {HERE / 'paraview_fields.png'}")
