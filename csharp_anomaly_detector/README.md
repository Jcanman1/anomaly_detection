# C# Anomaly Detector (minimal port)

This is a minimal, runnable .NET console application that implements the core anomaly detection behavior from the Python script in the repository:

- Load one or more reference images (Sample 1) and compute LAB mean/std from valid pixels (white background removed, optional border erosion).
- Load input images and compute a Mahalanobis-like distance in LAB space per pixel.
- Threshold pixels by sensitivity to mark anomalies, apply simple morphology, and save a highlighted image and mask.

Prerequisites
- .NET 7 SDK installed (dotnet CLI)

Build and run (PowerShell):

```powershell
cd csharp_anomaly_detector
dotnet restore
# Example: one reference and one input PNG
dotnet run -- --ref "..\\Assets\\acceptImages\\ref.png" --input "..\\Assets\\acceptImages\\test.png" --sensitivity 2.5 --border 1 --outdir ..\\out
```

Notes and assumptions
- This is a minimal CLI port (no GUI). It focuses on the detection/highlighting core.
- Supports common image formats via ImageSharp (PNG, JPG, and TIFF with the ImageSharp TIFF package).
- Uses an sRGB → CIE Lab conversion implemented in code (D65 white point).
- Morphological operations are simple iterative dilation/erosion on boolean masks.

Next steps (I can implement if you want):
- Add a WinForms or WPF GUI that mirrors the Python Tk UI (buttons, sample slots, preview thumbnails, sensitivity sliders).
- Add caching of distances, support multi-frame TIFF channel extraction, PCA/LDA plotting with Plotly or OxyPlot.
- Port the PCA/LDA/ellipsoid interactive controls into a .NET UI.

Tell me whether you want the minimal CLI (already created) or prefer I continue and implement a GUI port next.
