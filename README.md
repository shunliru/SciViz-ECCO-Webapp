# ECCO Thermohaline Web App

Interactive scientific visualization web app for exploring ECCO ocean simulation data using Trame, PyVista, VTK, and OpenVisus.

## Features

* Salinity isosurface visualization
* Temperature anomaly coloring on salinity surfaces
* Positive vertical velocity hotspot extraction
* Adjustable salinity layer percentiles
* Adjustable opacity controls
* Interactive timestep loading
* Plot information panel in the UI

## Environment Setup

Create and activate a Conda environment:

```bash
conda create -n ecco-webapp python=3.11
conda activate ecco-webapp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python app.py --port 8081
```

Then open the app in your browser:

```text
http://localhost:8081
```

## Notes

This app loads ECCO data through OpenVisus. Large spatial regions or high-resolution reads may require significant memory and long loading time.

For development, use a small spatial region and reduced data quality (availale quality levels are -12, -9, -6, -3, and 0 with -12 being the worst quality and 0 being the best quality).

