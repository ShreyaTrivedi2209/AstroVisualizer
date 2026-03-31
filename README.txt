AstroVisualizer

An interactive web app for exploring astronomical catalogs from [VizieR](https://vizier.cds.unistra.fr/), built with Dash and Plotly.

---

## Features

- Search any VizieR catalog by keyword or enter a catalog ID directly
- Quick-load presets for popular catalogs like Hipparcos, Gaia DR3, 2MASS, SDSS, and more
- Query objects within a sky region by RA, Dec, and radius
- Auto-computes derived quantities like distance, temperature, luminosity, and color indices
- Visualize data as scatter plots, HR diagrams, 3D scatter, sky maps, histograms, heatmaps, and correlation matrices
- Browse raw data in a sortable, filterable table with summary statistics

---

## Installation

```bash
pip install astroquery astropy plotly dash pandas numpy scipy
```

---

## Usage

```bash
python astrovisualizer.py
```

Open **http://127.0.0.1:8050** in your browser.

