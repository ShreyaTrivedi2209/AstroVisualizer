"""
AstroVisualizer — Astronomical Data Explorer
=============================================
A full interactive Dash application for searching astronomical catalogs
and visualizing astronomical data with multiple chart types.

Requirements:
    pip install astroquery astropy plotly dash pandas numpy scipy

Run:
    python astrovisualizer.py
    Then open http://127.0.0.1:8050 in your browser
"""

import warnings
warnings.filterwarnings("ignore")

import os

import numpy as np
import pandas as pd
from scipy import stats


# ── Dash / Plotly ─────────────────────────────────────────────────────────────
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# CONSTANTS
# =============================================================================

POPULAR_CATALOGS = {
    "Hipparcos (Stars)":          "I/239/hip_main",
    "Gaia DR3 (Stars)":           "I/355/gaiadr3",
    "Tycho-2 (Stars)":            "I/259/tyc2",
    "2MASS All-Sky (IR Stars)":   "II/246/out",
    "SDSS DR16 (Galaxies/QSOs)":  "V/154/sdss16",
    "Exoplanets (Butler 2006)":   "J/ApJ/646/505",
    "Open Clusters (Dias 2002)":  "B/ocl/clusters",
    "Variable Stars (GCVS)":      "B/gcvs/gcvs_cat",
}

CATALOG_FEATURE_INDEX = {
    "I/239/hip_main": {
        "name": "Hipparcos",
        "category": "Stars",
        "key_columns": ["HIP", "Vmag", "Plx", "B-V", "RAdeg", "DEdeg", "pmRA", "pmDE"],
        "description": "Hipparcos main catalog — parallax, magnitudes, colors, proper motions",
    },
    "I/355/gaiadr3": {
        "name": "Gaia DR3",
        "category": "Stars",
        "key_columns": ["Source", "Gmag", "BPmag", "RPmag", "Plx", "ra", "dec", "pmRA", "pmDE", "RVDR2", "Teff"],
        "description": "Gaia DR3 — positions, parallax, photometry, proper motions, radial velocities",
    },
    "I/259/tyc2": {
        "name": "Tycho-2",
        "category": "Stars",
        "key_columns": ["BTmag", "VTmag", "RAmdeg", "DEmdeg", "pmRA", "pmDE"],
        "description": "Tycho-2 catalog — BT/VT photometry, positions, proper motions",
    },
    "II/246/out": {
        "name": "2MASS",
        "category": "Infrared Stars",
        "key_columns": ["RAJ2000", "DEJ2000", "Jmag", "Hmag", "Kmag"],
        "description": "2MASS All-Sky — near-infrared JHK photometry",
    },
    "V/154/sdss16": {
        "name": "SDSS DR16",
        "category": "Galaxies/QSOs",
        "key_columns": ["RA_ICRS", "DE_ICRS", "umag", "gmag", "rmag", "imag", "zmag", "zsp"],
        "description": "SDSS DR16 — ugriz photometry, spectroscopic redshifts",
    },
    "J/ApJ/646/505": {
        "name": "Exoplanets (Butler 2006)",
        "category": "Exoplanets",
        "key_columns": ["Name", "Per", "Mass", "a", "Ecc"],
        "description": "Butler 2006 exoplanet catalog — orbital periods, masses, semi-major axes",
    },
    "B/ocl/clusters": {
        "name": "Open Clusters (Dias 2002)",
        "category": "Clusters",
        "key_columns": ["Cluster", "GLON", "GLAT", "Dist", "Diam", "Age", "RV"],
        "description": "Dias open cluster catalog — positions, distances, ages, radial velocities",
    },
    "B/gcvs/gcvs_cat": {
        "name": "Variable Stars (GCVS)",
        "category": "Variable Stars",
        "key_columns": ["GCVS", "VarType", "magMax", "Period", "RAJ2000", "DEJ2000"],
        "description": "GCVS variable star catalog — variable types, magnitudes, periods",
    },
}


CHART_TYPES = [
    "Scatter Plot",
    "HR Diagram (Auto)",
    "3D Scatter",
    "Sky Map (RA/Dec)",
    "Histogram",
    "Heatmap / Density",
    "Correlation Matrix",
]

DARK = "#0d1117"
PANEL = "#161b22"
ACCENT = "#58a6ff"
TEXT = "#c9d1d9"
BORDER = "#30363d"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=DARK,
        plot_bgcolor="#0d1117",
        font=dict(color=TEXT, family="monospace"),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
        coloraxis_colorbar=dict(tickfont=dict(color=TEXT)),
    )
)

# =============================================================================
# DATA FETCHING & PREPROCESSING
# =============================================================================

def fetch_catalog(catalog_id: str, row_limit: int = 5000,
                  col_filter: dict | None = None) -> pd.DataFrame:
    """
    Fetch a VizieR catalog by ID and return a cleaned pandas DataFrame.
    """
    v = Vizier(row_limit=row_limit, column_filters=col_filter or {})
    v.TIMEOUT = 60
    result = v.get_catalogs(catalog_id)
    if not result:
        raise ValueError(f"No data returned for catalog '{catalog_id}'")
    df = result[0].to_pandas()
    return df


def search_catalogs(keyword: str, max_results: int = 30) -> pd.DataFrame:
    """
    Search VizieR for catalogs matching a keyword. Returns a summary DataFrame.
    """
    v = Vizier()
    results = v.find_catalogs(keyword)
    rows = []
    for cat_id, table in list(results.items())[:max_results]:
        rows.append({
            "Catalog ID":   cat_id,
            "Description":  getattr(table, "description", "—"),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Catalog ID", "Description"])


def query_region(catalog_id: str, ra: float, dec: float,
                 radius_deg: float, row_limit: int = 2000) -> pd.DataFrame:
    """Fetch rows within radius_deg of (ra, dec)."""
    v = Vizier(row_limit=row_limit)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    result = v.query_region(coords, radius=radius_deg * u.deg, catalog=catalog_id)
    if not result:
        raise ValueError("No data in that region.")
    return result[0].to_pandas()


def preprocess(df: pd.DataFrame, zscore_thresh: float = 3.5) -> pd.DataFrame:
    """
    Auto-clean a DataFrame:
      - Drop columns that are >80% NaN
      - Convert object columns that look numeric
      - Z-score outlier removal on numeric columns (preserves at least 10% of rows)
      - Ensure all columns are JSON serializable
    """
    # Drop mostly-empty columns
    df = df.dropna(axis=1, thresh=int(len(df) * 0.2))

    # Flatten any multidimensional array columns (like astropy arrays) and coerce objects
    for col in list(df.columns):
        try:
            if df[col].apply(lambda x: isinstance(x, (list, np.ndarray))).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, np.ndarray)) else x)
        except Exception:
            df[col] = df[col].astype(str)

    for col in list(df.select_dtypes(include=["object"]).columns):
        try:
            # If it contains bytes instead of strings, decode it
            if df[col].apply(lambda x: isinstance(x, bytes)).any():
                df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df) * 0.5:
                df[col] = converted
        except Exception:
            # If a column can't be converted, leave it as string
            df[col] = df[col].astype(str)

    # Z-score filtering on all numeric cols — but never discard all rows
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols and len(df) > 0:
        try:
            zscores = np.abs(stats.zscore(
                df[num_cols].fillna(df[num_cols].median()), nan_policy="omit"
            ))
            mask = (zscores < zscore_thresh).all(axis=1)
            # Only apply filter if it keeps at least 10% of the data
            if mask.sum() >= max(1, int(len(df) * 0.1)):
                df = df[mask]
        except Exception:
            pass  # skip Z-score filtering if it fails

    return df.reset_index(drop=True)


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generalized column derivation for any astronomical dataset.
    Auto-detects which columns are present and computes all applicable
    derived quantities: distance, absolute magnitude, temperature,
    luminosity, color indices, proper motion, 3-D Cartesian coordinates.
    """
    df = df.copy()
    cols = set(df.columns)

    # ── Detect RA / Dec columns (various naming conventions) ──────────────
    ra_col = next((c for c in ["RAdeg", "ra", "RAmdeg", "RAJ2000", "RA_ICRS"] if c in cols), None)
    dec_col = next((c for c in ["DEdeg", "dec", "DEmdeg", "DEJ2000", "DE_ICRS"] if c in cols), None)

    # ── Detect parallax and magnitude columns ─────────────────────────────
    plx_col = "Plx" if "Plx" in cols else None
    mag_col = next((c for c in ["Vmag", "Gmag", "VTmag", "Jmag"] if c in cols), None)

    # ── 1. Tycho-2: derive Vmag and B-V from BT/VT photometry ────────────
    if {"BTmag", "VTmag"}.issubset(cols):
        df["B-V"] = 0.850 * (df["BTmag"] - df["VTmag"])
        df["Vmag"] = df["VTmag"] - 0.090 * (df["BTmag"] - df["VTmag"])
        cols = set(df.columns)
        if mag_col is None:
            mag_col = "Vmag"

    # ── 2. Gaia DR3: derive BP-RP color index ────────────────────────────
    if {"BPmag", "RPmag"}.issubset(cols):
        df["BP_RP"] = df["BPmag"] - df["RPmag"]
        cols = set(df.columns)

    # ── 3. Distance from parallax ─────────────────────────────────────────
    if plx_col and plx_col in cols:
        valid_plx = df[plx_col] > 0
        if valid_plx.any():
            df = df[valid_plx].copy()
            df["dist_pc"] = 1000 / df[plx_col]
            df["dist_ly"] = df["dist_pc"] * 3.26156
            cols = set(df.columns)

    # ── 4. Absolute magnitude ─────────────────────────────────────────────
    if "dist_pc" in cols and mag_col and mag_col in cols:
        df["abs_mag"] = df[mag_col] - 5 * np.log10(df["dist_pc"]) + 5
        cols = set(df.columns)

    # ── 5. Temperature & luminosity from B-V (Hipparcos / Tycho-2) ───────
    if "B-V" in cols and "abs_mag" in cols:
        bv = df["B-V"]
        df["temp_K"] = 4600 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62))
        df["luminosity"] = 10 ** ((4.83 - df["abs_mag"]) / 2.5)
        cols = set(df.columns)

    # ── 6. Temperature & luminosity from BP-RP (Gaia DR3) ────────────────
    if "BP_RP" in cols and "abs_mag" in cols and "temp_K" not in cols:
        bp_rp = df["BP_RP"]
        df["temp_K"] = 4600 * (1 / (0.92 * bp_rp + 1.7) + 1 / (0.92 * bp_rp + 0.62))
        df["luminosity"] = 10 ** ((4.83 - df["abs_mag"]) / 2.5)
        cols = set(df.columns)

    # ── 7. Use Gaia Teff directly if available and temp_K not yet set ────
    if "Teff" in cols and "temp_K" not in cols:
        df["temp_K"] = df["Teff"]
        if "abs_mag" in cols:
            df["luminosity"] = 10 ** ((4.83 - df["abs_mag"]) / 2.5)
        cols = set(df.columns)

    # ── 8. 2MASS near-infrared color indices ──────────────────────────────
    if {"Jmag", "Hmag", "Kmag"}.issubset(cols):
        df["J-H"] = df["Jmag"] - df["Hmag"]
        df["H-K"] = df["Hmag"] - df["Kmag"]
        df["J-K"] = df["Jmag"] - df["Kmag"]
        cols = set(df.columns)

    # ── 9. SDSS optical color indices ─────────────────────────────────────
    if {"umag", "gmag", "rmag", "imag", "zmag"}.issubset(cols):
        df["u-g"] = df["umag"] - df["gmag"]
        df["g-r"] = df["gmag"] - df["rmag"]
        df["r-i"] = df["rmag"] - df["imag"]
        df["i-z"] = df["imag"] - df["zmag"]
        cols = set(df.columns)

    # ── 10. Exoplanet orbital period in years ─────────────────────────────
    if "Per" in cols:
        df["Per_yr"] = df["Per"] / 365.25
        cols = set(df.columns)

    # ── 11. Total proper motion ───────────────────────────────────────────
    if {"pmRA", "pmDE"}.issubset(cols):
        df["pm_total"] = np.sqrt(df["pmRA"] ** 2 + df["pmDE"] ** 2)
        cols = set(df.columns)

    # ── 12. Open cluster log-age in Gyr ───────────────────────────────────
    if "Age" in cols:
        df["Age_Gyr"] = df["Age"] / 1e9
        cols = set(df.columns)

    # ── 13. 3-D Cartesian coordinates (if distance + RA/Dec available) ───
    if "dist_pc" in cols and ra_col and dec_col:
        ra_rad = np.radians(df[ra_col])
        dec_rad = np.radians(df[dec_col])
        df["x"] = df["dist_pc"] * np.cos(dec_rad) * np.cos(ra_rad)
        df["y"] = df["dist_pc"] * np.cos(dec_rad) * np.sin(ra_rad)
        df["z"] = df["dist_pc"] * np.sin(dec_rad)

    return df.reset_index(drop=True)

# =============================================================================
# CHART BUILDERS
# =============================================================================

def make_scatter(df, xcol, ycol, color_col=None, size_col=None, title="Scatter"):
    df = df.copy()
    # px.scatter size column must be non-negative; clip or drop size_col if invalid
    if size_col and size_col in df.columns:
        col_min = df[size_col].min()
        if col_min <= 0:
            df["_scatter_size"] = (df[size_col] - col_min + 0.1).clip(lower=0.1)
            effective_size = "_scatter_size"
        else:
            effective_size = size_col
    else:
        effective_size = None
    fig = px.scatter(
        df, x=xcol, y=ycol,
        color=color_col,
        size=effective_size,
        size_max=10,
        color_continuous_scale="Plasma",
        hover_data=df.columns[:6].tolist(),
        title=title,
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(opacity=0.75, line=dict(width=0)))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    return fig


def make_hr_diagram(df):
    if {"temp_K", "luminosity"}.issubset(df.columns):
        # Pick the best available color column
        color_col = next((c for c in ["B-V", "BP_RP"] if c in df.columns), None)
        color_label = {"B-V": "B−V", "BP_RP": "BP−RP"}.get(color_col, color_col)

        fig = px.scatter(
            df, x="temp_K", y="luminosity",
            color=color_col,
            color_continuous_scale="RdYlBu_r",
            log_y=True,
            hover_data=["abs_mag", "dist_ly"] if "dist_ly" in df.columns else None,
            title="Hertzsprung–Russell Diagram",
            labels={"temp_K": "Temperature (K)", "luminosity": "Luminosity (L☉)",
                     **(({color_col: color_label}) if color_col else {})},
            template="plotly_dark",
        )
        fig.update_xaxes(autorange="reversed")
        fig.update_traces(marker=dict(size=3, opacity=0.8, line=dict(width=0)))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())

        # Annotate stellar classes
        for label, x, y in [("O", 40000, 1e5), ("B", 20000, 1e3),
                             ("A", 9000, 50), ("F", 7000, 5),
                             ("G", 5500, 1), ("K", 4500, 0.2),
                             ("M", 3200, 0.02)]:
            fig.add_annotation(x=x, y=np.log10(y), text=label,
                               showarrow=False, font=dict(color=ACCENT, size=11))
        return fig

    return go.Figure().add_annotation(
        text="HR Diagram requires stellar data with parallax and photometry (e.g. Hipparcos, Gaia DR3, Tycho-2)",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        font=dict(color=TEXT, size=14)
    )


def make_3d_scatter(df, xcol, ycol, zcol, color_col=None):
    sample = df.sample(min(4000, len(df)), random_state=42)
    fig = px.scatter_3d(
        sample, x=xcol, y=ycol, z=zcol,
        color=color_col,
        color_continuous_scale="Plasma",
        size_max=4,
        opacity=0.75,
        title=f"3D: {xcol} / {ycol} / {zcol}",
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    return fig


def make_sky_map(df):
    ra_col  = next((c for c in df.columns if "ra"  in c.lower()), None)
    dec_col = next((c for c in df.columns if "de"  in c.lower() or "dec" in c.lower()), None)
    if not ra_col or not dec_col:
        return go.Figure().add_annotation(
            text="No RA/Dec columns found in this dataset",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=TEXT, size=14)
        )

    mag_col = next((c for c in df.columns if "mag" in c.lower()), None)

    # Normalize point size by magnitude (brighter = bigger)
    # px.scatter needs a column name or None for size, not a raw Series
    df = df.copy()
    if mag_col:
        df["_size"] = (1 / np.clip(df[mag_col].fillna(df[mag_col].median()) + 2, 0.5, 20) * 30).clip(lower=0.5)
        size_param = "_size"
    else:
        size_param = None

    fig = px.scatter(
        df, x=ra_col, y=dec_col,
        color=mag_col,
        size=size_param,
        size_max=12,
        color_continuous_scale="Blues_r",
        title="All-Sky Map",
        labels={ra_col: "Right Ascension (°)", dec_col: "Declination (°)"},
        template="plotly_dark",
    )
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0)))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json(),
                      height=500)
    return fig


def make_histogram(df, col, nbins=60):
    fig = px.histogram(
        df, x=col, nbins=nbins,
        title=f"Distribution of {col}",
        template="plotly_dark",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    return fig


def make_heatmap(df, xcol, ycol, nbins=60):
    fig = go.Figure(go.Histogram2dContour(
        x=df[xcol].dropna(), y=df[ycol].dropna(),
        colorscale="Plasma",
        contours=dict(coloring="heatmap"),
        line=dict(width=0),
    ))
    layout_kwargs = PLOTLY_TEMPLATE["layout"].to_plotly_json()
    layout_kwargs.pop("template", None)  # avoid conflict
    fig.update_layout(
        title=f"Density: {xcol} vs {ycol}",
        xaxis_title=xcol, yaxis_title=ycol,
        **layout_kwargs,
    )
    return fig


def make_correlation_matrix(df):
    """Generate a correlation heatmap for all numeric columns."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return go.Figure().add_annotation(
            text="Need at least 2 numeric columns for correlation matrix",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(color=TEXT, size=14)
        )
    corr = num_df.corr().round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
    ))
    layout_kwargs = PLOTLY_TEMPLATE["layout"].to_plotly_json()
    layout_kwargs.pop("template", None)
    fig.update_layout(
        title="Correlation Matrix",
        **layout_kwargs,
        height=600,
    )
    return fig


# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(__name__, title="AstroVisualizer", assets_folder="assets")

# Optional: load custom HTML template if present
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(BASE_DIR, "templates", "index.html")
if os.path.exists(template_path):
    with open(template_path, "r", encoding="utf-8") as f:
        app.index_string = f.read()

# ── Layout ───────────────────────────────────────────────────────────────────
app.layout = html.Div(className="app-root", children=[

    # Header
    html.Header(className="app-header", children=[
        html.Div(className="app-header-icon", children="🔭"),
        html.Div(className="app-header-text", children=[
            html.H1("AstroVisualizer", className="app-title"),
            html.Div("powered by astroquery + Plotly Dash", className="app-subtitle"),
        ]),
        html.Button("🔄 Refresh", id="btn-refresh", className="secondary-button", style={"marginLeft": "auto", "padding": "8px 16px", "borderRadius": "4px", "cursor": "pointer", "backgroundColor": BORDER, "color": TEXT, "border": "none"}),
    ]),
    # Dummy div to trigger a page refresh
    html.Div(id='dummy-refresh', style={'display': 'none'}),

    # Main body
    html.Div(className="app-main", children=[

        # Left sidebar
        html.Aside(className="sidebar", children=[

            # Search catalogs
            html.Div(className="panel", children=[
                html.H3("Search Catalogs", className="panel-title"),
                html.Label("Keyword", className="field-label", htmlFor="search-keyword"),
                dcc.Input(
                    id="search-keyword",
                    type="text",
                    placeholder="e.g. hipparcos, quasar...",
                    className="text-input",
                    value="",
                    debounce=False,
                ),
                html.Label("Dataset focus", className="field-label", htmlFor="search-filter"),
                dcc.Dropdown(
                    id="search-filter",
                    options=[
                        {"label": "Any dataset", "value": "all"},
                        {"label": "Stars / Stellar", "value": "stars"},
                        {"label": "Galaxies & Extragalactic", "value": "galaxies"},
                        {"label": "Exoplanets", "value": "exoplanets"},
                        {"label": "Clusters", "value": "clusters"},
                        {"label": "Variable stars", "value": "variables"},
                    ],
                    value="all",
                    clearable=False,
                    className="dropdown",
                ),
                html.Button(
                    "Search Catalogs",
                    id="btn-search",
                    n_clicks=0,
                    className="primary-button",
                ),
                html.Div(id="search-status", className="status-text"),
                html.Div(id="search-results-container", className="results-container"),
            ]),

            # Load catalog
            html.Div(className="panel", children=[
                html.H3("Load Catalog", className="panel-title"),
                html.Label("Quick-select popular catalog", className="field-label",
                           htmlFor="catalog-preset"),
                dcc.Dropdown(
                    id="catalog-preset",
                    options=[{"label": k, "value": v} for k, v in POPULAR_CATALOGS.items()],
                    placeholder="Choose a preset…",
                    className="dropdown",
                ),
                html.Label("Or enter Catalog ID manually", className="field-label",
                           htmlFor="catalog-id"),
                dcc.Input(
                    id="catalog-id",
                    type="text",
                    placeholder="e.g. I/239/hip_main",
                    className="text-input",
                    debounce=False,
                ),
                html.Label("Max rows to fetch", className="field-label",
                           htmlFor="row-limit"),
                dcc.Slider(
                    id="row-limit",
                    min=500,
                    max=20000,
                    step=500,
                    value=5000,
                    marks={500: "500", 5000: "5k", 10000: "10k", 20000: "20k"},
                    tooltip={"placement": "bottom"},
                    className="slider",
                ),
                html.Button(
                    "Load & Process",
                    id="btn-load",
                    n_clicks=0,
                    className="primary-button",
                ),
                html.Div(id="load-status", className="status-text"),
            ]),

            # Region query
            html.Div(className="panel", children=[
                html.H3("Query by Sky Region", className="panel-title"),
                html.Label("RA (deg)", className="field-label", htmlFor="region-ra"),
                dcc.Input(
                    id="region-ra",
                    type="number",
                    placeholder="83.82",
                    className="text-input",
                ),
                html.Label("Dec (deg)", className="field-label", htmlFor="region-dec"),
                dcc.Input(
                    id="region-dec",
                    type="number",
                    placeholder="-5.39",
                    className="text-input",
                ),
                html.Label("Radius (deg)", className="field-label", htmlFor="region-radius"),
                dcc.Input(
                    id="region-radius",
                    type="number",
                    placeholder="1.0",
                    value=1.0,
                    className="text-input",
                ),
                html.Button(
                    "Query Region",
                    id="btn-region",
                    n_clicks=0,
                    className="primary-button",
                ),
                html.Div(id="region-status", className="status-text"),
            ]),
        ]),

        # Right content
        html.Main(className="content", children=[

            # Tabs
            dcc.Tabs(
                id="tabs",
                value="tab-viz",
                className="tabs",
                children=[
                    dcc.Tab(
                        label="Visualizer",
                        value="tab-viz",
                        className="tab",
                        selected_className="tab--selected",
                    ),
                    dcc.Tab(
                        label="Data Table",
                        value="tab-table",
                        className="tab",
                        selected_className="tab--selected",
                    ),
                    dcc.Tab(
                        label="Stats",
                        value="tab-stats",
                        className="tab",
                        selected_className="tab--selected",
                    ),
                ],
            ),

            # Visualizer tab
            html.Div(id="tab-viz-content", className="tab-content", children=[
                html.Div(className="controls-row", children=[
                    html.Div(className="control", children=[
                        html.Label("Chart type", className="field-label", htmlFor="chart-type"),
                        dcc.Dropdown(
                            id="chart-type",
                            options=[{"label": c, "value": c} for c in CHART_TYPES],
                            value="Scatter Plot",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]),
                    html.Div(className="control", children=[
                        html.Label("X axis", className="field-label", htmlFor="xcol"),
                        dcc.Dropdown(
                            id="xcol",
                            clearable=False,
                            className="dropdown",
                        ),
                    ]),
                    html.Div(className="control", children=[
                        html.Label("Y axis", className="field-label", htmlFor="ycol"),
                        dcc.Dropdown(
                            id="ycol",
                            clearable=True,
                            className="dropdown",
                        ),
                    ]),
                    html.Div(className="control", children=[
                        html.Label("Z axis (3D only)", className="field-label", htmlFor="zcol"),
                        dcc.Dropdown(
                            id="zcol",
                            clearable=True,
                            className="dropdown",
                        ),
                    ]),
                    html.Div(className="control", children=[
                        html.Label("Color by", className="field-label", htmlFor="color-col"),
                        dcc.Dropdown(
                            id="color-col",
                            clearable=True,
                            className="dropdown",
                        ),
                    ]),
                    html.Div(className="control", children=[
                        html.Label("Size by (scatter)", className="field-label", htmlFor="size-col"),
                        dcc.Dropdown(
                            id="size-col",
                            clearable=True,
                            className="dropdown",
                        ),
                    ]),
                ]),

                dcc.Loading(
                    id="chart-loading",
                    type="circle",
                    color=ACCENT,
                    className="chart-loading",
                    children=dcc.Graph(
                        id="main-chart",
                        className="main-chart",
                        config={"scrollZoom": True, "displayModeBar": True},
                    ),
                ),
            ]),

            # Data table tab
            html.Div(
                id="tab-table-content",
                className="tab-content",
                style={"display": "none"},
                children=[html.Div(id="table-container", className="table-container")],
            ),

            # Stats tab
            html.Div(
                id="tab-stats-content",
                className="tab-content",
                style={"display": "none"},
                children=[html.Div(id="stats-container", className="stats-container")],
            ),
        ]),
    ]),

    # Hidden storage
    dcc.Store(id="store-df"),      # JSON-serialised DataFrame
    dcc.Store(id="store-columns"), # list of numeric column names
    dcc.Location(id="url-refresh", refresh=True),
])

# =============================================================================
# CALLBACKS
# =============================================================================

# ── Sync catalog preset → manual input ───────────────────────────────────────
@app.callback(
    Output("catalog-id", "value"),
    Input("catalog-preset", "value"),
    prevent_initial_call=True,
)
def sync_preset(value):
    return value or ""


# ── Search VizieR ─────────────────────────────────────────────────────────────
@app.callback(
    Output("search-results-container", "children"),
    Output("search-status", "children"),
    Input("btn-search", "n_clicks"),
    State("search-keyword", "value"),
    State("search-filter", "value"),
    prevent_initial_call=True,
)
def do_search(n, keyword, filter_value):
    if not keyword:
        return "", "⚠ Enter a keyword first."
    try:
        res = search_catalogs(keyword)

        # Apply simple thematic filters on the client side
        if filter_value and filter_value != "all" and not res.empty:
            filter_terms = {
                "stars": ["star", "stellar", "hipparcos", "gaia", "tycho"],
                "galaxies": ["galaxy", "galaxies", "extragalactic", "sdss"],
                "exoplanets": ["exoplanet", "exo", "planet", "kepler"],
                "clusters": ["cluster", "open cluster", "globular"],
                "variables": ["variable", "variables", "gcvs"],
            }.get(filter_value, [])

            if filter_terms:
                pattern = "|".join(filter_terms)
                mask = (
                    res["Description"].str.contains(pattern, case=False, na=False)
                    | res["Catalog ID"].str.contains(pattern, case=False, na=False)
                )
                res = res[mask]
        if res.empty:
            return "", "No catalogs found."
        table = dash_table.DataTable(
            data=res.to_dict("records"),
            columns=[{"name": c, "id": c} for c in res.columns],
            style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
            style_cell={"backgroundColor": DARK, "color": TEXT,
                        "border": f"1px solid {BORDER}", "fontSize": "11px",
                        "padding": "4px 8px", "fontFamily": "monospace",
                        "textAlign": "left", "whiteSpace": "normal"},
            style_header={"backgroundColor": PANEL, "color": ACCENT,
                          "fontWeight": "bold"},
            page_size=50,
        )
        return table, f"✅ Found {len(res)} catalogs for '{keyword}'"
    except Exception as e:
        return "", f"❌ Error: {e}"


# ── Load catalog ──────────────────────────────────────────────────────────────
@app.callback(
    Output("store-df", "data"),
    Output("store-columns", "data"),
    Output("load-status", "children"),
    Output("region-status", "children"),
    Input("btn-load", "n_clicks"),
    Input("btn-region", "n_clicks"),
    State("catalog-id", "value"),
    State("row-limit", "value"),
    State("region-ra", "value"),
    State("region-dec", "value"),
    State("region-radius", "value"),
    prevent_initial_call=True,
)
def load_data(n_load, n_region, catalog_id, row_limit, ra, dec, radius):
    triggered = ctx.triggered_id
    load_msg = region_msg = ""

    try:
        if triggered == "btn-load":
            if not catalog_id:
                return None, None, "⚠ Enter a catalog ID.", ""
            df = fetch_catalog(catalog_id.strip(), row_limit=int(row_limit or 5000))
            df = preprocess(df)
            df = derive_columns(df)
            load_msg = f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns"

        elif triggered == "btn-region":
            if not catalog_id:
                return None, None, "⚠ Enter a catalog ID first.", ""
            if ra is None or dec is None:
                return None, None, "", "⚠ Enter RA and Dec."
            df = query_region(catalog_id.strip(), float(ra), float(dec),
                               float(radius or 1.0))
            df = preprocess(df)
            df = derive_hipparcos_columns(df)
            region_msg = f"✅ Region query: {len(df):,} rows"
        else:
            return None, None, "", ""

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df.to_json(date_format="iso", orient="split"), num_cols, load_msg, region_msg

    except Exception as e:
        return None, None, f"❌ {e}", f"❌ {e}"


# ── Populate axis dropdowns ────────────────────────────────────────────────────
@app.callback(
    Output("xcol", "options"), Output("xcol", "value"),
    Output("ycol", "options"), Output("ycol", "value"),
    Output("zcol", "options"), Output("zcol", "value"),
    Output("color-col", "options"), Output("color-col", "value"),
    Output("size-col", "options"), Output("size-col", "value"),
    Input("store-columns", "data"),
    prevent_initial_call=True,
)
def populate_dropdowns(cols):
    if not cols:
        empty_opts, empty_val = [], None
        return (
            empty_opts, empty_val,  # x options, x value
            empty_opts, empty_val,  # y options, y value
            empty_opts, empty_val,  # z options, z value
            empty_opts, empty_val,  # color options, color value
            empty_opts, empty_val,  # size options, size value
        )

    opts = [{"label": c, "value": c} for c in cols]

    def pick(*candidates):
        for c in candidates:
            if c in cols:
                return c
        return cols[0] if cols else None

    x = pick("temp_K", "RAdeg", cols[0])
    y = pick("luminosity", "DEdeg", cols[1] if len(cols) > 1 else cols[0])
    z = pick("z", "dist_pc", cols[2] if len(cols) > 2 else None)
    color = pick("B-V", "Vmag", cols[0])
    size  = pick("luminosity", "Vmag")

    return opts, x, opts, y, opts, z, opts, color, opts, size


# ── Render chart ──────────────────────────────────────────────────────────────
@app.callback(
    Output("main-chart", "figure"),
    Input("store-df", "data"),
    Input("chart-type", "value"),
    Input("xcol", "value"),
    Input("ycol", "value"),
    Input("zcol", "value"),
    Input("color-col", "value"),
    Input("size-col", "value"),
    prevent_initial_call=True,
)
def render_chart(json_data, chart_type, xcol, ycol, zcol, color_col, size_col):
    empty_fig = go.Figure().update_layout(
        paper_bgcolor=DARK, plot_bgcolor=DARK,
        font=dict(color=TEXT),
        annotations=[dict(text="Load a catalog to see visualizations",
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font=dict(size=16, color="#8b949e"))]
    )

    if not json_data or not xcol:
        return empty_fig

    df = pd.read_json(json_data, orient="split")

    try:
        if chart_type == "HR Diagram (Auto)":
            return make_hr_diagram(df)
        elif chart_type == "Sky Map (RA/Dec)":
            return make_sky_map(df)
        elif chart_type == "Scatter Plot":
            return make_scatter(df, xcol, ycol or xcol, color_col, size_col)
        elif chart_type == "3D Scatter":
            if not zcol:
                return empty_fig.add_annotation(
                    text="Select a Z axis for 3D scatter",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return make_3d_scatter(df, xcol, ycol or xcol, zcol, color_col)
        elif chart_type == "Histogram":
            return make_histogram(df, xcol)
        elif chart_type == "Heatmap / Density":
            if not ycol:
                return go.Figure().add_annotation(
                    text="Select a Y axis for Heatmap",
                    xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return make_heatmap(df, xcol, ycol)
        elif chart_type == "Correlation Matrix":
            return make_correlation_matrix(df)
    except Exception as e:
        return empty_fig.add_annotation(
            text=f"Chart error: {e}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color="tomato", size=14))


# ── Tab visibility ─────────────────────────────────────────────────────────────
@app.callback(
    Output("tab-viz-content",   "style"),
    Output("tab-table-content", "style"),
    Output("tab-stats-content", "style"),
    Input("tabs", "value"),
)
def switch_tabs(tab):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (
        show if tab == "tab-viz"   else hide,
        show if tab == "tab-table" else hide,
        show if tab == "tab-stats" else hide,
    )


# ── Data table ────────────────────────────────────────────────────────────────
@app.callback(
    Output("table-container", "children"),
    Input("store-df", "data"),
    prevent_initial_call=True,
)
def render_table(json_data):
    if not json_data:
        return html.P("No data loaded yet.", style={"color": "#8b949e"})
    df = pd.read_json(json_data, orient="split")
    return dash_table.DataTable(
        data=df.head(500).round(4).to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": DARK, "color": TEXT,
                    "border": f"1px solid {BORDER}", "fontSize": "11px",
                    "padding": "4px 8px", "fontFamily": "monospace",
                    "minWidth": "80px", "maxWidth": "160px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
        style_header={"backgroundColor": PANEL, "color": ACCENT, "fontWeight": "bold"},
        page_size=20,
        sort_action="native",
        filter_action="native",
        tooltip_data=[{c: {"value": str(row[c]), "type": "markdown"}
                       for c in df.columns} for row in df.head(500).to_dict("records")],
        tooltip_duration=None,
    )


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.callback(
    Output("stats-container", "children"),
    Input("store-df", "data"),
    prevent_initial_call=True,
)
def render_stats(json_data):
    if not json_data:
        return html.P("No data loaded yet.", style={"color": "#8b949e"})
    df = pd.read_json(json_data, orient="split")
    desc = df.describe().round(4).reset_index()
    return html.Div([
        html.H3(f"Dataset: {len(df):,} rows × {len(df.columns)} columns",
                style={"color": ACCENT}),
        dash_table.DataTable(
            data=desc.to_dict("records"),
            columns=[{"name": c, "id": c} for c in desc.columns],
            style_table={"overflowX": "auto"},
            style_cell={"backgroundColor": DARK, "color": TEXT,
                        "border": f"1px solid {BORDER}", "fontSize": "11px",
                        "padding": "4px 8px", "fontFamily": "monospace"},
            style_header={"backgroundColor": PANEL, "color": ACCENT, "fontWeight": "bold"},
        )
    ])

# =============================================================================
# ENTRY POINT
# =============================================================================

@app.callback(
    Output('url-refresh', 'href'),
    Input('btn-refresh', 'n_clicks'),
    prevent_initial_call=True
)
def refresh_page(n_clicks):
    """Force a full page reload by navigating to the same URL."""
    if n_clicks:
        return "/"
    return dash.no_update

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AstroVisualizer — Astronomical Data Explorer")
    print("="*60)
    print("  Open your browser at:  http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=False, port=8050)