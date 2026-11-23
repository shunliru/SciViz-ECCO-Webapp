import os
import base64
from io import BytesIO
import numpy as np
import pyvista as pv
import openvisuspy as ovp

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import html
from pyvista.trame.ui import plotter_ui

from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import xarray as xr

# =========================================================
# 1. CONFIG
# =========================================================

os.environ["VISUS_CACHE"] = os.environ.get(
    "VISUS_CACHE",
    "./visus_cache_can_be_erased"
)

# Replace these with your actual dataset paths / URLs
temperature = "pelican://osg-htc.org/nasa/nsdf/climate1/llc4320/idx/theta/theta_llc4320_x_y_depth.idx"
salinity = "pelican://osg-htc.org/nasa/nsdf/climate1/llc4320/idx/salt/salt_llc4320_x_y_depth.idx"
vertical_velocity = "pelican://osg-htc.org/nasa/nsdf/climate2/llc4320/idx/w/w_llc4320_x_y_depth.idx"

#equivalent URL in case pelican does not work
#temperature = "osdf:///nasa/nsdf/climate1/llc4320/idx/theta/theta_llc4320_x_y_depth.idx"
#salinity = "osdf:///nasa/nsdf/climate1/llc4320/idx/salt/salt_llc4320_x_y_depth.idx"
#vertical_velocity = "osdf:///nasa/nsdf/climate2/llc4320/idx/w/w_llc4320_x_y_depth.idx"

# Time slider range (adjust as needed)
TIME_MIN = 0
TIME_MAX = 10000

# Region / quality
READ_KWARGS = dict(
    z=[0, 90],
    # global
    #x = [0, 17280],
    #y = [0, 12960],

    # north atlantic
    x=[14000, 17280],
    y=[8000, 11000],
    
    # japan sea
    #x = [7500, 9700],
    #y = [9000, 11000],

    # australia
    #x = [9000, 10200],
    #y = [5200, 7600],
    quality= -9,   # start coarse
)

# Fixed color range for temperature anomaly on salinity layers
T_ANOM_CLIM = (-20.0, 20.0)


# =========================================================
# 2. LOAD DATASETS
# =========================================================

t_ds = ovp.LoadDataset(temperature)
s_ds = ovp.LoadDataset(salinity)
w_ds = ovp.LoadDataset(vertical_velocity)

latlon_ds = xr.open_dataset("./llc4320_latlon.nc", engine="netcdf4")
lat_center = latlon_ds["latitude"].values
lon_center = latlon_ds["longitude"].values

# def extract_data_latlon(lat_center, lon_center, x_range, y_range):
#     lat = lat_center[y_range[0]:y_range[1], x_range[0]:x_range[1]]
#     lon = lon_center[y_range[0]:y_range[1], x_range[0]:x_range[1]]
#     return lat, lon

def extract_data_latlon_matching(lat_center, lon_center, x_range, y_range, nx, ny):
    """
    Extract lat/lon and resample them to match the actual data shape.

    This is needed when OpenVisus quality < 0 returns downsampled data.
    """
    x0, x1 = x_range
    y0, y1 = y_range

    x_idx = np.linspace(x0, x1 - 1, nx).astype(int)
    y_idx = np.linspace(y0, y1 - 1, ny).astype(int)

    lat = lat_center[np.ix_(y_idx, x_idx)]
    lon = lon_center[np.ix_(y_idx, x_idx)]

    return lat, lon

# =========================================================
# 3. DATA READING
# =========================================================

def read_time_step(time_index):
    """
    Read T, S, W at one time step and mask land.
    Land is where S == 0.
    """
    T = t_ds.db.read(time=int(time_index), **READ_KWARGS)
    S = s_ds.db.read(time=int(time_index), **READ_KWARGS)
    W = w_ds.db.read(time=int(time_index), **READ_KWARGS)

    T = T.astype(np.float32)
    S = S.astype(np.float32)
    W = W.astype(np.float32)

    # land mask from salinity
    ocean_mask = (S != 0)
    land_mask = (S == 0)

    T = np.where(ocean_mask, T, np.nan)
    S = np.where(ocean_mask, S, np.nan)
    W = np.where(ocean_mask, W, np.nan)

    # lat, lon = extract_data_latlon(
    #     lat_center,
    #     lon_center,
    #     READ_KWARGS["x"],
    #     READ_KWARGS["y"],
    # )
    nz, ny, nx = T.shape

    lat, lon = extract_data_latlon_matching(
        lat_center,
        lon_center,
        READ_KWARGS["x"],
        READ_KWARGS["y"],
        nx,
        ny,
    )
    return T, S, W, land_mask, lat, lon

def plot_salinity_histograms(timesteps):

    for t in timesteps:

        print(f"Generating histogram for timestep {t}...", flush=True)

        # Read salinity
        T, S, W, land_mask, lat2d, lon2d = read_time_step(t)

        # Remove land / invalid values
        S_valid = S[
            np.isfinite(S) &
            (S != 0)
        ]

        # Percentiles
        p25 = np.percentile(S_valid, 5)
        p50 = np.percentile(S_valid, 45)
        p75 = np.percentile(S_valid, 80)

        print(
            f"t={t}: "
            f"25th percentile = {p25: 4f},"
            f"50th percentile = {p50:.4f}, "
            f"75th percentile = {p75:.4f}"
        )

        # Create separate histogram
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.hist(
            S_valid,
            bins=np.linspace(30, 40, 101),
            density=True,
            edgecolor="black",
            alpha=0.8,
        )

        ax.axvline(
            p25,
            color="red",
            linestyle="-.",
            linewidth=2,
            label=f"25th percentile = {p25:.3f}"
            )

        ax.axvline(
            p50,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"50th percentile = {p50:.3f}"
        )

        ax.axvline(
            p75,
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"75th percentile = {p75:.3f}"
        )

        ax.set_xlabel("Salinity (g kg$^{-1}$)")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"Salinity Distribution — Time Step {t}")

        ax.set_xlim(30, 40)

        ax.legend()
        ax.grid(alpha=0.3)

        fig.tight_layout()

        fig.savefig(
            f"salinity_histogram_t{t}.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved salinity_histogram_t{t}.png", flush=True)
# =========================================================
# 4. DATE CONVERSION
# =========================================================

START_TIME = datetime(2011, 9, 13, 0, 0, 0)


def get_datetime_from_timestep(t):
    """
    Convert ECCO timestep index to actual datetime.
    Assumption: each timestep is 1 hour.
    """
    return START_TIME + timedelta(hours=int(t))



# =========================================================
# 5. REFERENCE TEMPERATURE 
# =========================================================

# One scalar regional average temperature
T_ref = None

# Get reference temperature at time 0, comment out if you want a fixed reference temperature or at a fixed timestep
def get_reference_temperature():
    global T_ref

    if T_ref is None:
        print("Loading reference temperature at time 0...", flush=True)

        T0, S0, W0, land_mask0, lat2d0, lon2d0 = read_time_step(time_index=0)
        T_ref = np.nanmean(T0)

        print("Computed T_ref =", T_ref, flush=True)

    return T_ref

# =========================================================
# 6. ARRAY CONVERSION
# =========================================================

def zyx_to_pyvista(arr):
    """
    Convert array from (z, y, x) to PyVista point ordering.
    """
    arr_xyz = np.transpose(arr, (2, 1, 0))   # (x, y, z)
    return arr_xyz.ravel(order="F")


# =========================================================
# 7. BUILD GRID
# =========================================================

# def make_grid(T, S, W, land_mask):
#     """
#     Build a PyVista ImageData grid with:
#       - salinity S
#       - vertical velocity W
#       - temperature anomaly T_anom = T - T_ref
#     """
#     T_anom = T - T_ref

#     nz, ny, nx = T.shape

#     grid = pv.ImageData()
#     grid.dimensions = (nx, ny, nz)
#     grid.spacing = (1.0, 1.0, 1.0)
#     grid.origin = (0.0, 0.0, 0.0)

#     grid.point_data["T"] = zyx_to_pyvista(T)
#     grid.point_data["S"] = zyx_to_pyvista(S)
#     grid.point_data["W"] = zyx_to_pyvista(W)
#     grid.point_data["T_anom"] = zyx_to_pyvista(T_anom)
#     grid["land"] = zyx_to_pyvista(land_mask.astype(float))

#     return grid

# def make_grid(T, S, W, land_mask, lat2d, lon2d):
#     """
#     Build a PyVista StructuredGrid using real longitude/latitude coordinates.

#     Input data shape:
#         T, S, W, land_mask: (z, y, x)
#         lat2d, lon2d:       (y, x)

#     PyVista grid coordinate shape:
#         (x, y, z)
#     """
#     nz, ny, nx = T.shape

#     # Convert horizontal coordinates from (y, x) to (x, y)
#     lon_xy = lon2d.T
#     lat_xy = lat2d.T

#     # Vertical coordinate: still use depth level index
#     z_levels = np.arange(nz, dtype=float)

#     # Build 3D coordinate arrays with shape (x, y, z)
#     lon_xyz = np.repeat(lon_xy[:, :, None], nz, axis=2)
#     lat_xyz = np.repeat(lat_xy[:, :, None], nz, axis=2)
#     z_xyz = np.repeat(z_levels[None, None, :], nx, axis=0)
#     z_xyz = np.repeat(z_xyz, ny, axis=1)

#     # diagnosis print
#     print("T shape:", T.shape, flush=True)
#     print("lat2d shape:", lat2d.shape, flush=True)
#     print("lon2d shape:", lon2d.shape, flush=True)
#     print("lon_xyz shape:", lon_xyz.shape, flush=True)
#     print("lat_xyz shape:", lat_xyz.shape, flush=True)
#     print("z_xyz shape:", z_xyz.shape, flush=True)

#     grid = pv.StructuredGrid(lon_xyz, lat_xyz, z_xyz)

#     # Attach scalar fields
#     grid["T"] = zyx_to_pyvista(T)
#     grid["S"] = zyx_to_pyvista(S)
#     grid["W"] = zyx_to_pyvista(W)
#     grid["land"] = zyx_to_pyvista(land_mask.astype(float))

#     return grid

def make_grid(T, S, W, land_mask, lat2d, lon2d, T_anom):
    nz, ny, nx = T.shape

    print("T shape:", T.shape, flush=True)
    print("lat2d shape:", lat2d.shape, flush=True)
    print("lon2d shape:", lon2d.shape, flush=True)

    if lat2d.shape != (ny, nx):
        raise ValueError(f"lat2d shape {lat2d.shape} does not match data horizontal shape {(ny, nx)}")

    if lon2d.shape != (ny, nx):
        raise ValueError(f"lon2d shape {lon2d.shape} does not match data horizontal shape {(ny, nx)}")

    lon_xy = lon2d.T
    lat_xy = lat2d.T
    z_levels = np.arange(nz, dtype=float)

    lon_xyz, lat_xyz, z_xyz = np.meshgrid(
        lon_xy[:, 0],
        lat_xy[0, :],
        z_levels,
        indexing="ij",
    )

    # Better for curvilinear lon/lat:
    lon_xyz = np.repeat(lon_xy[:, :, None], nz, axis=2)
    lat_xyz = np.repeat(lat_xy[:, :, None], nz, axis=2)
    z_xyz = np.repeat(z_levels[None, None, :], nx, axis=0)
    z_xyz = np.repeat(z_xyz, ny, axis=1)

    print("lon_xyz shape:", lon_xyz.shape, flush=True)
    print("lat_xyz shape:", lat_xyz.shape, flush=True)
    print("z_xyz shape:", z_xyz.shape, flush=True)

    grid = pv.StructuredGrid(lon_xyz, lat_xyz, z_xyz)

    grid["T"] = zyx_to_pyvista(T)
    grid["S"] = zyx_to_pyvista(S)
    grid["W"] = zyx_to_pyvista(W)
    grid["land"] = zyx_to_pyvista(land_mask.astype(float))
    grid["T_anom"] = zyx_to_pyvista(T_anom)

    return grid


# def add_origin_axes(plotter, grid):
#     """
#     Draw x, y, z axes starting from the true origin (0, 0, 0).
#     """
#     bounds = grid.bounds

#     x_max = bounds[1]
#     y_max = bounds[3]
#     z_max = bounds[5]

#     origin = (0.0, 0.0, 0.0)

#     x_axis = pv.Line(origin, (x_max, 0.0, 0.0))
#     y_axis = pv.Line(origin, (0.0, y_max, 0.0))
#     z_axis = pv.Line(origin, (0.0, 0.0, z_max))

#     plotter.add_mesh(x_axis, color="black", line_width=5, label="Longitude")
#     plotter.add_mesh(y_axis, color="black", line_width=5, label="Latitude")
#     plotter.add_mesh(z_axis, color="black", line_width=5, label="Depth Levels")

#     plotter.show_bounds(
#     xlabel="Longitude",
#     ylabel="Latitude",
#     zlabel="Depth Levels",
# )

def add_selected_region_axes(plotter, grid, thick_axis_color):
    """
    Draw axes using the selected region's bounding box.
    """
    bounds = grid.bounds

    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3]
    z_min, z_max = bounds[4], bounds[5]

    # Corner of the selected region
    origin = (x_min, y_min, z_min)

    x_axis = pv.Line(origin, (x_max, y_min, z_min))
    y_axis = pv.Line(origin, (x_min, y_max, z_min))
    z_axis = pv.Line(origin, (x_min, y_min, z_max))

    plotter.add_mesh(x_axis, color=thick_axis_color, line_width=5)
    plotter.add_mesh(y_axis, color=thick_axis_color, line_width=5)
    plotter.add_mesh(z_axis, color=thick_axis_color, line_width=5)

def transform_z_axis(grid, z_max=90.0):
    """
    Transform z so that:
        old z = 0   -> new z = z_max
        old z = z_max -> new z = 0

    Example with z_max=90:
        z_new = 90 - z_old
    """
    grid = grid.copy()
    grid.points[:, 2] = z_max - grid.points[:, 2]
    return grid

# =========================================================
# 8. SAFE PLOTTING HELPER
# =========================================================

def safe_add_mesh(plotter, mesh, **kwargs):
    """
    Add a mesh only if it is non-empty.
    """
    if mesh is not None and mesh.n_points > 0:
        plotter.add_mesh(mesh, **kwargs)
        return True
    return False



# =========================================================
# 9. ADD SALINITY LAYERS
# =========================================================

def add_salinity_layers(
    plotter,
    grid,
    layer_percentiles,
    layer_opacities,
    layer_cmap,
    legend_color
):
    """
    Create salinity isosurfaces from user-selected salinity percentiles.

    Geometry:
        salinity isosurfaces

    Color:
        temperature anomaly
    """
    S_vals = grid.point_data["S"]
    S_vals = S_vals[np.isfinite(S_vals)]

    if S_vals.size == 0:
        print("No finite salinity values found.")
        return

    first_added = False

    for i, (pct, opacity) in enumerate(zip(layer_percentiles, layer_opacities)):
        salinity_level = np.percentile(S_vals, pct)

        print(
            f"Layer {i+1}: percentile={pct:.1f}, "
            f"salinity={salinity_level:.3f}, "
            f"opacity={opacity:.2f}"
        )

        layer = grid.contour(
            isosurfaces=[float(salinity_level)],
            scalars="S",
        )

        added = safe_add_mesh(
            plotter,
            layer,
            scalars="T_anom",
            cmap=layer_cmap,
            clim=T_ANOM_CLIM,
            opacity=float(opacity),
            show_scalar_bar=not first_added,
            scalar_bar_args={
                "title": "Temperature anomaly (°C)",
                "color": legend_color,
            },
        )

        if added:
            first_added = True
        else:
            print(f"Empty salinity layer at percentile {pct:.1f}")

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_float(value, default):
    """
    Convert UI value to float safely.
    If value is missing, empty, or invalid, use default.
    """
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def get_salinity_layer_controls():
    """
    Get salinity layer settings from the UI.

    Supports fewer than 4 layers safely.
    """
    n = int(safe_float(getattr(state, "num_salinity_layers", 4), 4))
    n = int(clamp(n, 0, 4))

    layer_cmap = getattr(state, "salinity_cmap", "coolwarm")

    if n == 0:
        return [], [], layer_cmap

    default_percentiles = [20.0, 40.0, 60.0, 80.0]
    default_opacities = [0.45, 0.45, 0.45, 0.45]

    layer_percentiles = []
    layer_opacities = []

    for i in range(n):
        pct_name = f"salinity_pct_{i+1}"
        opacity_name = f"salinity_opacity_{i+1}"

        pct_value = safe_float(
            getattr(state, pct_name, default_percentiles[i]),
            default_percentiles[i],
        )

        opacity_value = safe_float(
            getattr(state, opacity_name, default_opacities[i]),
            default_opacities[i],
        )

        pct_value = clamp(pct_value, 1.0, 99.0)
        opacity_value = clamp(opacity_value, 0.05, 1.0)

        layer_percentiles.append(pct_value)
        layer_opacities.append(opacity_value)

    layer_cmap = getattr(state, "salinity_cmap", "coolwarm")

    return layer_percentiles, layer_opacities, layer_cmap


def get_salinity_values_from_percentiles(S, layer_percentiles):
    """
    Compute actual salinity values from the salinity layer percentiles.
    """
    S_vals = S[np.isfinite(S)]

    if len(S_vals) == 0:
        return []

    salinity_values = []

    for pct in layer_percentiles:
        salinity_value = np.percentile(S_vals, pct)
        salinity_values.append(float(salinity_value))

    return salinity_values

# =========================================================
# 10. ADD VERTICAL VELOCITY HOTSPOTS
# =========================================================

def add_velocity_hotspots(plotter, grid, percentile):
    """
    Show positive vertical velocity hotspots from the top percentile.
    Example:
      percentile = 99  -> top 1%
      percentile = 95  -> top 5%
    """
    W_vals = grid.point_data["W"]
    W_vals = W_vals[np.isfinite(W_vals)]

    positive_W = W_vals[W_vals > 0]

    if positive_W.size == 0:
        print("No positive vertical velocity values.")
        return

    threshold = np.percentile(positive_W, percentile)
    w_max = np.nanmax(positive_W)

    print(f"Hotspot percentile = {percentile}, threshold = {threshold:.3e}, Wmax = {w_max:.3e}")

    hotspots = grid.threshold(
        value=float(threshold),
        scalars="W",
    )

    added = safe_add_mesh(
        plotter,
        hotspots,
        #scalars="W",
        #cmap="Greens",
        #clim=(threshold, w_max),
        color = "green",
        opacity=0.90,
        show_scalar_bar=True,
    )

    if not added:
        print("Hotspot mesh is empty.")



def add_downwelling_hotspots(plotter, grid, percentile=99.0):
    """
    Add downwelling hotspots.

    Downwelling means strong negative vertical velocity.
    If percentile = 99, this selects the bottom 1% most negative W values.
    """
    W = grid["W"]
    W_vals = W[np.isfinite(W)]

    negative_W = W_vals[W_vals < 0]

    if len(negative_W) == 0:
        print("No negative vertical velocity values found.")
        return None

    # For percentile=99, use the 1st percentile of negative W values
    # because strongest downwelling is most negative.
    downwelling_threshold = np.percentile(negative_W, 100.0 - percentile)

    print(
        f"Downwelling threshold: W <= {downwelling_threshold}",
        flush=True,
    )

    w_min = np.nanmin(negative_W)

    downwelling = grid.threshold(
        value=[float(w_min), float(downwelling_threshold)],
        scalars="W",
    )

    if downwelling.n_points == 0:
        print("No downwelling hotspot mesh generated.")
        return None

    plotter.add_mesh(
        downwelling,
        #scalars="W",
        #cmap="RdPu",
        color = "Magenta",
        opacity=0.90,
        show_scalar_bar=True,
    )

    return downwelling

# =========================================================
# 11. ADD LAND
# =========================================================

def add_land(plotter, grid):
    """
    Add land as a gray mesh.

    land = 1 means land
    land = 0 means ocean
    """
    land = grid.threshold(
        value=0.5,
        scalars="land",
    )

    if land.n_points == 0:
        print("No land mesh generated.", flush=True)
        return None

    plotter.add_mesh(
        land,
        color="lightgray",
        opacity=0.5,
        show_scalar_bar=False,
    )

    return land


# =========================================================
# 12. SCENE UPDATE
# =========================================================

# Keep only the latest loaded salinity array on the server. The browser receives
# a small PNG, not the full 3D dataset. Percentiles are a snapshot of the controls
# applied with Load / Update, so the histogram and isosurfaces stay consistent.
_histogram_data = None


def make_salinity_histogram_image(S, time_index, percentiles):
    """Return a density-histogram PNG data URL and a short sample summary.

    Samples are equally weighted (not ocean-cell-volume weighted). All depths
    in READ_KWARGS are included at the selected read quality.
    """
    values = np.asarray(S)
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return "", "No valid ocean salinity values in this region."

    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=120)
    try:
        # An integer bin count spans the entire finite data range. Do not clip
        # to 30--40: that would normalize only the samples within that range.
        ax.hist(values, bins=100, density=True, color="#4682b4",
                edgecolor="white", linewidth=0.3, alpha=0.85)
        styles = ["--", "-.", ":", (0, (5, 2, 1, 2, 1, 2))]
        levels = np.percentile(values, percentiles) if percentiles else []
        for i, (pct, level) in enumerate(zip(percentiles, levels)):
            ax.axvline(float(level), color="red", linewidth=1.8,
                       linestyle=styles[i % len(styles)],
                       label=f"Layer {i + 1}: P{pct:g} = {level:.3f} g/kg")

        ax.set_xlabel("Salinity (g/kg)")
        ax.set_ylabel("Probability density (g/kg)$^{-1}$")
        date = get_datetime_from_timestep(time_index).strftime("%Y-%m-%d %H:%M")
        ax.set_title(f"Salinity PDF — timestep {int(time_index)}\n{date}", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        if percentiles:
            ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png", facecolor="white")
            image = base64.b64encode(buffer.getvalue()).decode("ascii")
        return "data:image/png;base64," + image, (
            f"{values.size:,} valid samples; all selected depth levels. "
            "Equal sample weighting; histogram area = 1."
        )
    finally:
        plt.close(fig)


def refresh_salinity_histogram():
    """Render on demand, without reading another timestep or rebuilding VTK."""
    if not state.show_histogram or _histogram_data is None:
        return
    if state.histogram_image:
        return  # Reopening an unchanged histogram needs no computation.
    S, time_index, percentiles = _histogram_data
    try:
        state.histogram_image, state.histogram_message = make_salinity_histogram_image(
            S, time_index, percentiles
        )
    except Exception as exc:
        state.histogram_image = ""
        state.histogram_message = f"Could not generate histogram: {exc}"
        print(state.histogram_message, flush=True)


plotter = pv.Plotter()
plotter.set_background("white")



def update_scene(time_index, hotspot_percentile):
    """
    Rebuild the scene for a selected time and hotspot percentile.
    """
    print(f"Updating scene: time={time_index}, hotspot_percentile={hotspot_percentile}")

    plotter.clear()
    if state.plot_bg == "light":
        plotter.set_background("white")
        axis_color = "#1e1e1e"
        grid_color = "#1e1e1e"
        legend_color = "black"
        thick_axis_color = "black"
    else: 
        # dark backgound
        plotter.set_background("#1e1e1e")
        axis_color = "white"
        grid_color = "white"
        legend_color = "white"
        thick_axis_color = "#edfd04"   # bright yellow

    global T_ref, _histogram_data
    # Read data
    # T, S, W, land_mask = read_time_step(time_index=time_index)
    T, S, W, land_mask, lat2d, lon2d = read_time_step(time_index=time_index)

    T_ref = np.nanmean(T)
    print("Computed T_ref =", T_ref, flush=True)

    # T_ref = get_reference_temperature()

    T_anom = T - T_ref

    print("T shape:", T.shape)
    print("Finite T:", np.isfinite(T).sum())
    print("Finite S:", np.isfinite(S).sum())
    print("Finite W:", np.isfinite(W).sum())

    # Convert datetime
    current_time = get_datetime_from_timestep(time_index)
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Build grid
    # grid = make_grid(T, S, W, land_mask)
    grid = make_grid(T, S, W, land_mask, lat2d, lon2d, T_anom)

    # Transform z: 0 -> 90 and 90 -> 0
    #grid = transform_z_axis(grid, z_max=90.0)

    print("Grid points:", grid.n_points)
    print("Grid dimensions:", grid.dimensions)

    # Add outline
    plotter.add_mesh(
        grid.outline(),
        color=grid_color,
    )

    # add_origin_axes(plotter, grid)

    # plotter.add_mesh(grid.outline(), color="white")

    add_selected_region_axes(plotter, grid, thick_axis_color=thick_axis_color)

    # Add salinity layers colored by temperature anomaly
    layer_percentiles, layer_opacities, layer_cmap = get_salinity_layer_controls()
    layer_salinity_values = get_salinity_values_from_percentiles(S, layer_percentiles,)

    layer_info_lines = []

    for i, (pct, sal, opacity) in enumerate(
        zip(layer_percentiles, layer_salinity_values, layer_opacities),
        start=1,
    ):
        layer_info_lines.append(
            f"Layer {i}:  percentile {pct:.1f} , salinity {sal:.4f} g/kg, opacity {opacity:.2f}"
        )

    layer_info_text = "\n".join(layer_info_lines)

    state.plot_info = f"""
        This figure plots a subregion of the North Atlantic Ocean salinity isosurfaces colored by temperature anomaly. 
        The colored regions(Hotspots) indicate locations where vertical velocity is beyond a certain percentile. 
        Users can use the slidebar and textbox to adjust numbers. 
        Once finished, click the Load/Update button to wait for the data to load and the changes to take effect. 

        Time index: {time_index}
        Current simulation time: {current_time_str}
        Hotspot percentile: {hotspot_percentile}

        Salinity layers: {len(layer_percentiles)}
        {layer_info_text}
        Layer colormap: {layer_cmap}

        Surface geometry: salinity isosurfaces
        Surface color: temperature anomaly
        Colored regions: positive vertical velocity extremes(green/yellow), negative vertical velocity extremes(purple/magenta)
        """

    # Add land if checkbox is enabled (default = True)
    if state.show_land:
        add_land(plotter, grid)

    add_salinity_layers(
        plotter,
        grid,
        layer_percentiles=layer_percentiles,
        layer_opacities=layer_opacities,
        layer_cmap=layer_cmap,
        legend_color=legend_color
    )

    # Add vertical velocity hotspots if checkbox is enabled (default = True)
    if state.show_hotspots:
        add_velocity_hotspots(
            plotter,
            grid,
            percentile=float(hotspot_percentile),
        )

        add_downwelling_hotspots(
            plotter,
            grid,
            percentile=float(hotspot_percentile),
        )

    plotter.show_bounds(
        bounds=grid.bounds,
        xtitle="Longitude",
        ytitle="Latitude",
        ztitle="Depth",
        color=axis_color,
        grid=True,
        location="outer",
    )

    plotter.camera_position = "iso"

    # Commit histogram data only after the selected scene has been built.
    _histogram_data = (S, int(time_index), list(layer_percentiles))
    state.histogram_image = ""
    state.histogram_message = ""
    state.histogram_region = (
        f"x={READ_KWARGS['x']}, y={READ_KWARGS['y']}, "
        f"z={READ_KWARGS['z']}; quality={READ_KWARGS['quality']}"
    )
    refresh_salinity_histogram()

    # Update browser view if available
    try:
        ctrl.view_update()
    except Exception:
        pass


# =========================================================
# 13. TRAME APP
# =========================================================

print("Creating server...", flush=True)

server = get_server()
state, ctrl = server.state, server.controller

state.ui_theme = "light"
state.plot_bg = "light"

state.show_land = True

state.time_index = 0
state.hotspot_percentile = 99.0

state.show_hotspots = True

state.show_histogram = False
state.histogram_image = ""
state.histogram_region = ""
state.histogram_message = 'Click "Load / Update Time Step" to load the selected data.'


@state.change("show_histogram")
def on_histogram_visibility_change(show_histogram, **kwargs):
    if show_histogram:
        refresh_salinity_histogram()

# Salinity layer controls
state.num_salinity_layers = 4

state.salinity_pct_1 = 20.0
state.salinity_pct_2 = 40.0
state.salinity_pct_3 = 60.0
state.salinity_pct_4 = 80.0

state.salinity_opacity_1 = 0.45
state.salinity_opacity_2 = 0.45
state.salinity_opacity_3 = 0.45
state.salinity_opacity_4 = 0.45

state.salinity_cmap = "coolwarm"

state.salinity_cmap_options = [
    "coolwarm",
    "RdBu_r",
    "viridis",
    "plasma",
    "turbo",
    "cividis",
]

state.plot_info = """
No data loaded yet.

Click "Load / Update Time Step" to generate the salinity layers and vertical velocity hotspots.
"""

#@state.change("time_index", "hotspot_percentile")

#def on_ui_change(time_index, hotspot_percentile, **kwargs):
#    update_scene(
#        time_index=int(time_index),
#        hotspot_percentile=float(hotspot_percentile),
#    )


# Initial scene
#update_scene(
#    time_index=state.time_index,
#    hotspot_percentile=state.hotspot_percentile,
#)
def load_current_time():
    update_scene(
        time_index=int(state.time_index),
        hotspot_percentile=float(state.hotspot_percentile),
    )

ctrl.load_current_time = load_current_time

# =========================================================
# 14. UI LAYOUT
# =========================================================
print("Building UI...", flush=True)

with SinglePageWithDrawerLayout(server, theme = ("ui_theme", "light")) as layout:
    
    layout.title.set_text("ECCO Thermohaline Structure Explorer (North Atlantic Ocean)")


    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VSwitch(
            v_model=("show_histogram", False),
            label="Histogram",
            color="primary",
            density="compact",
            hide_details=True,
            inset=True,
            classes="mr-4",
            style="flex: 0 0 auto;",
        )
        vuetify.VSwitch(
            v_model=("ui_theme", "light"),
            label="Dark mode",
            true_value="dark",
            false_value="light",
            density="compact",
            hide_details=True,
            inset=True,
            classes="mr-4",
            style="flex: 0 0 auto;",
        )

    with layout.drawer:

        vuetify.VDivider(classes="my-4")

        vuetify.VSlider(
            v_model=("time_index", 0),
            min=TIME_MIN,
            max=TIME_MAX,
            step=10,
            label="Time index",
            thumb_label=True,
        )
        vuetify.VTextField(
            v_model=("time_index", 0),
            label="Time index (exact value)",
            type="number",
            min=TIME_MIN,
            max=TIME_MAX,
            step=1,
            density="compact",
        )

        vuetify.VSlider(
            v_model=("hotspot_percentile", 99.0),
            min=0,
            max=99.9,
            step=0.1,
            label="Hotspot percentile",
            thumb_label=True,
        )

        vuetify.VTextField(
            v_model=("num_salinity_layers", 4),
            min=0,
            max=4,
            step=1,
            label="Number of salinity layers",
            type="number",
            density="compact",
        )
        with vuetify.VContainer(v_show=("num_salinity_layers >= 1",), fluid=True):
           vuetify.VTextField(
                v_model=("salinity_pct_1", 20.0),
                label="Layer 1 salinity percentile",
                type="number",
                min=1,
                max=99,
                step=1,
                density="compact",
        )
        with vuetify.VContainer(v_show=("num_salinity_layers >= 2",), fluid=True):
            vuetify.VTextField(
                v_model=("salinity_pct_2", 40.0),
                label="Layer 2 salinity percentile",
                type="number",
                min=1,
                max=99,
                step=1,
                density="compact",
            )
        with vuetify.VContainer(v_show=("num_salinity_layers >= 3",), fluid=True):
            vuetify.VTextField(
                v_model=("salinity_pct_3", 60.0),
                label="Layer 3 salinity percentile",
                type="number",
                min=1,
                max=99,
                step=1,
                density="compact",
            )
        with vuetify.VContainer(v_show=("num_salinity_layers >= 4",), fluid=True):
            vuetify.VTextField(
                v_model=("salinity_pct_4", 80.0),
                label="Layer 4 salinity percentile",
                type="number",
                min=1,
                max=99,
                step=1,
                density="compact",
            )
        with vuetify.VContainer(v_show=("num_salinity_layers >= 1",), fluid=True):
            vuetify.VTextField(
                v_model=("salinity_opacity_1", 0.45),
                label="Layer 1 opacity",
                type="number",
                min=0.05,
                max=1.0,
                step=0.05,
                density="compact",
            )
        with vuetify.VContainer(v_show=("num_salinity_layers >= 2",), fluid=True):
            vuetify.VTextField(
                v_model=("salinity_opacity_2", 0.45),
                label="Layer 2 opacity",
                type="number",
                min=0.05,
                max=1.0,
                step=0.05,
                density="compact",
            )
        with vuetify.VContainer(v_show=("num_salinity_layers >= 3",), fluid=True):
            vuetify.VTextField(
                v_model=("salinity_opacity_3", 0.45),
                label="Layer 3 opacity",
                type="number",
                min=0.05,
                max=1.0,
                step=0.05,
                density="compact",
            )
        with vuetify.VContainer(v_show=("num_salinity_layers >= 4",), fluid=True):
            vuetify.VTextField(
                v_model=("salinity_opacity_4", 0.45),
                label="Layer 4 opacity",
                type="number",
                min=0.05,
                max=1.0,
                step=0.05,
                density="compact",
            )

        vuetify.VSelect(
            v_model=("salinity_cmap", "coolwarm"),
            items=("salinity_cmap_options",),
            label="Layer colormap",
        )

        vuetify.VSelect(
            v_model=("plot_bg", "light"),
            items=(["light", "dark"],),
            label="Plot background color",
            #density="compact",
        )

        vuetify.VCheckbox(
            v_model=("show_land", True),
            label="Show land",
            density="compact",
        )

        vuetify.VCheckbox(
            v_model=("show_hotspots", True),
            label="Show vertical velocity hotspots",
            density="compact",
        )

        with vuetify.VCard(classes="pa-3 ma-2", variant="tonal"):
            vuetify.VCardTitle("Plot Information")

            html.Div(
                "{{ plot_info }}",
                style="white-space: pre-line; font-size: 12px;",
            )

        vuetify.VBtn(
            "Load / Update Time Step",
            click=ctrl.load_current_time,
            block=True,
        )

    with layout.content:
        with html.Div(style="position: relative; width: 100%; height: 100%;"):
            view = plotter_ui(plotter)
            ctrl.view_update = view.update

            # Floating top-right panel; hiding it does not resize the 3D view.
            with vuetify.VCard(
                v_show=("show_histogram",),
                elevation=6,
                style=("position: absolute; top: 12px; right: 12px; z-index: 5; "
                       "width: min(540px, calc(100% - 24px)); "
                       "max-height: calc(100% - 24px); overflow-y: auto;"),
            ):
                with html.Div(classes="d-flex align-center px-3 pt-2"):
                    html.Div("Salinity histogram", classes="text-subtitle-1")
                    vuetify.VSpacer()
                    vuetify.VBtn("Close", variant="text", size="small",
                                 click="show_histogram = false")
                html.Img(
                    v_if=("histogram_image",),
                    src=("histogram_image",),
                    alt="Salinity probability density with selected percentile lines",
                    style="display: block; width: 100%; height: auto;",
                )
                html.Div("{{ histogram_message }}", classes="px-3 pb-2",
                         style="font-size: 12px;")
                html.Div("{{ histogram_region }}", classes="px-3 pb-2",
                         style="font-size: 11px; overflow-wrap: anywhere;")
                html.Div(
                    "Uses the timestep and active layer percentiles from the last "
                    "Load / Update. Edit the left controls, then click Load / Update "
                    "to apply changes to both plots.",
                    classes="px-3 pb-3", style="font-size: 12px; color: #555;",
                )


# =========================================================
# 15. START
# =========================================================


if __name__ == "__main__":
    # plot_salinity_histograms([2000, 4000, 6000, 8000])

    print("Starting server...", flush=True)
    server.start()
