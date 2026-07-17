import os
import numpy as np
import pyvista as pv
import openvisuspy as ovp

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import html
from pyvista.trame.ui import plotter_ui

from datetime import datetime, timedelta

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
    x=[14000, 17200],
    y=[8000, 11000],
    quality=-12,   # start coarse
)

# Fixed color range for temperature anomaly on salinity layers
T_ANOM_CLIM = (-20.0, 20.0)


# =========================================================
# 2. LOAD DATASETS
# =========================================================

t_ds = ovp.LoadDataset(temperature)
s_ds = ovp.LoadDataset(salinity)
w_ds = ovp.LoadDataset(vertical_velocity)


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


    return T, S, W, land_mask

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

# Use baseline time = 0
T0, S0, W0, land_mask = read_time_step(time_index=0)

# One scalar regional average temperature
T_ref = np.nanmean(T0)

print("Reference average temperature T_ref =", T_ref)


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

def make_grid(T, S, W, land_mask):
    """
    Build a PyVista ImageData grid with:
      - salinity S
      - vertical velocity W
      - temperature anomaly T_anom = T - T_ref
    """
    T_anom = T - T_ref

    nz, ny, nx = T.shape

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)

    grid.point_data["T"] = zyx_to_pyvista(T)
    grid.point_data["S"] = zyx_to_pyvista(S)
    grid.point_data["W"] = zyx_to_pyvista(W)
    grid.point_data["T_anom"] = zyx_to_pyvista(T_anom)
    grid["land"] = zyx_to_pyvista(land_mask.astype(float))

    return grid


def add_origin_axes(plotter, grid):
    """
    Draw x, y, z axes starting from the true origin (0, 0, 0).
    """
    bounds = grid.bounds

    x_max = bounds[1]
    y_max = bounds[3]
    z_max = bounds[5]

    origin = (0.0, 0.0, 0.0)

    x_axis = pv.Line(origin, (x_max, 0.0, 0.0))
    y_axis = pv.Line(origin, (0.0, y_max, 0.0))
    z_axis = pv.Line(origin, (0.0, 0.0, z_max))

    plotter.add_mesh(x_axis, color="black", line_width=5, label="Longitude")
    plotter.add_mesh(y_axis, color="black", line_width=5, label="Latitude")
    plotter.add_mesh(z_axis, color="black", line_width=5, label="Depth Levels")

    plotter.show_bounds(
    xlabel="Longitude",
    ylabel="Latitude",
    zlabel="Depth Levels",
)

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
                "title": "Temperature anomaly",
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
        scalars="W",
        cmap="Wistia",
        clim=(threshold, w_max),
        opacity=0.80,
        show_scalar_bar=False,
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
        scalars="W",
        cmap="RdPu",
        opacity=0.80,
        show_scalar_bar=False,
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
        opacity=0.7,
        show_scalar_bar=False,
    )

    return land


# =========================================================
# 12. SCENE UPDATE
# =========================================================

plotter = pv.Plotter()
plotter.set_background("white")



def update_scene(time_index, hotspot_percentile):
    """
    Rebuild the scene for a selected time and hotspot percentile.
    """
    print(f"Updating scene: time={time_index}, hotspot_percentile={hotspot_percentile}")

    plotter.clear()

    # Read data
    T, S, W, land_mask = read_time_step(time_index=time_index)

    print("T shape:", T.shape)
    print("Finite T:", np.isfinite(T).sum())
    print("Finite S:", np.isfinite(S).sum())
    print("Finite W:", np.isfinite(W).sum())

    # Convert datetime
    current_time = get_datetime_from_timestep(time_index)
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Build grid
    grid = make_grid(T, S, W, land_mask)

    # Transform z: 0 -> 90 and 90 -> 0
    #grid = transform_z_axis(grid, z_max=90.0)

    print("Grid points:", grid.n_points)
    print("Grid dimensions:", grid.dimensions)

    # Add outline
    plotter.add_mesh(
        grid.outline(),
        color="black",
    )

    add_origin_axes(plotter, grid)

    # Add salinity layers colored by temperature anomaly
    layer_percentiles, layer_opacities, layer_cmap = get_salinity_layer_controls()
    layer_salinity_values = get_salinity_values_from_percentiles(S, layer_percentiles,)

    layer_info_lines = []

    for i, (pct, sal, opacity) in enumerate(
        zip(layer_percentiles, layer_salinity_values, layer_opacities),
        start=1,
    ):
        layer_info_lines.append(
            f"Layer {i}:  percentile {pct:.1f} , salinity {sal:.4f} ppt, opacity {opacity:.2f}"
        )

    layer_info_text = "\n".join(layer_info_lines)

    state.plot_info = f"""
        This figure plots the North American east coast salinity isosurfaces colored by temperature anomaly. 
        The hotspots indicate locations where vertical velocity is beyond a certain percentile. 
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
        Hotspot field: positive vertical velocity(green), negative vertical velocity(purple)
        """

    add_land(plotter, grid)

    add_salinity_layers(
        plotter,
        grid,
        layer_percentiles=layer_percentiles,
        layer_opacities=layer_opacities,
        layer_cmap=layer_cmap,
    )

    # Add vertical velocity hotspots
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

    plotter.camera_position = "iso"

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

state.time_index = 0
state.hotspot_percentile = 99.0

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

with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("ECCO Thermohaline Structure Explorer (North America)")

    with layout.toolbar:
        vuetify.VSpacer()

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
        view = plotter_ui(plotter)
        ctrl.view_update = view.update


# =========================================================
# 15. START
# =========================================================

if __name__ == "__main__":
    print("Starting server...", flush=True)
    server.start()