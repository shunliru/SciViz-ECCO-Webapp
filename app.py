import os
import numpy as np
import pyvista as pv
import openvisuspy as ovp

from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import vuetify3 as vuetify
from trame.widgets import html
from pyvista.trame.ui import plotter_ui



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

    T = np.where(ocean_mask, T, np.nan)
    S = np.where(ocean_mask, S, np.nan)
    W = np.where(ocean_mask, W, np.nan)

    return T, S, W


# =========================================================
# 4. REFERENCE TEMPERATURE (OPTION A)
# =========================================================

# Use baseline time = 0
T0, S0, W0 = read_time_step(time_index=0)

# One scalar regional average temperature
T_ref = np.nanmean(T0)

print("Reference average temperature T_ref =", T_ref)


# =========================================================
# 5. ARRAY CONVERSION
# =========================================================

def zyx_to_pyvista(arr):
    """
    Convert array from (z, y, x) to PyVista point ordering.
    """
    arr_xyz = np.transpose(arr, (2, 1, 0))   # (x, y, z)
    return arr_xyz.ravel(order="F")


# =========================================================
# 6. BUILD GRID
# =========================================================

def make_grid(T, S, W):
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

    return grid


# =========================================================
# 7. SAFE PLOTTING HELPER
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
# 8. ADD SALINITY LAYERS
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
    n = int(clamp(n, 1, 4))

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

# =========================================================
# 9. ADD VERTICAL VELOCITY HOTSPOTS
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


# =========================================================
# 10. SCENE UPDATE
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
    T, S, W = read_time_step(time_index=time_index)

    print("T shape:", T.shape)
    print("Finite T:", np.isfinite(T).sum())
    print("Finite S:", np.isfinite(S).sum())
    print("Finite W:", np.isfinite(W).sum())

    # Build grid
    grid = make_grid(T, S, W)

    print("Grid points:", grid.n_points)
    print("Grid dimensions:", grid.dimensions)

    # Add outline
    plotter.add_mesh(
        grid.outline(),
        color="black",
    )

    # Add salinity layers colored by temperature anomaly
    layer_percentiles, layer_opacities, layer_cmap = get_salinity_layer_controls()

    state.plot_info = f"""
        This figure plots the North American east coast salinity isosurfaces colored by temperature anomaly. 
        The hotspots indicate locations where vertical velocity is beyond a certain percentile. 
        Users can use the slidebar and textbox to adjust numbers. 
        Once finished, click the Load/Update button to wait for the data to load and the changes to take effect. 

        Time index: {time_index}
        Hotspot percentile: {hotspot_percentile}

        Salinity layers: {len(layer_percentiles)}
        Layer percentiles: {layer_percentiles}
        Layer opacities: {layer_opacities}
        Layer colormap: {layer_cmap}

        Surface geometry: salinity isosurfaces
        Surface color: temperature anomaly
        Hotspot field: positive vertical velocity
        """

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

    plotter.camera_position = "iso"

    # Update browser view if available
    try:
        ctrl.view_update()
    except Exception:
        pass


# =========================================================
# 11. TRAME APP
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
# 12. UI LAYOUT
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
            min=1,
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
# 13. START
# =========================================================

if __name__ == "__main__":
    print("Starting server...", flush=True)
    server.start()