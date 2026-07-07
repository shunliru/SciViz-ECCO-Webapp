import os
import numpy as np
import openvisuspy as ovp

os.environ["VISUS_CACHE"] = os.environ.get(
    "VISUS_CACHE",
    "./visus_cache_can_be_erased"
)

# Replace these with your actual dataset paths/URLs
vertical_velocity = "pelican://osg-htc.org/nasa/nsdf/climate2/llc4320/idx/w/w_llc4320_x_y_depth.idx"
salinity = "pelican://osg-htc.org/nasa/nsdf/climate1/llc4320/idx/salt/salt_llc4320_x_y_depth.idx"
temperature="pelican://osg-htc.org/nasa/nsdf/climate1/llc4320/idx/theta/theta_llc4320_x_y_depth.idx"



def read_ecco_subvolume(
    time=0,
    z=(0, 90),
    x=(14000, 17200),
    y=(8000, 11000),
    quality=-6,
):
    """
    Reads vertical velocity W and salinity S from ECCO.

    Returns:
        W_masked: 3D vertical velocity, land set to NaN
        S:        3D salinity
    """

    w_ds = ovp.LoadDataset(vertical_velocity)
    s_ds = ovp.LoadDataset(salinity)
    t_ds = ovp.LoadDataset(temperature)
    
    W = w_ds.db.read(
        time=time,
        z=list(z),
        x=list(x),
        y=list(y),
        quality=quality,
    )

    S = s_ds.db.read(
        time=time,
        z=list(z),
        x=list(x),
        y=list(y),
        quality=quality,
    )

    T = t_ds.db.read(
        time=time,
        z=list(z),
        x=list(x),
        y=list(y),
        quality=quality,
    )

    # Land = salinity == 0
    land_mask = (S == 0)

    # Mask land in vertical velocity
    W_masked = np.where(~land_mask, W, np.nan)

    print("W shape:", W.shape)
    print("S shape:", S.shape)
    print("T shape:", T.shape)
    print("Land fraction:", np.mean(land_mask))
    print("W min/max:", np.nanmin(W_masked), np.nanmax(W_masked))

    return W_masked, S, T
