import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

WINDOW = 11
POLYORDER = 3


def load_dlc(path):
    if str(path).endswith(".h5"):
        df = pd.read_hdf(path)
    else:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    df.columns = df.columns.droplevel(0)
    df.index.name = "frame"
    long = df.stack(level=0, future_stack=True).reset_index()
    long = long.rename(columns={long.columns[1]: "bodypart"})
    long = long[["frame", "bodypart", "x", "y", "likelihood"]]
    return long.sort_values(["bodypart", "frame"], ignore_index=True)


def filter_likelihood(df, p_cutoff=0.6):
    df = df.copy()
    df.loc[df["likelihood"] < p_cutoff, ["x", "y"]] = np.nan
    return df


def fill_short_gaps(s, max_gap=5):
    isna = s.isna()
    run_len = isna.groupby((isna != isna.shift()).cumsum()).transform("size")
    filled = s.interpolate(limit_area="inside")
    return filled.where(~isna | (run_len < max_gap))


def interpolate(df, max_gap=5):
    df = df.copy()
    for col in ["x", "y"]:
        df[col] = df.groupby("bodypart")[col].transform(lambda s: fill_short_gaps(s, max_gap))
    return df


def to_meters(df, pixels_per_meter):
    df = df.copy()
    df["x_m"] = df["x"] / pixels_per_meter
    df["y_m"] = df["y"] / pixels_per_meter
    return df


def savgol_segments(v, dt, deriv=0):
    out = np.full(len(v), np.nan)
    ok = ~np.isnan(v)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= WINDOW:
            out[a:b] = savgol_filter(v[a:b], WINDOW, POLYORDER, deriv=deriv, delta=dt)
    return out


def kinematics(df, fps):
    dt = 1.0 / fps
    out = {}
    for bp, g in df.groupby("bodypart", sort=False):
        x, y = g["x_m"].to_numpy(float), g["y_m"].to_numpy(float)
        vx, vy = savgol_segments(x, dt, 1), savgol_segments(y, dt, 1)
        out[bp] = {"speed_mps": np.hypot(vx, vy)}
    return out
