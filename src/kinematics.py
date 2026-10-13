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


def kinematics(df, fps, mass_kg):
    dt = 1.0 / fps
    parts = []
    for bp, g in df.groupby("bodypart", sort=False):
        x, y = g["x_m"].to_numpy(float), g["y_m"].to_numpy(float)
        vx, vy = savgol_segments(x, dt, 1), savgol_segments(y, dt, 1)
        ax, ay = savgol_segments(x, dt, 2), savgol_segments(y, dt, 2)
        accel = np.hypot(ax, ay)
        parts.append(pd.DataFrame({
            "frame": g["frame"].to_numpy(),
            "bodypart": bp,
            "time_s": g["frame"].to_numpy() * dt,
            "x_m": savgol_segments(x, dt),
            "y_m": savgol_segments(y, dt),
            "speed_mps": np.hypot(vx, vy),
            "accel_mps2": accel,
            "force_N": mass_kg * accel,
        }))
    return pd.concat(parts, ignore_index=True)


def parse_angle(spec):
    """'name=a,b' is the orientation of segment a->b; 'name=a,v,c' is the joint angle at v between v->a and v->c."""
    name, _, pts = spec.partition("=")
    pts = pts.split(",")
    if not name or len(pts) not in (2, 3):
        raise ValueError(f"bad angle spec {spec!r}; use name=a,b or name=a,vertex,c")
    return name, pts


def raw_angle(x, y, pts):
    if len(pts) == 2:
        a, b = pts
        return np.arctan2(y[b] - y[a], x[b] - x[a])
    a, v, c = pts
    ux, uy = x[a] - x[v], y[a] - y[v]
    wx, wy = x[c] - x[v], y[c] - y[v]
    return np.arctan2(ux * wy - uy * wx, ux * wx + uy * wy)


def unwrap_segments(theta):
    """Unwrap each NaN-free stretch, keeping each stretch within pi of where the previous one ended."""
    out = theta.copy()
    ok = ~np.isnan(theta)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    last = None
    for a, b in zip(edges[::2], edges[1::2]):
        out[a:b] = np.unwrap(theta[a:b])
        if last is not None:
            out[a:b] += 2 * np.pi * np.round((last - out[a]) / (2 * np.pi))
        last = out[b - 1]
    return out


def angle_kinematics(df, fps, angles):
    dt = 1.0 / fps
    x = df.pivot(index="frame", columns="bodypart", values="x_m")
    y = df.pivot(index="frame", columns="bodypart", values="y_m")
    parts = []
    for name, pts in angles:
        missing = [p for p in pts if p not in x.columns]
        if missing:
            raise KeyError(f"angle {name!r}: unknown bodypart(s) {missing}")
        theta = unwrap_segments(raw_angle(x, y, pts).to_numpy(float))
        if len(pts) == 3 and np.nanmedian(theta) < 0:
            theta = -theta
        parts.append(pd.DataFrame({
            "frame": x.index.to_numpy(),
            "angle": name,
            "time_s": x.index.to_numpy() * dt,
            "angle_deg": np.degrees(savgol_segments(theta, dt)),
            "ang_vel_rad_s": savgol_segments(theta, dt, 1),
            "ang_accel_rad_s2": savgol_segments(theta, dt, 2),
        }))
    cols = ["frame", "angle", "time_s", "angle_deg", "ang_vel_rad_s", "ang_accel_rad_s2"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def prepare(dlc_path, pixels_per_meter, p_cutoff=0.6, max_gap=5):
    df = load_dlc(dlc_path)
    df = filter_likelihood(df, p_cutoff)
    df = interpolate(df, max_gap)
    return to_meters(df, pixels_per_meter)


def run(dlc_path, fps, pixels_per_meter, mass_kg, p_cutoff=0.6, max_gap=5):
    return kinematics(prepare(dlc_path, pixels_per_meter, p_cutoff, max_gap), fps, mass_kg)
