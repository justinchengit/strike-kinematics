import numpy as np
import pandas as pd

from src.kinematics import angle_kinematics, kinematics, unwrap_segments

FPS = 1000.0


def make_track(x):
    n = len(x)
    return pd.DataFrame({"frame": np.arange(n), "bodypart": "p", "x_m": x, "y_m": np.zeros(n)})


def test_constant_acceleration():
    a = 9.81
    t = np.arange(200) / FPS
    out = kinematics(make_track(0.5 * a * t**2), FPS, mass_kg=1.0)
    np.testing.assert_allclose(out["accel_mps2"], a, rtol=1e-3)


def test_constant_velocity():
    t = np.arange(200) / FPS
    out = kinematics(make_track(3.0 * t), FPS, mass_kg=1.0)
    np.testing.assert_allclose(out["accel_mps2"], 0.0, atol=1e-6)


def make_points(points):
    rows = []
    for bp, (x, y) in points.items():
        rows.append(pd.DataFrame({"frame": np.arange(len(x)), "bodypart": bp, "x_m": x, "y_m": y}))
    return pd.concat(rows, ignore_index=True)


def test_segment_angle_constant_omega_across_pi():
    w = 50.0
    t = np.arange(300) / FPS
    th = 3.0 + w * t
    pts = make_points({"a": (np.zeros_like(t), np.zeros_like(t)), "b": (np.cos(th), np.sin(th))})
    out = angle_kinematics(pts, FPS, [("seg", ["a", "b"])])
    np.testing.assert_allclose(out["ang_vel_rad_s"], w, rtol=1e-6)
    np.testing.assert_allclose(out["ang_accel_rad_s2"], 0.0, atol=1e-6)
    np.testing.assert_allclose(out["angle_deg"], np.degrees(th), atol=1e-6)


def test_joint_angle_constant_alpha():
    alpha = 400.0
    t = np.arange(200) / FPS
    th = 0.3 + 0.5 * alpha * t**2
    z = np.zeros_like(t)
    pts = make_points({"a": (z + 2.0, z), "v": (z, z), "c": (np.cos(th), np.sin(th))})
    out = angle_kinematics(pts, FPS, [("joint", ["a", "v", "c"])])
    np.testing.assert_allclose(out["ang_accel_rad_s2"], alpha, rtol=1e-6)
    np.testing.assert_allclose(out["ang_vel_rad_s"], alpha * t, atol=1e-9)
    np.testing.assert_allclose(out["angle_deg"], np.degrees(th), atol=1e-6)


def test_unwrap_continues_across_gap():
    t = np.arange(100) / FPS
    th = 3.0 + 60.0 * t
    raw = np.arctan2(np.sin(th), np.cos(th))
    raw[40:44] = np.nan
    out = unwrap_segments(raw)
    ok = ~np.isnan(raw)
    np.testing.assert_allclose(out[ok], th[ok], atol=1e-9)
