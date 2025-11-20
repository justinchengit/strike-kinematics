import numpy as np
import pandas as pd

from src.kinematics import kinematics

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
