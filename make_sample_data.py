from pathlib import Path

import numpy as np
import pandas as pd

FPS = 50000
PIXELS_PER_METER = 20000
N_FRAMES = 4000

rng = np.random.default_rng(0)
t = np.arange(N_FRAMES) / FPS

pivot = np.array([300.0, 200.0])
arm_px = 100.0
t0, tau = 2000 / FPS, 4 / FPS
theta = np.pi / 2 - (np.pi / 2) / (1 + np.exp(-(t - t0) / tau))

points = {
    "dactyl_tip": (pivot[0] + arm_px * np.cos(theta), pivot[1] + arm_px * np.sin(theta)),
    "dactyl_base": (np.full(N_FRAMES, pivot[0]), np.full(N_FRAMES, pivot[1])),
    "propodus": (np.full(N_FRAMES, 220.0), np.full(N_FRAMES, 210.0)),
    "rostrum": (np.full(N_FRAMES, 120.0), np.full(N_FRAMES, 260.0)),
}

data = {}
for bp, (x, y) in points.items():
    x = x + rng.normal(0, 0.3, N_FRAMES)
    y = y + rng.normal(0, 0.3, N_FRAMES)
    lik = rng.uniform(0.9, 1.0, N_FRAMES)
    if bp == "dactyl_tip":
        drop = rng.choice(N_FRAMES, 40, replace=False)
        lik[drop] = rng.uniform(0.05, 0.5, 40)
        lik[1200:1203] = 0.2
        lik[3000:3010] = 0.2
    data[("sim", bp, "x")] = x
    data[("sim", bp, "y")] = y
    data[("sim", bp, "likelihood")] = lik

df = pd.DataFrame(data)
df.columns = pd.MultiIndex.from_tuples(df.columns, names=["scorer", "bodyparts", "coords"])

Path("data").mkdir(exist_ok=True)
df.to_csv("data/sample_strike.csv")
df.to_hdf("data/sample_strike.h5", key="df_with_missing", mode="w")
print(f"wrote data/sample_strike.csv and .h5 ({N_FRAMES} frames, fps={FPS}, px/m={PIXELS_PER_METER})")
