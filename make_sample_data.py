import numpy as np

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
    data[("sim", bp, "x")] = x
    data[("sim", bp, "y")] = y
