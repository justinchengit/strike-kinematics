import numpy as np

FPS = 50000
PIXELS_PER_METER = 20000
N_FRAMES = 4000

t = np.arange(N_FRAMES) / FPS

pivot = np.array([300.0, 200.0])
arm_px = 100.0
t0, tau = 2000 / FPS, 4 / FPS
theta = np.pi / 2 - (np.pi / 2) / (1 + np.exp(-(t - t0) / tau))

points = {
    "dactyl_tip": (pivot[0] + arm_px * np.cos(theta), pivot[1] + arm_px * np.sin(theta)),
}
