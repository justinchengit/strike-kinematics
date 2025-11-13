import argparse

from src.kinematics import run

p = argparse.ArgumentParser(description="DeepLabCut output to strike kinematics")
p.add_argument("dlc_path")
p.add_argument("--fps", type=float, default=50000)
p.add_argument("--pixels-per-meter", type=float, default=20000)
p.add_argument("--mass-kg", type=float, default=1e-5)
p.add_argument("--p-cutoff", type=float, default=0.6)
p.add_argument("--max-gap", type=int, default=5)
a = p.parse_args()

df = run(a.dlc_path, a.fps, a.pixels_per_meter, a.mass_kg, a.p_cutoff, a.max_gap)
