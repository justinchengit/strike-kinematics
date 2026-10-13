import argparse
from pathlib import Path

from src.kinematics import angle_kinematics, kinematics, parse_angle, prepare

p = argparse.ArgumentParser(description="DeepLabCut output to strike kinematics")
p.add_argument("dlc_path")
p.add_argument("--fps", type=float, default=50000)
p.add_argument("--pixels-per-meter", type=float, default=20000)
p.add_argument("--mass-kg", type=float, default=1e-5)
p.add_argument("--p-cutoff", type=float, default=0.6)
p.add_argument("--max-gap", type=int, default=5)
p.add_argument("--angle", action="append", default=[], type=parse_angle,
               help="name=a,b (segment a->b) or name=a,vertex,c (joint angle); repeatable")
a = p.parse_args()

clean = prepare(a.dlc_path, a.pixels_per_meter, a.p_cutoff, a.max_gap)
df = kinematics(clean, a.fps, a.mass_kg)
Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/kinematics.csv", index=False)
print(f"wrote outputs/kinematics.csv ({df['frame'].nunique()} frames x {df['bodypart'].nunique()} bodyparts)")

if a.angle:
    ang = angle_kinematics(clean, a.fps, a.angle)
    ang.to_csv("outputs/angles.csv", index=False)
    print(f"wrote outputs/angles.csv ({len(a.angle)} angles)")


def peak(g, col, absolute=False):
    v = g[col].abs() if absolute else g[col]
    if v.isna().all():
        return "n/a"
    i = v.idxmax()
    return f"{g.at[i, col]:.4g} @ {g.at[i, 'time_s'] * 1e3:.3f} ms"


print("\nPer-bodypart peaks")
print(f"{'bodypart':<14}{'valid':>7}  {'speed (m/s)':<24}{'accel (m/s^2)':<24}{'force (N)':<24}")
for bp, g in df.groupby("bodypart", sort=False):
    valid = f"{g['speed_mps'].notna().mean():.0%}"
    print(f"{bp:<14}{valid:>7}  {peak(g, 'speed_mps'):<24}{peak(g, 'accel_mps2'):<24}{peak(g, 'force_N'):<24}")

if a.angle:
    print("\nPer-angle summary")
    print(f"{'angle':<14}{'range (deg)':<18}{'sweep (deg)':>12}  {'peak omega (rad/s)':<26}{'peak alpha (rad/s^2)':<26}")
    for name, g in ang.groupby("angle", sort=False):
        lo, hi = g["angle_deg"].min(), g["angle_deg"].max()
        rng = f"{lo:.1f} to {hi:.1f}"
        print(f"{name:<14}{rng:<18}{hi - lo:>12.1f}  "
              f"{peak(g, 'ang_vel_rad_s', True):<26}{peak(g, 'ang_accel_rad_s2', True):<26}")
