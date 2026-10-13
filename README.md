# strike-kinematics

This is a kinematics pipeline I developed for the Patek Biomechanics Lab to automate analysis of claw features tracked in 1.2 million+ fps video.

It takes DeepLabCut pose output (the tracked x/y position of each body part in every frame) plus your camera calibration, and computes, for every frame:

- **for each tracked point:** position, speed, acceleration and force
- **for each angle you define:** angle, angular velocity and angular acceleration

## Quick start

```bash
pip install -r requirements.txt
python make_sample_data.py          # writes a simulated strike to data/sample_strike.h5
python run.py data/sample_strike.h5 \
    --fps 50000 --pixels-per-meter 20000 --mass-kg 1e-5 \
    --angle dactyl=dactyl_base,dactyl_tip \
    --angle hinge=propodus,dactyl_base,dactyl_tip
pytest                              # runs the tests
```

The sample is a simulated 4-point strike at 50,000 fps: `dactyl_tip` sweeps 90° about `dactyl_base`, and the other points sit still with tracking noise. For real data, point `run.py` at your own DeepLabCut `.h5` or `.csv` and use your own calibration.

## Usage

```
python run.py DLC_FILE --fps FPS --pixels-per-meter PPM --mass-kg KG [--angle SPEC ...]
```

| Option | Default | Meaning |
|---|---|---|
| `DLC_FILE` | (required) | DeepLabCut output, `.h5` or `.csv`, in DLC's standard 3-row header (scorer / bodyparts / coords) |
| `--fps` | 50000 | Camera frame rate (frames per second) |
| `--pixels-per-meter` | 20000 | Spatial calibration (pixels per meter, e.g. from a scale bar in the video) |
| `--mass-kg` | 1e-5 | Mass used for force = mass × acceleration |
| `--p-cutoff` | 0.6 | Points with DLC likelihood below this are treated as missing |
| `--max-gap` | 5 | Missing stretches shorter than this many frames are filled linearly; longer ones stay missing |
| `--angle` | none | An angle to track; repeat the flag for several (see below) |

### Defining angles

Use the body part names from your DLC project.

- **Segment angle**, `--angle NAME=A,B`: the orientation of the line from A to B, measured from the +x axis. Example: `dactyl=dactyl_base,dactyl_tip` gives the angle of the dactyl.
- **Joint angle**, `--angle NAME=A,V,C`: the angle at vertex V between V→A and V→C, i.e. the angle *between two segments*. Example: `hinge=propodus,dactyl_base,dactyl_tip` gives the opening angle between the propodus and the dactyl at their joint.

Angles are continuous: they don't jump at ±180°, so a part rotating past 180° keeps counting (190°, 200°, ...). Joint angles are oriented so they read positive. While a joint stays under 180°, the value is the interior angle.

## Output

Each run writes up to two CSVs to `outputs/` and prints a summary.

### Console summary (sample run above)

```
wrote outputs/kinematics.csv (4000 frames x 4 bodyparts)
wrote outputs/angles.csv (2 angles)

Per-bodypart peaks
bodypart        valid  speed (m/s)             accel (m/s^2)           force (N)
dactyl_base      100%  0.8457 @ 17.720 ms      1.621e+04 @ 79.980 ms   0.1621 @ 79.980 ms
dactyl_tip       100%  24.25 @ 40.000 ms       1.162e+05 @ 39.920 ms   1.162 @ 39.920 ms
propodus         100%  1.697 @ 79.980 ms       3.983e+04 @ 79.980 ms   0.3983 @ 79.980 ms
rostrum          100%  0.7654 @ 0.000 ms       1.526e+04 @ 0.000 ms    0.1526 @ 0.000 ms

Per-angle summary
angle         range (deg)        sweep (deg)  peak omega (rad/s)        peak alpha (rad/s^2)
dactyl        -0.4 to 90.3              90.7  -4946 @ 40.000 ms         -2.107e+07 @ 39.900 ms
hinge         82.3 to 173.7             91.3  4986 @ 40.000 ms          -2.266e+07 @ 40.120 ms
```

How to read it:

- **valid**: the share of frames with a usable value, after low-likelihood points are dropped and short gaps are filled.
- **peaks**: the largest value and when it happened. Here the dactyl tip hits 24.25 m/s at 40 ms, the middle of the strike.
- **sweep**: how far the angle travelled (max − min). The dactyl swings about 90°.
- **omega / alpha**: angular velocity and acceleration at their largest magnitude. The sign gives the direction: negative means the angle is decreasing.
- The still points (`dactyl_base`, `propodus`, `rostrum`) show small nonzero peaks. That is tracking noise, not motion, and it tells you the noise floor of the run (see Accuracy).

### `outputs/kinematics.csv` (one row per frame per body part)

| column | unit | meaning |
|---|---|---|
| `frame` | | frame index from the DLC file |
| `bodypart` | | tracked point |
| `time_s` | s | frame / fps |
| `x_m`, `y_m` | m | smoothed position |
| `speed_mps` | m/s | magnitude of velocity |
| `accel_mps2` | m/s² | magnitude of acceleration |
| `force_N` | N | mass × acceleration |

```
frame,bodypart,time_s,x_m,y_m,speed_mps,accel_mps2,force_N
0,dactyl_base,0.0,0.01501373114437583,0.010001717401467193,0.45559621660772937,6712.23768780664,0.06712237687806641
1,dactyl_base,2e-05,0.015009754126792976,0.009994553870568077,0.3684142703490864,5632.819578046559,0.0563281957804656
```

### `outputs/angles.csv` (one row per frame per angle, only with `--angle`)

| column | unit | meaning |
|---|---|---|
| `frame` | | frame index |
| `angle` | | the NAME you gave |
| `time_s` | s | frame / fps |
| `angle_deg` | deg | smoothed angle |
| `ang_vel_rad_s` | rad/s | angular velocity |
| `ang_accel_rad_s2` | rad/s² | angular acceleration |

```
frame,angle,time_s,angle_deg,ang_vel_rad_s,ang_accel_rad_s2
0,dactyl,0.0,90.14543710563086,-28.870445558695515,-1265823.684092581
1,dactyl,2e-05,90.09989796383911,-48.82206646146332,-729338.4061841995
```

A value is blank (NaN) wherever a point is missing, or wherever an unbroken run of valid frames is shorter than the 11-frame smoothing window. An angle needs every one of its points to be present.

## How it works

1. **Load:** read the DLC file into one row per frame per body part (x, y, likelihood).
2. **Filter:** points with likelihood below `--p-cutoff` become missing.
3. **Fill:** gaps shorter than `--max-gap` frames are linearly interpolated; longer gaps and gaps at the start or end stay missing.
4. **Convert:** pixels → meters using `--pixels-per-meter`.
5. **Smooth and differentiate:** a Savitzky-Golay filter (window 11 frames, polynomial order 3) gives smoothed position, velocity and acceleration in one step, applied separately to each unbroken stretch of data.
6. **Angles:** computed from the positions with `atan2`, unwrapped so they don't jump at ±180°, then smoothed and differentiated with the same filter.

Coordinates use DLC's image convention (x right, y down). In real video a positive segment angle therefore turns clockwise on screen. Speeds, accelerations and the size of each angle are unaffected.

## Accuracy

The sample strike is generated from a known formula, so the exact answer can be compared with what the pipeline reports:

| quantity | exact | pipeline | error |
|---|---|---|---|
| dactyl angle (RMS over all frames) | | | 0.11° |
| peak angular velocity | 4909 rad/s | 4946 rad/s | +0.8% |
| peak tip speed | 24.54 m/s | 24.25 m/s | −1.2% |
| peak angular acceleration | 2.356e7 rad/s² | 2.107e7 rad/s² | −10.6% |
| peak tip acceleration | 1.341e5 m/s² | 1.162e5 m/s² | −13.4% |

Angles, angular velocity and speed are accurate to about 1%. Peak accelerations come out low here because the sample's acceleration spike lasts only about 10 frames, roughly the width of the 11-frame smoothing window, so the filter flattens it. The fix is more frames per event. At 1.2 million fps the same strike covers about 24× as many frames, and this error becomes small.

The tests (`pytest`) check exact cases: constant velocity, constant acceleration, constant spin across ±180°, constant angular acceleration, and continuity across a tracking gap.
