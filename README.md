# Robust Kalman Filtering for Attitude/Pitch Estimation (REKF & RUKF)

> **Course Project:** ELL7102 – Stochastic Process and Filtering  
> **Institution:** IIT Delhi  
> **Instructor:** Prof. Shubhendu Bhasin  

Course project implementing and comparing **standard vs. robust Kalman filters**
(EKF, UKF, REKF, RUKF) for two estimation problems:

1. **1‑DOF pitch angle estimation** from noisy sensor data, with outliers/faults
   injected to test robustness (`src/pitch_1d_filters/`).
2. **Full 6‑state pico‑satellite attitude estimation** (Euler angles + body rates)
   using a magnetometer-only measurement model, following the robust filtering
   approach of Söken & Hajiyev (2014) (`src/pico_satellite_REKF_RUKF/`).

The core idea in both cases: the standard EKF/UKF trusts every measurement
equally, so a single faulty/outlier measurement can throw the estimate off
and take a long time to recover. The robust versions (REKF/RUKF) monitor the
innovation (measurement residual) each step, and when it's statistically too
large, they **inflate the effective measurement noise covariance** via a
**measurement noise scale factor (MNSF)** — this automatically down-weights
the bad measurement in the Kalman gain instead of trusting it.

## Repository structure

```
.
├── src/
│   ├── pitch_1d_filters/
│   │   ├── ekf_pitch_baseline.m                 # Basic EKF on real pitch sensor data
│   │   └── robust_filters_pitch_comparison.m    # EKF/REKF/UKF/RUKF comparison
│   └── pico_satellite_REKF_RUKF/
│       └── satellite_attitude_sim.m             # Full 6-state nonlinear sim
│                                                 # (EKF/REKF/UKF/RUKF), with a
│                                                 # 300–301 s magnetometer fault
├── data/
│   ├── pitch_data.csv                # Time, True_Pitch, Sensor_Pitch
│   └── satellite_magnetometer_data.csv
├── results/
│   └── figures/                      # Generated comparison plots
├── docs/
│   ├── REKF_and_RUKF_for_Pico_Satellite_Attitude_Estimation.pptx
│   └── Presentation_Template_and_Evaluation.pdf
└── README.md
```

> **Note on the reference paper:** This project follows the method described in
> Söken, H. E., & Hajiyev, C. (2014). *REKF and RUKF for pico satellite attitude
> estimation in the presence of measurement faults.* Journal of Systems
> Engineering and Electronics, 25(2), 288–297.
> The PDF of that paper is **not included** in this repo (it's a published,
> copyrighted journal article) — cite/link it instead if needed.

## Part 1 — 1‑DOF pitch estimation (`src/pitch_1d_filters/`)

- **State**: `[pitch angle; pitch rate]`, constant-velocity model.
- **`ekf_pitch_baseline.m`**: straightforward linear/EKF baseline that reads
  `data/pitch_data.csv`, filters the noisy `Sensor_Pitch` measurements, and
  reports RMSE improvement over the raw sensor.
- **`robust_filters_pitch_comparison.m`**: runs EKF, REKF, UKF, and RUKF side
  by side on the same data. The robust filters compute a Mahalanobis-distance
  based test on the innovation each step; if it exceeds a threshold
  (`k_huber = 2.0` sigmas), the measurement noise is scaled up
  (`MNSF = (mah_dist / k_huber)^2`) before computing the Kalman gain.

**To run:**
```matlab
% From src/pitch_1d_filters/, with data/pitch_data.csv on the MATLAB path
% (or copy the CSV next to the script / update the path in the script)
robust_filters_pitch_comparison
```

## Part 2 — Pico-satellite attitude estimation (`src/pico_satellite_REKF_RUKF/`)

- **State**: `[phi, theta, psi, wx, wy, wz]` — 3 Euler angles + 3 body rates.
- **Dynamics**: gravity-gradient torque + rigid-body Euler equations,
  propagated with simple Euler integration.
- **Measurement**: orbit-referenced Earth magnetic field model (IGRF-style
  dipole approximation) rotated into the body frame via the DCM.
- **Fault**: a 50,000 nT bias is injected into one magnetometer axis between
  t = 300 s and t = 301 s to simulate a sensor malfunction.
- Implements EKF, REKF (numerically linearized), UKF, and RUKF, and plots the
  pitch angle estimate and MNSF evolution for each.

**To run:**
```matlab
% From src/pico_satellite_REKF_RUKF/
satellite_attitude_sim
```

## Results

| Filter pair | Behaviour under fault (300–301 s) |
|---|---|
| EKF vs REKF | EKF estimate spikes sharply at the fault and takes time to recover; REKF's MNSF jumps (rejecting the bad measurement) and the estimate stays close to truth. |
| UKF vs RUKF | Same pattern — UKF spikes, RUKF's scale factor absorbs the fault. |
| REKF vs RUKF | Both robust filters track the true pitch closely; RUKF is marginally smoother since it avoids explicit Jacobian linearization. |

See `results/figures/` for the generated plots (pitch tracking + MNSF vs.
time for each filter pair).

## Requirements

- MATLAB (tested with base MATLAB, no toolboxes required — `chol`, matrix
  operations, and `readtable` are the only non-trivial built-ins used).

## Author

Muhammed Aslam A — M.Tech, Electrical Engineering, IIT Delhi  
Govind U - M.S.R, Electrical Engineering, IIT Delhi


