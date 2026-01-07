# SPROJ — IMU-Based Bench Press Form Monitoring

This repository contains firmware and Python tools for a senior project focused on IMU-based upper-limb motion tracking for bench press form monitoring.

## Repository Layout (as committed)

- `controller_side_wireless/`  
  ESP32 controller-side firmware for the wireless setup.

- `node_side_wireless/`  
  ESP32 IMU node firmware for the wireless setup.

- `master_wired/`  
  ESP32 firmware for the wired single-board IMU setup.

- `bench_fin.py`  
  Python pipeline for reading IMU data (NDJSON), applying calibration/mount correction, reconstructing a simplified upper-body model, and computing form features.

- `replay_3d.py`  
  Python 3D replay/visualization script for reviewing recorded sessions.

- `notebooks/`  
  Jupyter notebooks used for angle extraction, EDA, correlation analysis, and model training/experiments.

- `data/`  
  Data folder taken from Y. Min Ko et al CV data set.

- `SPROJ_report.pdf`  
  Project report.

- `bench_imu_untested__corrections.py`
    I found some errors in my initial implementation but at this point was unable to confirm with the hardware. Change imports in 3d vis to this file. 

