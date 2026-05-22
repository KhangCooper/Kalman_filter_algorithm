# 1D Position-Velocity Kalman Filter

This project simulates a 1D kinematic system (position and velocity) and applies a **Kalman Filter** to process measurement noise, thereby accurately estimating the true state of the system.

## Project Overview

The source code is written in Python and provides an intuitive approach to understanding the Kalman Filter algorithm. The project includes:
* **`KalmanFilter` class**: Implements the core logic of the algorithm using two main steps:
    * **Predict:** Estimates the current state based on the previous state and the mathematical model (State transition matrix `A`).
    * **Update:** Corrects the prediction based on actual measurement data (`z`) via the Kalman Gain (`K`).
* **`simulate_position_velocity_1d` function**: Generates simulated data including the true state, measurement noise, and measured data collected from sensors.

## Libraries

This project relies on standard Python libraries for numerical computation and plotting. You will need to install:
* `numpy`
* `matplotlib`

You can install them quickly using pip:
```bash
pip install numpy matplotlib

Using python to develop the Kalman Filter algorithm and draw the graph for illustration

