# Narrow Field-of-View Missile Guidance

MATLAB simulation for missile guidance under a narrow seeker field-of-view constraint, developed for the 2026 KSAS Spring Conference.

The guidance system combines target-state estimation, target reacquisition, and terminal guidance to maintain interception capability when seeker measurements are temporarily unavailable.

## Key Features

- 3D missile–target engagement simulation
- Narrow seeker Field-of-View constraint
- Bearing-only target estimation using Extended Kalman Filter (EKF)
- Prediction-based state estimation during target loss
- Monte Carlo prediction of target position uncertainty
- Target reacquisition using an uncertainty-based waypoint
- Proportional Navigation (PN) guidance
- Helical Navigation Guidance (HNG)
- FOV-constrained acceleration control
- Multi-phase guidance and terminal interception logic

## Guidance Flow

**Phase 1 — Target Uncertainty Prediction**  
Monte Carlo simulation predicts the possible target-position distribution during the blind-flight period.  
A waypoint and required altitude are generated from the predicted uncertainty region.

**Phase 2 — Target Reacquisition**  
The missile approaches the predicted target-distribution center until the target is detected again and the position uncertainty decreases.

**Phase 3 — FOV-Constrained Guidance**  
PN guidance, helical guidance, and FOV correction are combined to maintain target tracking while reducing interception error.

**Phase 4 — Terminal Guidance**  
The missile switches to terminal PN guidance for final interception.

## Simulation Outputs

The script visualizes:

- 3D missile and target trajectories
- Monte Carlo target probability cloud
- Seeker look angle and FOV limit
- True and estimated target range
- Missile acceleration profile
- PN / FOV / Helical guidance acceleration

## File

`KSAS_Final.m` — Main MATLAB simulation and visualization script

## Environment

- MATLAB
