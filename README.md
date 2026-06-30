# neuromorphic-crossbar-sim

A research-oriented simulation framework for studying neuromorphic crossbar arrays built from memristor and memcapacitor devices. Built for undergraduate research at the Portland State DISC Lab (Project 18, Summer Proceedings 2026), investigating how device-level non-idealities, including parasitic wire resistance and device variability, propagate to network-level behavior in mixed memristive/memcapacitive crossbar arrays.

## Repository structure

### `devices/`
Validated device models, the core reusable physics layer.
- `TeamMemristor.py`: TEAM threshold-switching memristor model (Kvatinsky et al. 2013)
- `FracMemCap.py`: Biolek memcapacitor model (Biolek et al. 2010), integer-order solver validated, fractional-order solver implemented but not yet validated under the crossbar/MNA path
- `device.py`, `memristive.py`, `memcapacitive.py`: abstract base interfaces shared by both device types

### `sim/`
The simulation framework built on top of validated devices.
- `crossbar/crossbar.py`: N×N crossbar with Kirchhoff-consistent current summation. Supports an ideal (zero-resistance) mode and a Modified Nodal Analysis (MNA) solver for row and column parasitic wire resistance. Devices are accessed through a common interface (`network_step`, `network_current`, `current_conductance`, `current_offset`), so the crossbar has no knowledge of device-specific physics.
- `examples/crossDemo.py`: validation suite for the crossbar, confirms MNA matches the ideal path at zero resistance, confirms correct voltage sag under row/column parasitics, and validates mixed memristor/memcapacitor arrays
- `plotting/`: I-V and Q-V plotting utilities for individual devices

### `Sandbox/`
Earlier, pre-refactor implementations of the device models and standalone packages (`fractional_memcapacitor/`, `Team_memristor/`). Kept for reference and for the validation work done against published curves before the current `devices/` package existed. Not part of the active framework.

### `testing/`
Xyce/SPICE netlists and supporting scripts used to cross-check device and crossbar behavior against an independent circuit simulator. Includes single-device netlists (`singleMemristor.cir`, `singleMemCap.cir`) and 4x4 crossbar netlists (`memResCrossBar4x4.cir`, `memCapCrossBar4x4.cir`, `resCrossBar4x4.cir`).

## Current status

- TEAM memristor: validated against published I-V curves, validated under MNA with and without parasitic resistance
- Biolek memcapacitor (integer-order): validated under MNA at zero resistance, companion model confirmed numerically identical to standalone device behavior
- Crossbar: device-agnostic architecture, MNA solver handles row and column wire resistance simultaneously, validated with mixed memristor/memcapacitor arrays
- Fractional-order memcapacitor solver: implemented, not yet validated under MNA

## Typical workflow

### Python crossbar simulation
python -m sim.examples.crossDemo

### Xyce/SPICE cross-validation
cd testing

Xyce singleMemristor.cir

cat singleMemristor.cir.prn

## Roadmap

Per the project plan (Summer Proceedings 2026):
- Weeks 1-2: device models (complete)
- Weeks 3-4: crossbar assembly with parasitic resistance, mixed device columns (in progress)
- Weeks 5-6: train a small fully connected network in PyTorch, map weights to crossbar conductances
- Weeks 7-8: Monte Carlo simulation over device-to-device variability from CMOS-integrated RRAM distributions
- Weeks 9-10: optional reservoir computing reconfiguration, technical report