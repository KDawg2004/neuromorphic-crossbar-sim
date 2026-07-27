# neuromorphic-crossbar-sim

A research-oriented simulation framework for studying memristive and memcapacitive crossbar arrays in neuromorphic inference settings. This project is currently in an active development and validation stage: the core simulator, mixed-device crossbar support, and several analysis sweeps are implemented, but some device/circuit consistency checks are still being resolved.

The codebase combines device-level physics, circuit-style crossbar solving, and simple neural-network inference experiments. It is intended as a research prototype rather than a finished release package.

---

## What the project does

- Implements TEAM memristor and Biolek memcapacitor device models
- Builds heterogeneous crossbars with mixed device types
- Supports ideal and parasitic-resistance solve paths
- Maps trained network weights into device states for crossbar-style inference
- Includes scripts for variability, parasitic-resistance, sneak-path, and endurance-style analysis

---

## Current status

### Implemented

- TEAM memristor model with state evolution and conductance programming
- Biolek memcapacitor model with state update and crossbar-compatible interface
- Mixed memristor/memcapacitor crossbar construction
- Ideal and MNA-style solve paths
- Row/column parasitic-resistance support
- Weight-programming and inference workflow for simple neural-network experiments
- Variability and reproducibility scaffolding
- Several example scripts for demonstrations and diagnostics

### Active work

- Validating the Biolek current formulation against the crossbar companion-model path
- Comparing coupled and decoupled crossbar reads for sneak-path analysis
- Refining the interpretation of sweep results and reproducibility behavior
- Improving examples and documentation as the simulator evolves

---

## Repository layout

### devices/

Contains the device physics layer and shared interfaces.

- device.py
- memristive.py
- memcapacitive.py
- TeamMemristor.py
- FracMemCap.py

These define the common device API used by the crossbar solver.

### sim/

Contains the simulation framework.

- sim/crossbar/: crossbar topology and solver logic
- sim/nn/: simple neural-network abstraction and weight mapping
- sim/training/: toy dataset and inference-related utilities
- sim/examples/: demonstration and analysis scripts

### Sandbox/

Older standalone implementations retained for reference.

### testing/

Development and validation scripts used during the project’s iterative debugging process.

---

## Example scripts

Run a basic crossbar demo:

```bash
python -m sim.examples.crossDemo
```

Run a simple inference example:

```bash
python -m sim.examples.networkDemo
```

Run a parasitic-resistance sweep:

```bash
python -m sim.examples.parasiticSweep
```

Run the current sneak-path comparison workflow:

```bash
python -m sim.examples.sneakCurrent
```

Run a variability sweep:

```bash
python -m sim.examples.varabilitySweep
```

---

## Notes

This repository is best viewed as a research prototype and experimental playground. Some of the current examples and sweep results are useful for exploring behavior and identifying modeling issues, but they should be treated as work-in-progress rather than final, publication-ready conclusions.

The immediate focus is on tightening the consistency between device-level current laws and the crossbar solver, then using that foundation for more reliable comparisons and sweeps.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.