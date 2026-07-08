# neuromorphic-crossbar-sim

A research-oriented simulation framework for studying neuromorphic crossbar arrays built from memristor and memcapacitor devices. Developed for undergraduate research at the Portland State University DISC Lab (Project 18, Summer Proceedings 2026), the framework investigates how device-level non-idealities—including parasitic wire resistance, device-to-device variability, and mixed-device architectures—affect neural network inference.

The simulator separates device physics from circuit simulation through a common device interface, allowing heterogeneous crossbars containing arbitrary combinations of memristive and memcapacitive devices.

---

# Features

- Device-agnostic crossbar architecture
- TEAM threshold memristor implementation
- Biolek memcapacitor implementation (integer and fractional order)
- Modified Nodal Analysis (MNA) solver for row/column parasitic resistance
- Mixed memristor/memcapacitor arrays
- Weight mapping from neural networks to physical device states
- Monte Carlo device variability
- Reproducible random seeds
- PyTorch integration for training and inference experiments
- Validation utilities and comparison examples

---

# Repository Structure

## devices/

Reusable device physics layer.

### Core interfaces

- `device.py`
- `memristive.py`
- `memcapacitive.py`

Defines the abstract interfaces shared by all device models.

Every device exposes a common API used by the crossbar solver:

- `network_current()`
- `network_step()`
- `current_conductance()`
- `current_offset()`
- `program()`

This allows the circuit solver to remain completely independent of device physics.

### Device models

#### `TeamMemristor.py`

Implementation of the TEAM threshold adaptive memristor model.

Features:

- Threshold switching
- Continuous state evolution
- Conductance programming
- Crossbar-compatible linearization
- Variability support
- MNA validated

#### `FracMemCap.py`

Implementation of the Biolek memcapacitor model.

Features:

- Integer-order companion model
- Fractional-order implementation
- Internal state dynamics
- Charge-based behavior
- Crossbar-compatible interface

---

## sim/

Simulation framework.

### crossbar/

#### `crossbar.py`

Implements an arbitrary M×N crossbar supporting

- ideal arrays
- row parasitic resistance
- column parasitic resistance
- full MNA solution
- heterogeneous device arrays

The solver never depends on a specific device type.

#### `builders.py`

Crossbar construction utilities.

Current functionality includes

- automatic device construction
- mixed-device arrays
- parameter presets
- reproducible RNG/seed handling
- device variability injection

---

### nn/

Simple neural network abstraction.

Includes

- fully connected layer
- network container
- weight mapper

The mapper converts trained floating-point weights into physical device states without requiring device-specific code.

---

### training/

PyTorch training utilities.

Includes

- toy dataset
- simple training pipeline
- inference preparation

---

### plotting/

Visualization utilities.

Includes plotting for

- TEAM I-V hysteresis
- Biolek Q-V hysteresis
- device validation
- parameter sweeps

---

### examples/

Example simulations demonstrating framework capabilities.

Current examples include

- `crossDemo.py`

  General crossbar demonstration.

- `CrossBarAccuracy.py`

  Verifies ideal and MNA paths produce identical solutions when parasitic resistance is zero.

- `networkDemo.py`

  Neural network inference through the simulated crossbar.

- `mixedBuildTest.py`

  Mixed memristor/memcapacitor construction.

- `mixedInferenceSmoke.py`

  Smoke test validating heterogeneous inference.

- `mixedMapTest.py`

  Device-independent weight mapping.

- `parasiticSweep.py`

  Effect of increasing wire resistance.

- `stateManagement.py`

  Device programming and state consistency.

- `variabilitySweep.py`

  Monte Carlo variability experiments.

- `variabilityTest.py`

  Validation of variability implementation, ordering, and reproducibility.

---

## Sandbox/

Historical implementations developed before the current framework architecture.

Contains

- standalone TEAM package
- standalone fractional memcapacitor package
- validation utilities
- prototype implementations

Retained for reference only.

---

## testing/

Validation scripts used during development.

Includes

- single-device verification
- plotting utilities
- comparison against standalone implementations

---

# Current Status

## Device Models

### TEAM Memristor

✔ Validated against published I-V hysteresis

✔ Threshold switching verified

✔ Crossbar compatible

✔ MNA validated

---

### Biolek Memcapacitor

✔ Integer-order implementation validated

✔ Fractional-order implementation complete

✔ Crossbar compatible

⚠ Fractional-order MNA validation still in progress

---

## Crossbar

✔ Device-agnostic architecture

✔ Mixed-device arrays

✔ Full MNA solver

✔ Ideal solver

✔ Row and column parasitic resistance

✔ Shared device interface

---

## Neural Network Support

✔ Weight mapping

✔ Device-independent programming

✔ Mixed-device inference

✔ Smoke-tested inference pipeline

---

## Variability

Implemented:

- reproducible random seeds
- device-to-device variability
- log-normal resistance distributions
- configurable coefficient of variation
- deterministic Monte Carlo experiments

Current variability parameters use estimated CV values pending publication-quality statistical data from experimental RRAM literature.

---

# Validation Summary

Verified:

- TEAM standalone vs crossbar
- Biolek standalone vs crossbar
- Ideal solver vs MNA
- Zero-parasitic equivalence
- Mixed-device construction
- Device-independent mapping
- Programming consistency
- Variability ordering
- RNG reproducibility
- Non-degenerate inference outputs

---

# Typical Usage

Run a crossbar demonstration

```bash
python -m sim.examples.crossDemo
```

Run mixed-device inference

```bash
python -m sim.examples.networkDemo
```

Run variability sweep

```bash
python -m sim.examples.variabilitySweep
```

Run parasitic resistance sweep

```bash
python -m sim.examples.parasiticSweep
```

---

# Research Roadmap

Completed

- TEAM memristor implementation
- Biolek memcapacitor implementation
- Mixed-device architecture
- Device-independent crossbar
- Full MNA solver
- Neural-network weight mapping
- Device variability
- Monte Carlo framework

In Progress

- Training larger neural networks
- Experimental variability calibration
- Fractional-order MNA validation

Planned

- Reservoir computing experiments
- Additional compact device models
- Larger benchmark datasets
- Performance optimization
- Summer Proceedings technical report