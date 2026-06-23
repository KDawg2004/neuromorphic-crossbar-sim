# Fractional Memcapacitor

A reusable Python implementation of the **Biolek charge-controlled memcapacitor** with support for a **Caputo fractional-order state equation**.

The project began as a direct translation of the original SPICE model into Python and has been refactored into a reusable package for simulation, validation, and experimentation with fractional order memdevices.

## Features

* Biolek charge controlled memcapacitor model
* Integer-order and fractional-order state dynamics
* Caputo fractional derivative implementation
* Charge-driven excitation with bipolar reference signals
* Validation utilities for:

  * α = 1 equivalence with the ordinary model
  * Fractional-order (α) hysteresis sweeps
  * Figure-eight hysteresis area measurements
* Modular structure suitable for research and future extensions

## Package Structure

```
fractional_memcapacitor/

├── __init__.py
├── model.py
├── validation.py
├── plotting.py
├── examples/
│   ├── __init__.py
│   └── demo.py
└── README.md
```

### `model.py`

Contains the `BiolekMemcapacitor` class and the core simulation logic, including:

* Capacitance model
* Window function
* Voltage calculation
* Charge reference generation
* Fractional state solver
* Time-domain simulation

### `validation.py`

Contains utilities used to verify model behavior, including:

* α = 1 validation against the ordinary model
* Fractional-order parameter sweeps
* Figure eight hysteresis loop area calculations

### `plotting.py`

Contains helper functions for visualizing simulation results and hysteresis loops.

### `examples/demo.py`

Demonstrates typical usage and reproduces the validation workflow previously implemented in the original standalone script.

## Example

```python
from fractional_memcapacitor import BiolekMemcapacitor

model = BiolekMemcapacitor(
    Cmin=50e-9,
    Cmax=200e-9,
    Cinit=100e-9,
    k=1e7,
    p=10,
)

t, q, x, v, i = model.simulate(
    t_end=5.0,
    freq=1.0,
    Q_amp=100e-9,
    alpha=0.8,
)
```

## Validation

The current implementation has been validated by:

* Recovering the ordinary Biolek model for α = 1
* Performing fractional-order α sweeps
* Measuring figure-eight hysteresis loop area across fractional orders
* Observing expected hysteresis collapse with increasing excitation frequency

These validation routines are intended to ensure that future refactoring or feature additions preserve the numerical behavior of the model.

## Project Goals

Future development aims to include:

* Frequency sweep validation utilities
* Additional memcapacitor window functions
* Alternative fractional numerical solvers
* Additional fractional memdevice models
* Expanded documentation and examples

## Background

This package is intended as a research and educational tool for exploring fractional-order memcapacitive systems while maintaining a clear separation between:

* model implementation,
* validation experiments,
* visualization utilities, and
* example applications.
