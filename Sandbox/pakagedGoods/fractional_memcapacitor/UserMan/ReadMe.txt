# Biolek Memcapacitor

A Python model of a fractional-order memcapacitor with a configurable window function and hysteresis behavior.

## Overview

This package implements a Biolek-style memcapacitor model with:
- fractional-order state evolution,
- selectable window functions,
- charge-driven simulation,
- capacitance, voltage, and hysteresis analysis.

It is intended for research, simulation, and validation of memcapacitive devices.

## Features

- Fractional-order state update.
- Joglekar or Biolek window function.
- Charge-driven excitation.
- Capacitance and voltage output.
- Plotting utilities for q-v loops and state evolution.

## Requirements

- Python 3
- NumPy
- SciPy
- Matplotlib

## Installation

```bash
pip install numpy scipy matplotlib
```

## Basic Usage

```python
from BiolekMemcapacitor import BiolekMemcapacitor
from plotting import plot_model

model = BiolekMemcapacitor(
    Cmin=10e-9,
    Cmax=10e-6,
    Cinit=100e-9,
    k=1e7,
    p=1,
    IC=0.0,
    window_type="biolek"
)

t, q, x, v, i = plot_model(
    model,
    t_end=5.0,
    freq=1.0,
    Q_amp=100e-9,
    alpha=0.5,
    n_points=1000
)
```

## Parameters

- `Cmin`: minimum capacitance.
- `Cmax`: maximum capacitance.
- `Cinit`: initial capacitance.
- `k`: state evolution rate.
- `p`: window sharpness.
- `IC`: initial charge offset.
- `window_type`: `biolek` or `joglekar`.

## Simulation Output

`simulate()` returns:
- `t`: time array
- `q`: charge array
- `x`: internal state variable
- `v`: terminal voltage
- `i`: drive current

## Notes

- `x` is clipped to the range `[0, 1]`.
- `alpha=1` reduces to the ordinary ODE implementation.
- Lower `alpha` values produce fractional-order dynamics.
- The default setup is designed to produce pinched hysteresis in the `q-v` plane.

## Validation

The package includes helper routines for:
- sweeping `alpha`,
- checking `alpha=1` behavior,
- computing hysteresis loop area.

## Author
Kaevin Barta