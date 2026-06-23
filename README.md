# neuromorphic-crossbar-sim

A research-oriented simulation workspace for studying neuromorphic crossbar behavior using memristor and memcapacitor devices. The project combines circuit-level simulation with Python-based modeling to explore how device non-idealities influence behavior in compact crossbar structures.

## Project overview

This repository contains:
- SPICE/Xyce netlists for memristor and memcapacitor crossbar experiments
- Python implementations of fractional-order memcapacitor models
- Validation and plotting utilities for analyzing hysteresis and device dynamics
- Example scripts that reproduce the main simulation workflow

## Repository contents

- Circuit examples under the testing directory for single-device and crossbar simulations
- Python modules such as BiolekMemcap.py, fractionMemCap.py, TeamMemristor.py, and the packaged fractional memcapacitor implementation under fractional_memcapacitor/
- Supporting scripts for plotting and validating results

## Typical workflow

### Circuit simulation with Xyce
- Create or edit a netlist: `vim circuit.cir` or `code circuit.cir`
- Run the simulation: `Xyce circuit.cir`
- Inspect the output: `cat circuit.prn`

### Python-based modeling
- Run the fractional memcapacitor demo from the package directory:
  `python -m fractional_memcapacitor.examples.demo`

## Goals

The long term goal is to provide a reproducible framework for investigating how fractional-order dynamics, device variability, and crossbar topology affect emergent behavior in neuromorphic hardware-inspired systems.

