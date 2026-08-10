# neuromorphic-crossbar-sim

A research simulation framework for studying mixed memristive/memcapacitive crossbar arrays in neuromorphic inference settings. Built for Project 18, DISC Lab, Portland State University (Summer Proceedings 2026). The simulator models device-level non-idealities in a mixed TEAM memristor / Biolek memcapacitor crossbar and measures how they propagate to network-level inference accuracy.

This project moved through an extended validation and debugging phase before landing on results we trust. That process is part of the story, not just the destination, some of it is documented directly in code comments and in the sweep scripts themselves, since the bugs found along the way (device programming conventions, read-disturb, a dead current-calculation path, discretization vs. physical state mismatches) shaped what the final results actually mean.

---

## What the project does

- Implements TEAM memristor and Biolek memcapacitor device models, validated against known switching/hysteresis behavior
- Builds heterogeneous crossbars with mixed device types via a modified nodal analysis (MNA) solver
- Maps digitally-trained neural network weights into device states for crossbar-style inference
- Measures how four device-level non-idealities affect classification accuracy on an MNIST-derived task:
  - **Device-to-device variability** (log-normal spread on device parameters)
  - **Endurance degradation** (cycling-induced window closure, sourced from published HZO-MS ferroelectric memcapacitor data)
  - **Parasitic row/column wire resistance**
  - **Sneak-path currents**, isolated from series resistance loss via a purpose-built decoupled solver

---

## Key findings

- **Sneak-path currents, not series resistance, dominate parasitic-resistance-driven accuracy loss.** A decoupled solver (array coupling terms removed, series resistance kept) shows only mild accuracy loss from resistance alone. The real, coupled solver collapses far more sharply over the same resistance range. The gap between the two is attributable specifically to sneak-path coupling.
- **Endurance degradation has a non-monotonic effect on accuracy in this mixed architecture.** An early dip is followed by a statistically significant net accuracy *improvement* at high cycle counts, driven by the degrading Biolek devices' shrinking relative contribution to the differential readout, effectively increasing the more stable TEAM devices' influence on the network's decisions. This is very likely specific to how this network's weights were trained (in an idealized digital domain, with no awareness of Biolek's nonlinear response) rather than a general property of endurance degradation.
- **Device-to-device variability has a comparatively mild effect** on accuracy across the tested range, small next to the other two mechanisms.
- **Correctly mapping a trained weight onto a Biolek memcapacitor's device state is not a trivial choice.** Several device-programming conventions were tried and rejected during this project (state linear in the internal state variable, state linear in a discretized solver-specific conductance) before arriving at the physically correct approach: state linear in capacitance, the quantity a real write-verify circuit would actually target. This mapping choice measurably affects results and is documented in `devices/FracMemCap.py`.

---

## Known limitations / future work

- **Per-layer sensitivity analysis is not completed.** Extending the network to multiple crossbar layers requires resolving inter-layer signal calibration (a real crossbar's physical current output must be brought back into a usable input range for the next stage, analogous to a programmable gain amplifier). An attempted single- and dual-scale calibration approach surfaced a deeper per-device-type nonlinearity issue rather than resolving it cleanly, and the two-layer sensitivity sweep was not completed on solid ground before the project's timeline closed.
- **TEAM memristor endurance is not modeled.** Only Biolek's endurance behavior, sourced from a real ferroelectric memcapacitor endurance curve, is characterized. No equivalent source was found for TEAM within the project timeline.
- **The device-to-device variability magnitude is a literature-motivated placeholder, not pinned to a specific sourced number.**
- **Weights were trained in an idealized digital domain**, with no awareness of device-specific nonlinearities. The accuracy gap between digital and simulated crossbar inference likely reflects this train/deploy mismatch as well as genuine device-mixing cost. Hardware-aware training, through a differentiable device-response surrogate or a fully differentiable crossbar solver, is the natural next step to separate the two.
- **No train/test split.** Reported accuracy is evaluated on the same 800 samples used for training. Low overfitting risk given the small, unregularized linear/two-layer models used, but worth a real held-out split if this work continues.

---

## Repository layout

### devices/

Device physics layer and shared interfaces.

- `device.py`, `memristive.py`, `memcapacitive.py` — abstract base classes
- `TeamMemristor.py` — TEAM memristor model, canonical parameters in `TEAM_DEFAULTS`
- `FracMemCap.py` — Biolek memcapacitor model, canonical parameters in `BIOLEK_DEFAULTS`. See `program()`'s docstring for the device-programming mapping rationale.
- `endurance.py` — cycle-count-dependent Biolek degradation model, digitized from published HZO-MS endurance data

### sim/

The simulation framework.

- `sim/crossbar/` — crossbar topology, sparse MNA solver, and the decoupled (no-sneak-path) solver variant
- `sim/nn/` — crossbar-backed neural network layer abstraction and weight-to-device mapping
- `sim/training/` — dataset (MNIST subset, 8x8 downsampled, 4 classes) and model training scripts
- `sim/examples/` — sweep scripts (variability, endurance, parasitic resistance, sneak-path comparison) and diagnostic scripts used during validation
- `sim/plotting/` — result plotting scripts

### Sandbox/

Older standalone implementations retained for reference.

### testing/

Development and validation scripts used during the project's iterative debugging process.

---

## Running the sweeps

Variability sweep:
```bash
python -m sim.examples.varabilitySweep
```

Endurance sweep:
```bash
python -m sim.examples.enduranceSweep
```

Combined parasitic-resistance / variability grid:
```bash
python -m sim.examples.parasiticVariabilitySweep
```

Sneak-path comparison (coupled vs. decoupled solver):
```bash
python -m sim.examples.sneakComparison
```

All sweep scripts save results incrementally per-cell to `sim/training/*.json` and resume automatically if interrupted. Corresponding plotting scripts live in `sim/plotting/`.

---

## Notes

Results in this repository reflect the corrected device-programming mapping (capacitance-linear in state) finalized late in the project. Earlier sweep results, run under prior mapping conventions, were superseded and rerun; see commit history and in-code comments for the reasoning behind each correction.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.