import numpy as np

from devices.endurance import degraded_biolek_defaults
from devices.FracMemCap import BIOLEK_DEFAULTS
from sim.crossbar.builders import build_crossbar
from sim.nn.layer import CrossbarLayer
from sim.nn.mapper import CrossbarProgrammer
from sim.training.toydataset import X

dt = 1e-4

W = np.load("sim/training/trained_weights.npy")  # single-layer weights, (64,4)
in_features, out_features = W.shape

CV_FIXED = 0.10
R_FIXED = 0.0
DEVICE_TYPES = ["team", "biolek", "team", "biolek"]
N_SAMPLES = 50  # enough to average out per-sample noise, cheap to run


def build_and_run(N_cyc, seed):
    cmin_deg, cmax_deg, cinit_deg = degraded_biolek_defaults(
        N_cyc,
        BIOLEK_DEFAULTS["Cmin"],
        BIOLEK_DEFAULTS["Cmax"],
        BIOLEK_DEFAULTS["Cinit"],
    )
    biolek_params = {"Cmin": cmin_deg, "Cmax": cmax_deg, "Cinit": cinit_deg}

    cb = build_crossbar(
        in_features, out_features,
        R_row=R_FIXED, R_col=R_FIXED,
        device_types=DEVICE_TYPES,
        variability_cv=CV_FIXED,
        seed=seed,
        biolek_params=biolek_params,
    )

    programmer = CrossbarProgrammer()
    layer = CrossbarLayer(cb)
    programmer.map_weights(cb, W)

    # collect raw per-column current BEFORE differential recombination,
    # so we can compare TEAM columns vs Biolek columns directly
    raw_outputs = []
    for i in range(N_SAMPLES):
        programmer.map_weights(cb, W)  # per-sample reprogram, read-disturb safety
        cb.apply_row_inputs(X[i])
        currents = cb.forward_step(dt)  # raw per-column currents, pre-differential
        raw_outputs.append(currents)

    return np.array(raw_outputs)  # (N_SAMPLES, cols)


for N_cyc, seed in [(0, 100), (1e10, 100)]:
    currents = build_and_run(N_cyc, seed)
    # column layout: [team+, biolek+, team+, biolek+] pairs per DEVICE_TYPES,
    # each doubled for differential encoding -> 8 physical columns for out_features=4
    col_types = []
    for t in DEVICE_TYPES:
        col_types += [t, t]  # each logical output has 2 physical columns (+/-)

    team_cols = [c for c in range(currents.shape[1]) if col_types[c] == "team"]
    biolek_cols = [c for c in range(currents.shape[1]) if col_types[c] == "biolek"]

    team_mag = np.abs(currents[:, team_cols]).mean()
    biolek_mag = np.abs(currents[:, biolek_cols]).mean()

    print(f"N={N_cyc:.0e}: mean |TEAM column current| = {team_mag:.6e}, "
          f"mean |Biolek column current| = {biolek_mag:.6e}, "
          f"ratio biolek/team = {biolek_mag/team_mag:.4f}")