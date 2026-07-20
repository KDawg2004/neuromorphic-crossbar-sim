import numpy as np
from devices.TeamMemristor import TEAMMemristor
from devices.FracMemCap import BiolekMemcapacitor
from sim.crossbar.crossbar import Crossbar

# Placeholder coefficient of variation. NOT sourced from a confirmed citation.
# Log-normal shape is literature-justified (RRAM conductance is strictly positive,
# device-to-device spread commonly modeled log-normal). Magnitude is a guess pending
# real numbers from advisors or the originally-intended source.
DEFAULT_VARIABILITY_CV = 0.15


def _make_device(kind, rng, variability_cv, biolek_params=None):
    if kind == "team":
        return TEAMMemristor(variability_cv=variability_cv, rng=rng)
    elif kind == "biolek":
        params = biolek_params or {}
        return BiolekMemcapacitor(variability_cv=variability_cv, rng=rng, **params)
    else:
        raise ValueError(f"unknown device kind: {kind}")

def build_crossbar(in_features, out_features, R_row=0.0, R_col=0.0,
                    device_types=None, variability_cv=0.0, seed=None, biolek_params=None):
    """
    Build a crossbar with differential pairs, optionally mixed device types
    and device-to-device variability.

    device_types : list[str] of length out_features, values in {"team", "biolek"}.
        Defaults to all TEAM if not given.
    variability_cv : coefficient of variation applied to G_on/G_off (TEAM) and
        Cmin/Cmax (Biolek) independently, drawn once per device at construction.
        0.0 (default) reproduces fully deterministic nominal-parameter behavior.
    seed : int or None. If given, used to build a reproducible np.random.Generator
        for all devices in this crossbar. If None, variability draws are non-reproducible.
    """
    if device_types is None:
        device_types = ["team"] * out_features

    if len(device_types) != out_features:
        raise ValueError(f"device_types length {len(device_types)} != out_features {out_features}")

    rng = np.random.default_rng(seed)

    cb = Crossbar(rows=in_features, cols=out_features * 2, R_row=R_row, R_col=R_col)

    for row in range(cb.rows):
        for out_col in range(out_features):
            kind = device_types[out_col]
            cb.set_device(row, out_col * 2, _make_device(kind, rng, variability_cv, biolek_params))
            cb.set_device(row, out_col * 2 + 1, _make_device(kind, rng, variability_cv, biolek_params))

    return cb