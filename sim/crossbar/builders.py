from devices.TeamMemristor import TEAMMemristor
from devices.FracMemCap import BiolekMemcapacitor
from sim.crossbar.crossbar import Crossbar


TEAM_DEFAULTS = dict(
    k_off=1.333,
    k_on=-1.333,
    alpha_off=2,
    alpha_on=2,
    i_off=0.5e-3,
    i_on=-0.5e-3,
    G_on=1/500,
    G_off=1/5000
)


def _make_device(kind):
    if kind == "team":
        return TEAMMemristor(**TEAM_DEFAULTS)
    elif kind == "biolek":
        return BiolekMemcapacitor()
    else:
        raise ValueError(f"unknown device type: {kind!r}, expected 'team' or 'biolek'")


def build_crossbar(in_features, out_features, R_row=0.0, R_col=0.0, device_types=None):
    """
    Build a crossbar with differential pairs, optionally mixed device types.

    device_types : list[str] of length out_features, values in {"team", "biolek"}.
        device_types[i] sets the type for output column i's differential pair.
        Defaults to all TEAM if not given.
    """
    if device_types is None:
        device_types = ["team"] * out_features

    if len(device_types) != out_features:
        raise ValueError(f"device_types length {len(device_types)} != out_features {out_features}")

    cb = Crossbar(rows=in_features, cols=out_features * 2, R_row=R_row, R_col=R_col)

    for row in range(cb.rows):
        for out_col in range(out_features):
            kind = device_types[out_col]
            cb.set_device(row, out_col * 2, _make_device(kind))
            cb.set_device(row, out_col * 2 + 1, _make_device(kind))

    return cb