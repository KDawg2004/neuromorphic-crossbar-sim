import numpy as np
from devices.TeamMemristor import TEAMMemristor
from devices.FracMemCap import BiolekMemcapacitor
from sim.crossbar.builders import build_crossbar
from sim.nn.mapper import CrossbarProgrammer

IN_FEATURES = 4
OUT_FEATURES = 4


def test_mixed_map_weights_programs_correctly():
    device_types = ["team", "biolek", "team", "biolek"]
    cb = build_crossbar(IN_FEATURES, OUT_FEATURES, R_row=0.0, R_col=0.0, device_types=device_types)
    programmer = CrossbarProgrammer()

    weights = np.array([
        [ 1.0, -1.0,  0.5, -0.5],
        [ 0.3,  0.0, -0.8,  0.2],
        [-0.6,  0.9,  0.0,  1.0],
        [ 0.0, -0.3,  0.4, -1.0],
    ])

    programmer.map_weights(cb, weights)

    w_max = np.abs(weights).max()

    for row in range(IN_FEATURES):
        for out_col in range(OUT_FEATURES):
            w_norm = weights[row, out_col] / w_max
            mag = abs(w_norm)
            expected_plus = mag if w_norm >= 0 else 0.0
            expected_minus = 0.0 if w_norm >= 0 else mag

            plus = cb.get_device(row, out_col * 2)
            minus = cb.get_device(row, out_col * 2 + 1)

            kind = device_types[out_col]
            exp_cls = TEAMMemristor if kind == "team" else BiolekMemcapacitor
            assert isinstance(plus, exp_cls) and isinstance(minus, exp_cls)

            # check programmed state landed correctly regardless of device type
            plus_state = plus.w_init if kind == "team" else (1.0 - plus.x_init)
            minus_state = minus.w_init if kind == "team" else (1.0 - minus.x_init)

            assert abs(plus_state - expected_plus) < 1e-9, \
                f"row {row} out_col {out_col} (+, {kind}): expected {expected_plus}, got {plus_state}"
            assert abs(minus_state - expected_minus) < 1e-9, \
                f"row {row} out_col {out_col} (-, {kind}): expected {expected_minus}, got {minus_state}"

    print("mixed map_weights programming: PASS")


def test_no_cross_contamination_between_types():
    cb = build_crossbar(IN_FEATURES, OUT_FEATURES, R_row=0.0, R_col=0.0,
                         device_types=["team", "biolek", "team", "biolek"])
    programmer = CrossbarProgrammer()

    w1 = np.full((IN_FEATURES, OUT_FEATURES), 0.5)
    programmer.map_weights(cb, w1)
    team_states_before = [cb.get_device(r, 0).w_init for r in range(IN_FEATURES)]

    # keep w_max identical so normalization doesn't change; only change biolek column's sign/value
    # while preserving the matrix max magnitude at 0.5
    w2 = w1.copy()
    w2[:, 1] = -0.5  # biolek out_col, same magnitude, different sign, w_max unchanged
    programmer.map_weights(cb, w2)
    team_states_after = [cb.get_device(r, 0).w_init for r in range(IN_FEATURES)]

    assert team_states_before == team_states_after, \
        "TEAM device state changed after reprogramming unrelated Biolek column, with w_max held constant"

    print("no cross-contamination between device types: PASS")


if __name__ == "__main__":
    test_mixed_map_weights_programs_correctly()
    test_no_cross_contamination_between_types()
    print("all mapper tests passed")