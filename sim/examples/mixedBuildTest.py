from devices.TeamMemristor import TEAMMemristor
from devices.FracMemCap import BiolekMemcapacitor
from sim.crossbar.builders import build_crossbar

IN_FEATURES = 4
OUT_FEATURES = 4

def test_all_team_default():
    cb = build_crossbar(IN_FEATURES, OUT_FEATURES, R_row=0.0, R_col=0.0)
    for row in range(cb.rows):
        for col in range(cb.cols):
            assert isinstance(cb.get_device(row, col), TEAMMemristor), \
                f"expected TEAM at ({row},{col}), got {type(cb.get_device(row, col))}"
    print("all-TEAM default: PASS")


def test_mixed_pairs_match_and_alternate():
    device_types = ["team", "biolek", "team", "biolek"]
    cb = build_crossbar(IN_FEATURES, OUT_FEATURES, R_row=0.0, R_col=0.0, device_types=device_types)

    expected_class = {"team": TEAMMemristor, "biolek": BiolekMemcapacitor}

    for row in range(cb.rows):
        for out_col in range(OUT_FEATURES):
            kind = device_types[out_col]
            exp_cls = expected_class[kind]

            plus = cb.get_device(row, out_col * 2)
            minus = cb.get_device(row, out_col * 2 + 1)

            assert isinstance(plus, exp_cls), \
                f"row {row} out_col {out_col} (+): expected {exp_cls.__name__}, got {type(plus).__name__}"
            assert isinstance(minus, exp_cls), \
                f"row {row} out_col {out_col} (-): expected {exp_cls.__name__}, got {type(minus).__name__}"
            assert type(plus) == type(minus), \
                f"row {row} out_col {out_col}: differential pair type mismatch, + is {type(plus).__name__}, - is {type(minus).__name__}"

    print("mixed pairs match and alternate: PASS")


def test_device_identity_not_shared():
    # every crosspoint must be its own object, not a shared reference
    cb = build_crossbar(IN_FEATURES, OUT_FEATURES, R_row=0.0, R_col=0.0,
                         device_types=["biolek"] * OUT_FEATURES)
    seen = set()
    for row in range(cb.rows):
        for col in range(cb.cols):
            dev = cb.get_device(row, col)
            assert id(dev) not in seen, f"device at ({row},{col}) shares identity with another crosspoint"
            seen.add(id(dev))
    print("device identity uniqueness: PASS")


def test_bad_device_types_length_raises():
    try:
        build_crossbar(IN_FEATURES, OUT_FEATURES, device_types=["team"] * (OUT_FEATURES - 1))
        raise AssertionError("expected ValueError for mismatched device_types length")
    except ValueError:
        print("bad length raises ValueError: PASS")


def test_unknown_device_type_raises():
    try:
        build_crossbar(IN_FEATURES, OUT_FEATURES, device_types=["quantum"] * OUT_FEATURES)
        raise AssertionError("expected ValueError for unknown device type")
    except ValueError:
        print("unknown type raises ValueError: PASS")


if __name__ == "__main__":
    test_all_team_default()
    test_mixed_pairs_match_and_alternate()
    test_device_identity_not_shared()
    test_bad_device_types_length_raises()
    test_unknown_device_type_raises()
    print("all builder tests passed")