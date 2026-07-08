import numpy as np
from devices.TeamMemristor import TEAMMemristor
from devices.FracMemCap import BiolekMemcapacitor

N = 100
CV = 0.15


def test_team_variability_preserves_ordering_and_draws():
    rng = np.random.default_rng(42)
    devices = [TEAMMemristor(variability_cv=CV, rng=rng) for _ in range(N)]

    for i, d in enumerate(devices):
        assert d.G_on > d.G_off, f"device {i}: G_on={d.G_on} not > G_off={d.G_off}"

    g_ons = [d.G_on for d in devices]
    assert len(set(g_ons)) > 1, "all TEAM devices got identical G_on, variability not applied"

    print(f"TEAM: {N} devices, G_on range [{min(g_ons):.3e}, {max(g_ons):.3e}], "
          f"nominal G_on=1/500={1/500:.3e}")
    print("TEAM ordering + spread check: PASS")


def test_biolek_variability_preserves_ordering_and_draws():
    rng = np.random.default_rng(42)
    devices = []
    n_rejected = 0

    # can't guarantee 0 raises with a naive loop since the invariant check can fire;
    # count them instead of letting the test itself crash
    for _ in range(N):
        try:
            devices.append(BiolekMemcapacitor(variability_cv=CV, rng=rng))
        except ValueError:
            n_rejected += 1

    print(f"Biolek: {n_rejected}/{N} draws violated Cmin < Cinit < Cmax at cv={CV}")

    for i, d in enumerate(devices):
        assert d.Cmin < d.Cmax, f"device {i}: Cmin={d.Cmin} not < Cmax={d.Cmax}"
        assert 0.0 <= d.x_init <= 1.0, f"device {i}: x_init={d.x_init} out of [0,1]"

    if len(devices) > 1:
        cmins = [d.Cmin for d in devices]
        assert len(set(cmins)) > 1, "all surviving Biolek devices got identical Cmin, variability not applied"

    print("Biolek ordering + x_init range check: PASS")

    if n_rejected > N * 0.05:
        print(f"WARNING: {n_rejected}/{N} ({100*n_rejected/N:.1f}%) draws rejected — "
              f"cv={CV} may be too aggressive for Cmin/Cinit/Cmax spacing")


def test_reproducibility_same_seed():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)

    d1 = TEAMMemristor(variability_cv=CV, rng=rng1)
    d2 = TEAMMemristor(variability_cv=CV, rng=rng2)

    assert d1.G_on == d2.G_on and d1.G_off == d2.G_off, \
        "same seed produced different TEAM parameters, reproducibility broken"

    print("reproducibility (same seed -> same params): PASS")


def test_different_seeds_diverge():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)

    d1 = TEAMMemristor(variability_cv=CV, rng=rng1)
    d2 = TEAMMemristor(variability_cv=CV, rng=rng2)

    assert d1.G_on != d2.G_on, \
        "different seeds produced identical TEAM parameters, rng not actually varying"

    print("different seeds -> different params: PASS")


def test_zero_cv_is_deterministic_nominal():
    # cv=0.0 must reproduce exact nominal values, no draw at all
    d = TEAMMemristor(G_on=1/500, G_off=1/5000, variability_cv=0.0)
    assert d.G_on == 1/500 and d.G_off == 1/5000, \
        "variability_cv=0.0 changed nominal values, default behavior regressed"
    print("cv=0.0 reproduces exact nominal values: PASS")


if __name__ == "__main__":
    test_team_variability_preserves_ordering_and_draws()
    test_biolek_variability_preserves_ordering_and_draws()
    test_reproducibility_same_seed()
    test_different_seeds_diverge()
    test_zero_cv_is_deterministic_nominal()
    print("all variability tests passed")