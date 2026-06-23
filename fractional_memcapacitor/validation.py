import numpy as np
import matplotlib.pyplot as plt


def polygon_area(x, y):

    return 0.5 * abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )


def polygon_area(x, y):
    """
    Shoelace formula for a simple closed polygon.
    """
    return 0.5 * abs(
        np.dot(x, np.roll(y, -1))
        - np.dot(y, np.roll(x, -1))
    )


def figure8_area(v, q):
    """
    Compute the total area of a pinched figure-eight hysteresis loop
    by splitting it into positive-q and negative-q lobes.
    """

    # Upper lobe
    upper = q >= 0

    # Lower lobe
    lower = q <= 0

    area_upper = polygon_area(v[upper], q[upper])
    area_lower = polygon_area(v[lower], q[lower])

    return area_upper + area_lower


def validate_alpha_sweep(model):

    alphas = [1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    loop_areas = []

    # ----------------------------------
    # q-v hysteresis plot
    # ----------------------------------
    plt.figure(figsize=(7,7))

    for alpha in alphas:

        t, q, x, v, i = model.simulate(
            t_end=5.0,
            freq=1.0,
            Q_amp=100e-9,
            alpha=alpha
        )

        # Closed-loop area using the line integral ∮q dV
        area = figure8_area(v, q)

        loop_areas.append(area)

        print(f"alpha = {alpha:.2f}   loop area = {area:.4e}")

        plt.plot(v, q*1e9, label=f"α={alpha}")

    plt.xlabel("Voltage (V)")
    plt.ylabel("Charge (nC)")
    plt.title("Fractional Memcapacitor α Sweep")
    plt.grid(True)
    plt.legend()

    plt.show()

    # ----------------------------------
    # Loop area vs alpha
    # ----------------------------------
    plt.figure(figsize=(6,4))

    plt.plot(alphas, loop_areas, "o-", linewidth=2, markersize=6)

    plt.xlabel("Fractional Order α")
    plt.ylabel("Loop Area |∮q dV|")
    plt.title("Hysteresis Area vs Fractional Order")
    plt.grid(True)

    plt.show()


def validate_alpha_one(model):

    print("\n============================")
    print("Validating alpha = 1.0 (ordinary ODE)")
    print("============================")

    t, q, x, v, i = model.simulate(
        t_end=5.0,
        freq=3.0,
        Q_amp=100e-9,
        alpha=1.0
    )

    xc = np.clip(x, 0, 1)

    print(f"x range : {xc.min():.4f} -> {xc.max():.4f}")
    print(f"q range : {q.min()*1e9:.2f} -> {q.max()*1e9:.2f} nC")
    print(f"v range : {v.min():.4f} -> {v.max():.4f} V")
    print(f"C range : {model.C(xc).min()*1e9:.2f} -> {model.C(xc).max()*1e9:.2f} nF")

    # zero crossing check
    v_zero = t[np.where(np.diff(np.sign(v)))[0]]
    q_zero = t[np.where(np.diff(np.sign(q)))[0]]

    print("\nZero crossings")
    print("v :", np.round(v_zero[:6], 3))
    print("q :", np.round(q_zero[:6], 3))

    # simple pass/fail tests
    passed = True

    if np.any(xc < -1e-10) or np.any(xc > 1 + 1e-10):
        print("FAIL : x left [0,1]")
        passed = False

    if not np.allclose(v_zero, q_zero, atol=1e-2):
        print("FAIL : zero crossings do not match")
        passed = False

    if passed:
        print("\nPASS : alpha=1 validation successful")

    return t, q, x, v, i
