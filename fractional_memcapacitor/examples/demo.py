import matplotlib.pyplot as plt

from ..model import BiolekMemcapacitor
from ..plotting import plot_model
from ..validation import validate_alpha_one, validate_alpha_sweep


if __name__ == "__main__":
    # Paper validation params: Cmin=50n, Cmax=200n, Cinit=100n, p=10, Joglekar window
    m = BiolekMemcapacitor(Cmin=50e-9, Cmax=200e-9, Cinit=100e-9, k=1e7, p=10,
                           window_type='joglekar')
    print(f"x_init = {m.x_init:.4f}")
    print(f"C at x_init = {m.C(m.x_init)*1e9:.2f} nF (should be ~100nF)")

    t, q, x, v, i = validate_alpha_one(m)

    plt.figure(figsize=(6,6))
    plt.plot(v, q*1e9)
    plt.xlabel("Voltage (V)")
    plt.ylabel("Charge (nC)")
    plt.title("Alpha = 1 Validation")
    plt.grid(True)
    plt.show()

    validate_alpha_sweep(m)
