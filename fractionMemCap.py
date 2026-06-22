import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


class FractionalBiolekMemcapacitor:

    def __init__(
        self,
        Cmin=10e-9,
        Cmax=10e-6,
        Cinit=100e-9,
        k=1e7,
        p=1,
        alpha=0.8,
        IC=0.0,
    ):

        self.Cmin = Cmin
        self.Cmax = Cmax
        self.Cinit = Cinit
        self.k = k
        self.p = p
        self.alpha = alpha
        self.IC = IC

        self.x_init = (
            (1 / Cinit - 1 / Cmax)
            / (1 / Cmin - 1 / Cmax)
        )

    # ---------------------------------

    def DM(self, x):

        return (
            1 / self.Cmax
            + (1 / self.Cmin - 1 / self.Cmax) * x
        )

    def C(self, x):

        return 1 / self.DM(x)

    def window(self, x):

        return 1 - (2 * x - 1) ** (2 * self.p)

    def voltage(self, q, x):

        return self.DM(x) * (
            q + self.IC * self.Cinit
        )

    # ---------------------------------

    def current_drive(self, t, freq=1.0, amp=1e-3):

        return amp * np.sin(2 * np.pi * freq * t)

    # ---------------------------------

    def simulate(
        self,
        t_end=2.0,
        freq=1.0,
        amp=1e-3,
        n_points=3000,
    ):

        t = np.linspace(0, t_end, n_points)

        dt = t[1] - t[0]

        i = self.current_drive(t, freq, amp)

        #
        # ordinary charge integration
        #

        q = np.zeros(n_points)

        for n in range(1, n_points):

            q[n] = q[n - 1] - i[n - 1] * dt

        #
        # fractional Caputo state equation
        #

        x = np.zeros(n_points)

        x[0] = self.x_init

        coeff = dt ** self.alpha / gamma(self.alpha + 1)

        for n in range(1, n_points):

            total = 0.0

            for j in range(n):

                weight = (
                    (n - j) ** self.alpha
                    - (n - j - 1) ** self.alpha
                )

                total += (
                    weight
                    * self.k
                    * q[j]
                    * self.window(x[j])
                )

            x[n] = self.x_init + coeff * total

            #
            # keep state physical
            #

            x[n] = np.clip(x[n], 0.0, 1.0)

        v = self.voltage(q, x)

        return t, q, x, v, i

    # ---------------------------------

    def plot(self, **kwargs):

        t, q, x, v, i = self.simulate(**kwargs)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        axes[0].plot(v, q)
        axes[0].set_title("q-v loop")

        axes[1].plot(t, x)
        axes[1].set_title("fractional state")

        axes[2].plot(t, self.C(x) * 1e9)
        axes[2].set_title("capacitance (nF)")

        axes[3].plot(t, q, label="q")
        axes[3].plot(t, i, label="i")
        axes[3].legend()

        for ax in axes:
            ax.grid(True)

        plt.tight_layout()

        return t, q, x, v, i


if __name__ == "__main__":

   alphas = [1.0, 0.8, 0.6, 0.4]

plt.figure(figsize=(7,5))

for a in alphas:

    m = FractionalBiolekMemcapacitor(alpha=a)

    t, q, x, v, i = m.simulate(
        t_end=2,
        freq=1,
        amp=1e-3,
    )

    plt.plot(v, q, label=f"α={a}")

    plt.xlabel("Voltage (V)")
    plt.ylabel("Charge (C)")
    plt.title("Fractional Memcapacitor q-v loops")
    plt.grid(True)
    plt.legend()

    plt.show()

    print("x range:", x.min(), x.max())
    print("q range:", q.min(), q.max())