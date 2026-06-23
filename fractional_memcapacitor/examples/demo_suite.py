from ..model import BiolekMemcapacitor
from ..plotting import plot_validation_suite

if __name__ == "__main__":
    m = BiolekMemcapacitor(Cmin=50e-9, Cmax=200e-9, Cinit=100e-9, k=1e7, p=10,
                           window_type='joglekar')
    plot_validation_suite(m)