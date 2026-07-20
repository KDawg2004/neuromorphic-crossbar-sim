import numpy as np

LOG_N = [0.0, 1.1, 2.3, 3.6, 4.9, 5.0, 6.0, 7.2, 8.0, 10.0]
MIN_VALS = [-37, -36, -35, -34, -32, -32, -24, -14, -8, -7]
MAX_VALS = [22, 23, 25, 27, 28, 27, 21, 17, 15, 8]

BASE_CMIN_MAG = abs(MIN_VALS[0])
BASE_MAX_MAG = abs(MAX_VALS[0])


def _interp_at(N, log_n_table, val_table):
    log_n = np.log10(max(N, 1))
    if log_n <= log_n_table[0]:
        return val_table[0]
    if log_n >= log_n_table[-1]:
        return val_table[-1]
    return float(np.interp(log_n, log_n_table, val_table))


def degraded_biolek_defaults(N, base_cmin, base_cmax, base_cinit):
    """Apply cycle-count-dependent endurance degradation to Biolek
    Cmin/Cmax/Cinit, scaled from digitized HZO-MS Fig 3f min/max curves.

    Cmin/Cmax scaled by fractional change relative to N=0 (see LOG_N table,
    digitized off the figure, linearly interpolated -- not a fitted closed
    form). Cinit is NOT characterized in the source paper; it's held at the
    same fractional position within the [Cmin, Cmax] window it started at,
    so it stays inside the shrinking window instead of falling outside it.
    This is an assumption, not sourced -- flag in writeup.
    """
    high_mag = _interp_at(N, LOG_N, MAX_VALS)
    low_mag = abs(_interp_at(N, LOG_N, MIN_VALS))

    high_frac = high_mag / BASE_MAX_MAG
    low_frac = low_mag / BASE_CMIN_MAG

    cmax_degraded = base_cmax * high_frac
    cmin_degraded = base_cmin * low_frac

    init_frac = (base_cinit - base_cmin) / (base_cmax - base_cmin)
    cinit_degraded = cmin_degraded + init_frac * (cmax_degraded - cmin_degraded)

    return cmin_degraded, cmax_degraded, cinit_degraded