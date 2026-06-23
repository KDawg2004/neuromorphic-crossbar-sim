from .TeamMemristor import TEAMMemristor


def plot(*args, **kwargs):
    from .teamPlotting import plot as _plot
    return _plot(*args, **kwargs)


__all__ = ["TEAMMemristor", "plot"]
