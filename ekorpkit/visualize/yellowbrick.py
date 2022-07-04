import matplotlib.pyplot as plt
from inspect import getfullargspec as getargspec
from ekorpkit import eKonf


def yellowbrick_features(ax=None, x=None, y=None, data=None, **kwargs):
    from yellowbrick.features import (
        RadViz,
        Rank1D,
        Rank2D,
        JointPlotVisualizer,
        ParallelCoordinates,
        PCA,
        Manifold,
    )

    rcParams = {} or kwargs.get(eKonf.Keys.rcPARAMS)
    _name_ = kwargs.get(eKonf.Keys.METHOD_NAME)
    _method_ = kwargs.get(eKonf.Keys.METHOD)

    if ax is None:
        ax = plt.gca()
    if x is not None:
        x = data[x]
    if y is not None:
        y = data[y]
    viz = locals()[_name_](ax=ax, **rcParams)
    for _m in _method_:
        _fn = getattr(viz, _m)
        _fn_args = getargspec(_fn).args
        if "X" in _fn_args and "y" in _fn_args:
            _fn(x, y)
        elif "X" in _fn_args:
            _fn(x)
        else:
            _fn()
