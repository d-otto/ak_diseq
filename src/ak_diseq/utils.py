# -*- coding: utf-8 -*-

import numpy as np
import scipy as sci
import pandas as pd

def hist2(x, binfreq, area=None, cumsum=False, edge='right', from_zero=False):
    if cumsum | from_zero:
        if x.min() < 0:
            xmin = x.min()  # if the min is less than zero, put it back to the actual min
        else:
            xmin = 0
    else:
        xmin = x.min()
    # make initial array of bin edges
    bin_edges = np.arange(xmin, x.max() + 2*binfreq, binfreq)
    # adjust the bin edges based on flags
    if edge == 'center':
        bin_edges = 0.5*(bin_edges[:-1]+bin_edges[1:])
    # calculate the value for each bin
    if area is None:
        weights = np.ones(len(x)) / len(x)
        values, bins = np.histogram(x, weights=weights, bins=bin_edges)
    else:
        weights = area / area.sum()
        values, bins = np.histogram(x, weights=weights, bins=bin_edges)

    # apply fn to result
    if cumsum:
        values = np.cumsum(values)

    if edge == 'left':
        values = np.insert(values, -1, values[-1])
    elif edge == 'right':
        values = np.insert(values, 0, values[0])

    return values, bins


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def detrend(y, x=None, x_index=False):
    if x is None:
        x = y.index
    m, b, r_val, p_val, std_err = sci.stats.linregress(x, y)
    y = y - (m * x + b)
    if x_index:
        return pd.Series(y, index=x)
    else:
        return pd.Series(y)