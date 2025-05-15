"""Smoother module."""

import polars as pl


def weighted_mean_smoother(
    values: pl.Series,
    weights: pl.Series,
) -> pl.Series:
    """
    Calculate the weighted mean of a series.

    Args:
        values: Series to smooth.
        weights: Weights for the series.

    Returns:
        Smoothed series.
    """
    return weights.dot(values).truediv(weights.sum())
