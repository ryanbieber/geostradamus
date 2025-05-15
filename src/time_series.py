"""Time series module."""

from data_models import WeightedTimeSeriesSchema, FitTimeSeriesSchema, ProphetSettings
import polars as pl
import pandas as pd
from prophet import Prophet
from typing import Any


def run_prophet(
    data: WeightedTimeSeriesSchema,
    other_columns: list[str] | None = None,
    model_settings: dict | None = None,
    periods: int = 0,
) -> FitTimeSeriesSchema | ValueError:
    """
    Run the Prophet model on the time series data.

    Args:
        data: Time series data.
        **kwargs: Additional arguments for the Prophet model.

    Returns:
        FitTimeSeriesSchema: Fitted time series data with predictions.
    """
    model_settings_dict: dict[str, Any] = model_settings or {}
    prophet_settings = ProphetSettings(**model_settings_dict)

    model = Prophet(**prophet_settings.model_dump())
    ## rename columns to fit prophet
    schema = (
        data.with_columns(
            pl.col("date").alias("ds"),
            pl.col("y").alias("y"),
        )
        .select(["ds", "y"])
        .to_pandas()  ## prophet needs pandas dataframe
    )
    schema["ds"] = pd.to_datetime(
        schema["ds"]
    )  ## convert to datetime as the to_pandas() method breaks prophet
    model.fit(schema)
    future = model.make_future_dataframe(periods=periods)
    predictions = model.predict(future)

    polars_predictions = pl.from_pandas(predictions)
    ## merge predictions with original data

    return_columns = (
        polars_predictions.columns + other_columns
        if other_columns
        else polars_predictions.columns
    )

    for col in return_columns:
        if col not in polars_predictions.columns:
            raise ValueError(f"Column {col} not found in predictions")

    return data.join(
        polars_predictions.select(return_columns).with_columns(
            pl.col("ds").dt.date().alias("ds")
        ),
        left_on="date",
        right_on="ds",
        how="left",
    )
