"""Models class for the application."""

import pandera.polars as pa
from pandera.typing import Series
import datetime
from enum import Enum
from pydantic import BaseModel


class TimeSeriesSchema(pa.DataFrameModel):
    date: Series[datetime.datetime]
    y: Series[float]
    lat: Series[float]
    lng: Series[float]

    class Config:
        coerce = True


class TimeSeriesSchemaWithDistance(TimeSeriesSchema):
    distance: Series[float]

    class Config:
        coerce = True


class WeightedTimeSeriesSchema(TimeSeriesSchemaWithDistance):
    weights: Series[float]

    class Config:
        coerce = True


class FitTimeSeriesSchema(WeightedTimeSeriesSchema):
    yhat: Series[float]

    class Config:
        coerce = True


class ActiveModels(str, Enum):
    """
    Enum for active models.
    """

    prophet = "prophet"

    def __str__(self) -> str:
        return self.value


class ModelSettings(BaseModel):
    """
    Model settings for the application.
    """

    class Config:
        arbitrary_types_allowed = True


class ProphetSettings(ModelSettings):
    """
    Settings for the Prophet model.
    """

    changepoint_prior_scale: float = 0.05
    seasonality_mode: str = "additive"
    yearly_seasonality: bool = False
    weekly_seasonality: bool = False
    daily_seasonality: bool = False
    interval_width: float = 0.95
    uncertainty_samples: int = 0
    mcmc_samples: int = 0
    seasonality_prior_scale: float = 10.0
    holidays_prior_scale: float = 10.0
    changepoint_range: float = 0.8
    growth: str = "linear"
