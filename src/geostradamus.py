""" "Geostradamus: A Python package for geospatial data analysis and visualization.

Your time series models are built from 3 components:
- A kernel, fits the date for weights based on lat/lon.
- A smoother, fits the line based on the weights and values.
- A model, the time-series model that uses the smoother to fit the data.

"""

from dataclasses import dataclass
from typing import Callable
import polars as pl
from data_models import (
    FitTimeSeriesSchema,
    TimeSeriesSchema,
    ActiveModels,
    WeightedTimeSeriesSchema,
    TimeSeriesSchemaWithDistance,
)
from kernel import Kernel, GaussianKernel
from utils import haversine
from time_series import run_prophet
from smoother import weighted_mean_smoother


@dataclass
class Geostradamus:
    """
    Geostradamus class for geospatial time series forecasting.

    Args:
        kernel: Kernel function for weighting.
        smoother: Smoother function for fitting the data.
        data: Model function for time series forecasting.
    """

    data: TimeSeriesSchema
    kernel: Kernel = GaussianKernel()
    smoother: Callable = weighted_mean_smoother

    def fit(
        self,
        lat: float,
        lng: float,
        model: ActiveModels = ActiveModels.prophet,
        distance: pl.Series = None,
        weights: pl.Series = None,
        other_columns: list[str] | None = None,
        model_settings: dict = {},
        periods: int = 0,
    ) -> FitTimeSeriesSchema | ValueError:
        """
        Fit the model to the data.

        Args:
            lat: Latitude for the kernel.
            lng: Longitude for the kernel.
            model: Model for time series forecasting.
            distance: Distance for the kernel.
            weights: Weights for values.
            other_columns: Other columns to include in the output.
            model_settings: Additional settings for the model.

        Returns:
            Fitted model parameters.
        """

        if model not in ActiveModels.__members__:
            raise ValueError(
                f"Invalid model: {model}. Must be one of {list(ActiveModels)}."
            )

        # Apply kernel to get weights
        if distance is None:
            self.data = self.get_haversine_distance(lat, lng)

        if weights is None:
            self.data = self.get_weights(kernel=self.kernel)

        # Aggregate data and smooth
        self.data = self.smooth_series(self.data)

        ## apply model
        if model == ActiveModels.prophet:
            return run_prophet(
                data=self.data,
                other_columns=other_columns,
                model_settings=model_settings,
                periods=periods,
            )
        else:
            raise ValueError(
                f"Invalid model: {model}. Must be one of {list(ActiveModels)}."
            )

    def get_haversine_distance(
        self, lat: float, lng: float
    ) -> TimeSeriesSchemaWithDistance:
        """
        Calculate haversine distance between two points.

        Args:
            lat: Latitude of the origin point.
            lng: Longitude of the origin point.

        Returns:
            TimeSeriesSchemaWithDistance: DataFrame with haversine distance.
        """
        return self.data.with_columns(
            haversine(lat, lng, pl.col("lat"), pl.col("lng")).alias("distance")
        )

    def get_weights(self, kernel: Kernel) -> WeightedTimeSeriesSchema:
        """
        Apply kernel to get weights.

        Args:
            kernel: Kernel function for weighting.

        Returns:
            WeightedTimeSeriesSchema: DataFrame with weights.
        """
        return self.data.with_columns(kernel(pl.col("distance")).alias("weights"))

    def smooth_series(self, data: WeightedTimeSeriesSchema) -> WeightedTimeSeriesSchema:
        """
        Smooth the time series data.

        Args:
            data: Time series data.

        Returns:
            FitTimeSeriesSchema: Smoothed time series data.
        """
        return data.with_columns(
            self.smoother(pl.col("y"), pl.col("weights")).alias(
                "y"
            )  ## overwrite y with smoothed value
        )
