# Geostradamus

A Python library for geospatial time series forecasting that takes into account spatial relationships between data points.

## Features

- Geospatial-aware time series forecasting
- Support for weighted predictions based on distance
- Integration with popular forecasting models
- Built-in data validation using Pandera
- Type-safe implementations with Pydantic

## Installation

```bash
pip install geostradamus
```

## Quick Start

Here's a simple example of how to use Geostradamus:

```python
import polars as pl
from geostradamus import Geostradamus
from kernel import GaussianKernel
from smoother import weighted_mean_smoother
from datetime import datetime

# Create sample data
data = pl.read_csv("data/combined.csv")

# Configure model settings
settings = {
    "changepoint_prior_scale": 0.05,
    "seasonality_mode": "additive",
    "yearly_seasonality": False
}

## make model class with your distance kernel and the way you want to create the line(gaussian and weighted_mean_smoother are defualted)
m = Geostradamus(
    kernel=GaussianKernel(500),
    smoother=weighted_mean_smoother,
    data=df,
)

# Fit model and get predictions
predict = m.fit(42, -72, periods=30, model_settings=settings)
predict.head(5)
```

## Data Model

Geostradamus uses strongly typed schemas for data validation:

```python
class TimeSeriesSchema:
    date: datetime       # Timestamp of the observation
    y: float            # Target variable
    lat: float          # Latitude
    lng: float          # Longitude
```

## Model Configuration

Currently supported models:
- Prophet (with geospatial weighting)

Example configuration:
```python
settings = ProphetSettings(
    changepoint_prior_scale=0.05,
    seasonality_mode="additive",
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    interval_width=0.95
)
```

## Development

This project uses modern Python development tools:
- Ruff for linting and formatting
- MyPy for type checking
- Pre-commit hooks for code quality
- Pandera for runtime data validation

To set up the development environment:

```bash
# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## License

MIT License
