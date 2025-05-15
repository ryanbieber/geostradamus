"""Calculate distance between points to weight them."""

import numpy as np
import polars as pl


def haversine(olat: float, olng: float, dlat: float, dlng: float) -> pl.Expr:
    """Calculate haversine distance between two points"""
    R = 6371  # Earth's radius in kilometers

    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [olat, olng, dlat, dlng])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c

    return distance
