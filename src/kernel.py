"""Kernel functions for weighting."""

import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Kernel(ABC):
    """
    Abstract base class for kernel functions.
    """

    @abstractmethod
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel for X.

        Args:
            X: First array of points

        Returns:
            Array of kernel values
        """
        pass


@dataclass
class GaussianKernel(Kernel):
    """
    Gaussian kernel.

    Args:
        length_scale: Length scale parameter for the kernel.
    """

    length_scale: float = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = (x - 0) ** 2 / 2 / self.length_scale**2
        return np.exp(-z)


@dataclass
class ExponentialKernel(Kernel):
    """
    Exponential kernel.

    Args:
        length_scale: Length scale parameter for the kernel.
    """

    length_scale: float = 1.0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        z = np.abs(x - 0) / self.length_scale
        return np.exp(-z)
