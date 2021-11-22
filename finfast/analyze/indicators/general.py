import numba
import numpy as np

from . import general_kernels as kernels

from typing import Literal, Optional


def delta(x: np.ndarray, interval: int) -> np.ndarray:
    """Returns the delta between two timeseries with an interval offset

    Args:
        x (np.ndarray): n-vector timeseries
        interval (int): Interval of the deltas

    Raises:
        ValueError: Interval is less than 0

    Returns:
        np.ndarray: n-vector where the i-th entry is x[i] - x[i-interval]. The j
        leading entries are NaN, where j is the interval.
    """
    if interval < 0:
        raise ValueError("interval less than 0")

    return kernels.delta(x, interval)


def roc(
    x: np.ndarray, interval: int, method: Literal["div", "log"] = "div"
) -> np.ndarray:
    """Returns the rate of change of a timeseries

    Args:
        x (np.ndarray): n-vector timeseries
        interval (int): Interval of the rate of change
        method (Literal["div", "log"], optional): Method to compute the rate of
          change. Defaults to "div".

    Raises:
        ValueError: Interval is less than 0

    Returns:
        np.ndarray: n-vector where the i-th entry is the rate of change between
        x[i] and x[i-interval]. The j leading entries are NaN, where j
        is the interval.
    """
    if interval < 0:
        raise ValueError("interval must be greater than or equal to 0")

    if method == "div":
        return kernels.roc_div(x, interval)
    elif method == "log":
        return kernels.roc_log(x, interval)
    raise RuntimeError("unreachable")


def returns(x: np.ndarray, method: Literal["div", "log"] = "div") -> np.ndarray:
    """Returns the returns of a timeseries

    Args:
        x (np.ndarray): n-vector timeseries
        method (Literal["div", "log"], optional): Method to compute the returns.
          Defaults to "div".

    Returns:
        np.ndarray: n-vector where the i-th entry is the i-th return. The 0-th
        entry is always NaN.
    """
    return roc(x, 1, method)
