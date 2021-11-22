import numba
import numpy as np

from . import general_kernels as kernels

from typing import Literal, Optional


def delta(
    x: np.ndarray, interval: int = 1, reference: Optional[np.ndarray] = None
) -> np.ndarray:
    """Returns the delta between two timeseries with an interval offset

    Args:
        x (np.ndarray): n-vector timeseries
        interval (int, optional): Interval offset. Defaults to 1.
        reference (Optional[np.ndarray], optional): n-vector reference
          timeseries. If None, x is used as the reference. Defaults to None.

    Raises:
        ValueError: Interval is less than 0 or reference has a different shape
          than x

    Returns:
        np.ndarray: n-vector where the i-th entry is x[i] -
        reference[i-interval]. The j leading entries are NaN, where j is the
        interval.
    """
    if interval < 0:
        raise ValueError("interval less than 0")
    if reference is not None and x.shape != reference.shape:
        raise ValueError(
            f"reference dimensions {reference.shape} does not match input dimensions {x.shape}"
        )

    if reference is None or reference is x:
        return kernels.delta(x, interval)
    else:
        return kernels.delta_ref(x, reference, interval)


def roc(
    x: np.ndarray,
    interval: int,
    reference: Optional[np.ndarray] = None,
    method: Literal["div", "log"] = "div",
) -> np.ndarray:
    """Returns the rate of change of a timeseries

    Args:
        x (np.ndarray): n-vector timeseries
        interval (int): Interval
        reference (Optional[np.ndarray], optional): n-vector reference
          timeseries. If None, x is used as the reference. Defaults to None.
        method (Literal["div", "log"], optional): Method to compute the rate of
          change. Defaults to "div".

    Raises:
        ValueError: Interval is less than 0 or reference has a different shape
          than x

    Returns:
        np.ndarray: n-vector where the i-th entry is the rate of change between
        x[i] and reference[i-interval]. The j leading entries are NaN, where j
        is the interval.
    """
    if interval < 0:
        raise ValueError("interval must be greater than or equal to 0")
    if reference is not None and x.shape != reference.shape:
        raise ValueError(
            f"reference dimensions {reference.shape} does not match input dimensions {x.shape}"
        )

    if reference is None or reference is x:
        if method == "div":
            return kernels.roc_div(x, interval)
        elif method == "log":
            return kernels.roc_log(x, interval)
    else:
        if method == "div":
            return kernels.roc_ref_div(x, reference, interval)
        elif method == "log":
            return kernels.roc_ref_log(x, reference, interval)
    raise RuntimeError("unreachable")


def returns(
    x: np.ndarray,
    reference: Optional[np.ndarray] = None,
    method: Literal["div", "log"] = "div",
) -> np.ndarray:
    """Returns the returns of a timeseries

    Args:
        x (np.ndarray): n-vector timeseries
        reference (Optional[np.ndarray], optional): n-vector reference
          timeseries. If None, x is used as the reference. Defaults to None.
        method (Literal["div", "log"], optional): Method to compute the returns.
          Defaults to "div".

    Returns:
        np.ndarray: n-vector where the i-th entry is the i-th return. The 0-th
        entry is always NaN.
    """
    return roc(x, 1, reference, method)
