import numba
import numpy as np

from typing import Optional


@numba.njit
def delta_host(x: np.ndarray, interval: int) -> np.ndarray:
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = x[interval:] - x[:-interval]
    return out


@numba.njit
def delta_host_ref(x: np.ndarray, reference: np.ndarray, interval: int) -> np.ndarray:
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = x[interval:] - reference[:-interval]
    return out


def delta(
    x: np.ndarray, interval: int = 1, reference: Optional[np.ndarray] = None
) -> np.ndarray:
    if interval < 0:
        raise ValueError("interval must be greater than or equal to 0")
    if reference is not None and x.shape != reference.shape:
        raise ValueError(
            f"reference dimensions {reference.shape} does not match input dimensions {x.shape}"
        )

    if reference is None or reference is x:
        return delta_host(x, interval)
    else:
        return delta_host_ref(x, reference, interval)


@numba.njit
def roc_host_div(x: np.ndarray, interval: int) -> np.ndarray:
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = (x[interval:] / x[:-interval]) - 1
    return out


@numba.njit
def roc_host_ref_div(x: np.ndarray, reference: np.ndarray, interval: int) -> np.ndarray:
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = (x[interval:] / reference[:-interval]) - 1
    return out


@numba.njit(fastmath=True)
def roc_host_log(x: np.ndarray, interval: int) -> np.ndarray:
    x_log = np.log(x)

    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = np.exp(x_log[interval:] - x_log[:-interval]) - 1
    return out


@numba.njit(fastmath=True)
def roc_host_ref_log(x: np.ndarray, reference: np.ndarray, interval: int) -> np.ndarray:
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = np.exp(np.log(x[interval:]) - np.log(reference[:-interval])) - 1
    return out


def roc(
    x: np.ndarray,
    interval: int,
    reference: Optional[np.ndarray] = None,
    method: str = "div",
) -> np.ndarray:
    if interval < 0:
        raise ValueError("interval must be greater than or equal to 0")
    if reference is not None and x.shape != reference.shape:
        raise ValueError(
            f"reference dimensions {reference.shape} does not match input dimensions {x.shape}"
        )

    if reference is None or reference is x:
        if method == "div":
            return roc_host_div(x, interval)
        elif method == "log":
            return roc_host_log(x, interval)
        else:
            raise ValueError(f"unknown method {method}")
    else:
        if method == "div":
            return roc_host_ref_div(x, reference, interval)
        elif method == "log":
            return roc_host_ref_log(x, reference, interval)
        else:
            raise ValueError(f"unknown method {method}")


def returns(
    x: np.ndarray, reference: Optional[np.ndarray] = None, method: str = "div"
) -> np.ndarray:
    return roc(x, 1, reference, method)
