import numba
import numpy as np


@numba.njit(fastmath=True)
def delta(x: np.ndarray, interval: int) -> np.ndarray:
    """See finfast.analyze.indicators.delta"""
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = x[interval:] - x[: x.shape[0] - interval]
    return out


@numba.njit(fastmath=True)
def delta_ref(x: np.ndarray, reference: np.ndarray, interval: int) -> np.ndarray:
    """See finfast.analyze.indicators.delta"""
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = x[interval:] - reference[: reference.shape[0] - interval]
    return out


@numba.njit(fastmath=True)
def roc_div(x: np.ndarray, interval: int) -> np.ndarray:
    """See finfast.analyze.indicators.roc"""
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = (x[interval:] / x[: x.shape[0] - interval]) - 1
    return out


@numba.njit(fastmath=True)
def roc_ref_div(x: np.ndarray, reference: np.ndarray, interval: int) -> np.ndarray:
    """See finfast.analyze.indicators.roc"""
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = (x[interval:] / reference[: reference.shape[0] - interval]) - 1
    return out


@numba.njit(fastmath=True)
def roc_log(x: np.ndarray, interval: int) -> np.ndarray:
    """See finfast.analyze.indicators.roc"""
    x_log = np.log(x)

    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = np.exp(x_log[interval:] - x_log[: x.shape[0] - interval]) - 1
    return out


@numba.njit(fastmath=True)
def roc_ref_log(x: np.ndarray, reference: np.ndarray, interval: int) -> np.ndarray:
    """See finfast.analyze.indicators.roc"""
    out = np.empty_like(x)
    out[:interval] = np.nan
    out[interval:] = (
        np.exp(
            np.log(x[interval:]) - np.log(reference[: reference.shape[0] - interval])
        )
        - 1
    )
    return out
