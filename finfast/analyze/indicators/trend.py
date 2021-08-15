import numba
import numpy as np


@numba.njit(fastmath=True)
def sma_host(x: np.ndarray, period: int) -> np.ndarray:
    last_out = np.mean(x[:period])

    out = np.empty_like(x)
    out[: period - 1] = np.nan
    out[period - 1] = last_out
    for idx in range(period, out.shape[0]):
        last_out += (x[idx] - x[idx - period]) / period
        out[idx] = last_out
    return out


def sma(x: np.ndarray, period: int) -> np.ndarray:
    """Returns a simple moving average

    Args:
      x: Any vector
      period: Period of the SMA

    Returns:
      SMA with the same length as x. The period leading values are NaN.
    """
    if period <= 0:
        raise ValueError("period must be greater than 0")
    if period == 1:
        return x.copy()
    if x.shape[0] < period: # TODO: Handle naturally in sma_host
        return np.full_like(x, np.nan)

    return sma_host(x, period)


@numba.njit(fastmath=True)
def gma_host_prod(x: np.ndarray, period: int) -> np.ndarray:
    proot = 1.0 / period
    start = np.prod(x[:period] ** proot)

    out = np.empty_like(x)
    out[: period - 1] = np.nan
    out[period - 1] = start
    out[period:] = start * np.cumprod(x[period:] / x[:-period]) ** proot
    return out


@numba.njit(fastmath=True)
def gma_host_log(x: np.ndarray, period: int) -> np.ndarray:
    return np.exp(sma_host(np.log(x), period))


def gma(x: np.ndarray, period: int, method: str = "log") -> np.ndarray:
    if period <= 0:
        raise ValueError("period must be greater than 0")
    if period == 1:
        return x.copy()

    if method == "prod":
        return gma_host_prod(x, period)
    elif method == "log":
        return gma_host_log(x, period)
    else:
        raise ValueError(f"unknown method {method}")


@numba.njit  # TODO: fastmath=True performance regression?
def ema_host(x: np.ndarray, period: int, smoothing: float) -> np.ndarray:
    mult = smoothing / (period + 1)
    mult_comp = 1.0 - mult
    last_out = np.mean(x[:period])

    out = np.empty_like(x)
    out[: period - 1] = np.nan
    out[period - 1] = last_out
    for idx in range(period, out.shape[0]):
        last_out = last_out * mult_comp + x[idx] * mult
        out[idx] = last_out
    return out


def ema(x: np.ndarray, period: int, smoothing: float = 2.0) -> np.ndarray:
    if period <= 0:
        raise ValueError("period must be greater than 0")
    if smoothing <= 0:
        raise ValueError("smoothing must be greater than 0")

    return ema_host(x, period, smoothing)


@numba.njit(fastmath=True)
def wma_host_sum(x: np.ndarray, weights: np.ndarray, period: int) -> np.ndarray:
    # TODO: Fix overflow edge cases with large values (wma_host_log?)
    last_num = np.sum(x[:period] * weights[:period])
    last_denom = np.sum(weights[:period])

    out = np.empty_like(x)
    out[: period - 1] = np.nan
    out[period - 1] = last_num / last_denom
    for idx in range(period, out.shape[0]):
        w_add, w_sub = weights[idx], weights[idx - period]
        last_num += x[idx] * w_add - x[idx - period] * w_sub
        last_denom += w_add - w_sub
        out[idx] = last_num / last_denom
    return out


@numba.njit
def wma_host_log(x: np.ndarray, weights: np.ndarray, period: int) -> np.ndarray:
    raise NotImplementedError


def wma(
    x: np.ndarray, weights: np.ndarray, period: int, method: str = "sum"
) -> np.ndarray:
    if period <= 0:
        raise ValueError("period must be greater than 0")
    if x.shape != weights.shape:
        raise ValueError(
            f"weighting dimensions {weights.shape} does not match input array {x.shape}"
        )
    if period == 1:
        return x.copy()

    if method == "sum":
        return wma_host_sum(x, weights, period)
    elif method == "log":
        return gma_host_log(x, weights, period)
    else:
        raise ValueError(f"unknown method {method}")


def vwma(x: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray:
    return wma(x, volume, period)

