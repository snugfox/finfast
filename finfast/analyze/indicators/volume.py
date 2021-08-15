import numpy as np
import numba


@numba.njit
def obv_host(r: np.ndarray, volume: np.ndarray) -> np.ndarray:
    return np.cumsum(np.sign(r).astype(np.int_) * volume)


def obv(x: np.ndarray, volume: np.ndarray) -> np.ndarray:
    return obv_host(x, volume)


@numba.njit
def money_flow_host(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    return (high + low + close) / 3 * volume


def money_flow(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray
) -> np.ndarray:
    return money_flow_host(high, low, close, volume)
