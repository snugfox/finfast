import numba
import numpy as np


# @numba.njit(cache=True, fastmath=True)
def beta(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    rp_cent = rp - np.mean(rp, axis=1, keepdims=True)
    rb_cent = rb - np.mean(rb, axis=1, keepdims=True)
    rb_var = np.mean(np.square(rb_cent), axis=1, keepdims=True)
    cov = (rp_cent @ rb_cent.T) / rp.shape[1]
    return cov / rb_var.T


# @numba.njit(cache=True, fastmath=True)
def alpha(rp: np.ndarray, rb: np.ndarray, rf: np.ndarray) -> np.ndarray:
    return (
        (rf - np.mean(rb, axis=1, keepdims=True)).T * beta(rp, rb)
        + np.mean(rp, axis=1, keepdims=True)
        - rf
    )


# @numba.njit(cache=True, fastmath=True)
def sharpe(rp: np.ndarray, rf: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return (np.mean(rp, axis=1, keepdims=True) - rf) / np.std(
            rp, axis=1, keepdims=True
        )


# @numba.njit(cache=True, fastmath=True)
def treynor(rp: np.ndarray, rb: np.ndarray, rf: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return (np.mean(rp, axis=1, keepdims=True) - rf) / beta(rp, rb)


# @numba.njit(cache=True, fastmath=True)
def sortino(rp: np.ndarray, rf: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return (np.mean(rp, axis=1, keepdims=True) - rf) / np.std(
            np.minimum(rp, 0), axis=1, keepdims=True
        )


# @numba.njit(cache=True, fastmath=True)
def tracking_error(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    rp_expanded = np.expand_dims(rp, 1)
    rb_expanded = np.expand_dims(rb, 0)
    return np.std(rp_expanded - rb_expanded, axis=2)


# @numba.njit(cache=True, fastmath=True)
def information(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    eps = np.finfo(rp.dtype).tiny
    return (
        np.mean(rp, axis=1, keepdims=True) - np.mean(rb, axis=1, keepdims=True).T
    ) / (tracking_error(rp, rb) + eps)


# @numba.njit(cache=True, fastmath=True)
def up_capture(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    eps = np.finfo(rp.dtype).tiny
    rp_expanded = np.expand_dims(rp, 1)
    rb_expanded = np.expand_dims(rb, 0)
    up_mask = rb_expanded > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum((up_mask * rp_expanded) / rb_expanded, axis=2) / (
            np.count_nonzero(up_mask, axis=2)
        )


# @numba.njit(cache=True, fastmath=True)
def down_capture(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    eps = np.finfo(rp.dtype).tiny
    rp_expanded = np.expand_dims(rp, 1)
    rb_expanded = np.expand_dims(rb, 0)
    down_mask = rb_expanded < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum((down_mask * rp_expanded) / rb_expanded, axis=2) / (
            np.count_nonzero(down_mask, axis=2)
        )


# @numba.njit(cache=True, fastmath=True)
def capture(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    eps = np.finfo(rp.dtype).tiny
    rp_expanded = np.expand_dims(rp, 1)
    rb_expanded = np.expand_dims(rb, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.mean(rp_expanded / rb_expanded, axis=2)
