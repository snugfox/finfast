import numba
import numpy as np


@numba.njit(
    ["float64(float64[:],float64[:])", "float32(float32[:],float32[:])"],
    cache=True,
    fastmath=True,
)
def beta_numpy_single(rp: np.ndarray, rb: np.ndarray) -> np.floating:
    rp_centered = rp - np.mean(rp)
    rb_centered = rb - np.mean(rb)
    rb_var: np.floating = np.mean(np.square(rb_centered))
    cov = np.mean(rp_centered * rb_centered)
    return cov / rb_var


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:])",
    ],
    cache=True,
)
def beta_numpy(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    out = np.empty((rp.shape[0], 1), dtype=rp.dtype)
    for mi in range(rp.shape[0]):
        out[mi, 0] = beta_numpy_single(rp[mi, :], rb[mi, :])
    return out


def beta(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Returns the beta of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        np.ndarray: Beta
    """
    return beta_numpy(rp, rb)


@numba.njit(
    [
        "float64(float64[:],float64[:],float64)",
        "float32(float32[:],float32[:],float32)",
    ],
    cache=True,
    fastmath=True,
)
def alpha_numpy_single(rp: np.ndarray, rb: np.ndarray, rf: np.floating) -> np.floating:
    return np.mean(rp) - (rf + (np.mean(rb) - rf) * beta_numpy_single(rp, rb))


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:],float32[:,:])",
    ],
    cache=True,
)
def alpha_numpy(rp: np.ndarray, rb: np.ndarray, rf: np.ndarray) -> np.ndarray:
    out = np.empty((rp.shape[0], 1), dtype=rp.dtype)
    for mi in range(rp.shape[0]):
        out[mi, 0] = alpha_numpy_single(rp[mi, :], rb[mi, :], rf[mi, 0])
    return out


def alpha(rp: np.ndarray, rb: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """Returns the alpha of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns
        rf (np.ndarray): Risk-free rate

    Returns:
        np.ndarray: Alpha
    """
    return alpha_numpy(rp, rb, rf)


@numba.njit(
    ["float64(float64[:],float64)", "float32(float32[:],float32)"],
    cache=True,
    fastmath=True,
)
def sharpe_numpy_single(rp: np.ndarray, rf: np.floating) -> np.floating:
    return (np.mean(rp) - rf) / np.std(rp)


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:])",
    ],
    cache=True,
)
def sharpe_numpy(rp: np.ndarray, rf: np.ndarray) -> np.ndarray:
    out = np.empty((rp.shape[0], 1), dtype=rp.dtype)
    for mi in range(rp.shape[0]):
        out[mi, 0] = sharpe_numpy_single(rp[mi, :], rf[mi, 0])
    return out


def sharpe(rp: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """Returns the Sharpe ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rf (np.ndarray): Risk-free rate

    Returns:
        np.ndarray: Sharpe ratio
    """
    return sharpe_numpy(rp, rf)


@numba.njit(
    [
        "float64(float64[:],float64[:],float64)",
        "float32(float32[:],float32[:],float32)",
    ],
    cache=True,
    fastmath=True,
)
def treynor_numpy_single(
    rp: np.ndarray, rb: np.ndarray, rf: np.floating
) -> np.floating:
    return (np.mean(rp) - rf) / beta_numpy_single(rp, rb)


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:],float32[:,:])",
    ],
    cache=True,
)
def treynor_numpy(rp: np.ndarray, rb: np.ndarray, rf: np.ndarray) -> np.ndarray:
    out = np.empty((rp.shape[0], 1), dtype=rp.dtype)
    for mi in range(rp.shape[0]):
        out[mi, 0] = treynor_numpy_single(rp[mi, :], rb[mi, :], rf[mi, 0])
    return out


def treynor(rp: np.ndarray, rb: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """Returns the Treynor ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns
        rf (np.ndarray): Risk-free rate

    Returns:
        np.ndarray: Treynor ratio
    """
    return treynor_numpy(rp, rb, rf)


@numba.njit(
    ["float64(float64[:],float64)", "float32(float32[:],float32)"],
    cache=True,
    fastmath=True,
)
def sortino_numpy_single(rp: np.ndarray, rf: np.floating) -> np.floating:
    return (np.mean(rp) - rf) / np.std(np.minimum(rp, 0))


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:])",
    ],
    cache=True,
)
def sortino_numpy(rp: np.ndarray, rf: np.ndarray) -> np.ndarray:
    out = np.empty((rp.shape[0], 1), dtype=rp.dtype)
    for mi in range(rp.shape[0]):
        out[mi, 0] = sortino_numpy_single(rp[mi, :], rf[mi, 0])
    return out


def sortino(rp: np.ndarray, rf: np.ndarray) -> np.ndarray:
    """Returns the Sortino ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rf (np.ndarray): Risk-free rate

    Returns:
        np.ndarray: Sortino ratio
    """
    return sortino_numpy(rp, rf)


@numba.njit(
    ["float64(float64[:],float64[:])", "float32(float32[:],float32[:])"],
    cache=True,
    fastmath=True,
)
def tracking_error_numpy_single(rp: np.ndarray, rb: np.ndarray) -> np.floating:
    return np.std(rp - rb)


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:])",
    ],
    cache=True,
)
def tracking_error_numpy(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    out = np.empty((rp.shape[0], 1), dtype=rp.dtype)
    for mi in range(rp.shape[0]):
        out[mi, 0] = tracking_error_numpy_single(rp[mi, :], rb[mi, :])
    return out


def tracking_error(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Returns the tracking error of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        np.ndarray: Tracking error
    """
    return tracking_error_numpy(rp, rb)


@numba.njit(
    ["float64(float64[:],float64[:])", "float32(float32[:],float32[:])"],
    cache=True,
    fastmath=True,
)
def information_numpy_single(rp: np.ndarray, rb: np.ndarray) -> np.floating:
    eps = np.finfo(rp.dtype).tiny  # Prevent division by zero when rp == rb
    return (np.mean(rp) - np.mean(rb)) / (tracking_error_numpy_single(rp, rb) + eps)


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:])",
    ],
    cache=True,
)
def information_numpy(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    out = np.empty((rp.shape[0], 1), dtype=rp.dtype)
    for mi in range(rp.shape[0]):
        out[mi, 0] = information_numpy_single(rp[mi, :], rb[mi, :])
    return out


def information(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Returns the information ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        np.ndarray: Information ratio
    """
    return information_numpy(rp, rb)


@numba.njit(
    ["float64(float64[:],float64[:])", "float32(float32[:],float32[:])"],
    cache=True,
    fastmath=True,
)
def up_capture_numpy_single(rp: np.ndarray, rb: np.ndarray) -> np.floating:
    eps = np.finfo(rp.dtype).tiny  # Prevent division by zero when rb == 0.0
    up_mask = rb > 0
    n_nonzero = np.count_nonzero(up_mask)
    if n_nonzero == 0:
        return np.nan
    return np.sum(up_mask * rp / (rb + eps)) / n_nonzero


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:])",
    ],
    cache=True,
)
def up_capture_numpy(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    out = np.empty((rp.shape[0], 1), dtype=rp.dtype)
    for mi in range(rp.shape[0]):
        out[mi, 0] = up_capture_numpy_single(rp[mi, :], rb[mi, :])
    return out


def up_capture(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Returns the upside capture ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        np.ndarray: Upside capture ratio
    """
    return up_capture_numpy(rp, rb)


@numba.njit(
    ["float64(float64[:],float64[:])", "float32(float32[:],float32[:])"],
    cache=True,
    fastmath=True,
)
def down_capture_numpy_single(rp: np.ndarray, rb: np.ndarray) -> np.floating:
    eps = np.finfo(rp.dtype).tiny  # Prevent division by zero when rb == 0.0
    down_mask = rb < 0
    n_nonzero = np.count_nonzero(down_mask)
    if n_nonzero == 0:
        return np.nan
    return np.sum(down_mask * rp / rb) / n_nonzero


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:])",
    ],
    cache=True,
)
def down_capture_numpy(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    out = np.empty((rp.shape[0], 1), dtype=rp.dtype)
    for mi in range(rp.shape[0]):
        out[mi, 0] = down_capture_numpy_single(rp[mi, :], rb[mi, :])
    return out


def down_capture(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Returns the downside capture ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        np.ndarray: Downside capture ratio
    """
    return down_capture_numpy(rp, rb)


@numba.njit(
    [
        "float64[:,:](float64[:,:],float64[:,:])",
        "float32[:,:](float32[:,:],float32[:,:])",
    ],
    cache=True,
    fastmath=True,
)
def capture_numpy(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    eps = np.finfo(rp.dtype).tiny  # Prevent division by zero
    return up_capture_numpy(rp, rb) / (down_capture_numpy(rp, rb) + eps)


def capture(rp: np.ndarray, rb: np.ndarray) -> np.ndarray:
    """Returns the upside/downside capture ratio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        np.ndarray: Upside/downside capture ratio
    """
    return capture_numpy(rp, rb)
