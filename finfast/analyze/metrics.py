import numba
import numpy as np


@numba.njit
def beta_host(rp: np.ndarray, rb: np.ndarray) -> float:
    cov = np.cov(rp, rb)
    return cov[0, 1] / cov[1, 1]


def beta(rp: np.ndarray, rb: np.ndarray) -> float:
    """Returns the beta of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        float: Beta
    """
    return beta_host(rp, rb)


@numba.njit
def alpha_host(rp: np.ndarray, rb: np.ndarray, rf: float) -> float:
    return np.mean(rp) - (rf + (np.mean(rb) - rf) * beta_host(rp, rb))


def alpha(rp: np.ndarray, rb: np.ndarray, rf: float) -> float:
    """Returns the alpha of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns
        rf (float): Risk-free rate

    Returns:
        float: Alpha
    """
    return alpha_host(rp, rb, rf)


@numba.njit(fastmath=True)
def sharpe_host(rp: np.ndarray, rf: float) -> float:
    return (np.mean(rp) - rf) / np.std(rp)


def sharpe(rp: np.ndarray, rf: float) -> float:
    """Returns the Sharpe ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rf (float): Risk-free rate

    Returns:
        float: Sharpe ratio
    """
    return sharpe_host(rp, rf)


@numba.njit
def treynor_host(rp: np.ndarray, rb: np.ndarray, rf: float) -> float:
    return (np.mean(rp) - rf) / beta_host(rp, rb)


def treynor(rp: np.ndarray, rb: np.ndarray, rf: float) -> float:
    """Returns the Treynor ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns
        rf (float): Risk-free rate

    Returns:
        float: Treynor ratio
    """
    return treynor_host(rp, rb, rf)


@numba.njit(fastmath=True)
def sortino_host(rp: np.ndarray, rf: float) -> float:
    return (np.mean(rp) - rf) / np.std(np.minimum(rp, 0))


def sortino(rp: np.ndarray, rf: float) -> float:
    """Returns the Sortino ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rf (float): Risk-free rate

    Returns:
        float: Sortino ratio
    """
    return sortino_host(rp, rf)


@numba.njit
def information_host(rp: np.ndarray, rb: np.ndarray) -> float:
    return (np.mean(rp) - np.mean(rb)) / tracking_error_host(rp, rb)


def information(rp: np.ndarray, rb: np.ndarray) -> float:
    """Returns the information ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        float: Information ratio
    """
    return information_host(rp, rb)


@numba.njit(fastmath=True)
def up_capture_host(rp: np.ndarray, rb: np.ndarray) -> float:
    up_mask = rb > 0
    return np.sum(up_mask * rp / rb) / np.count_nonzero(up_mask)


def up_capture(rp: np.ndarray, rb: np.ndarray) -> float:
    """Returns the upside capture ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        float: Upside capture ratio
    """
    return up_capture_host(rp, rb)


@numba.njit(fastmath=True)
def down_capture_host(rp: np.ndarray, rb: np.ndarray) -> float:
    down_mask = rb < 0
    return np.sum(down_mask * rp / rb) / np.count_nonzero(down_mask)


def down_capture(rp: np.ndarray, rb: np.ndarray) -> float:
    """Returns the downside capture ratio of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        float: Downside capture ratio
    """
    return down_capture_host(rp, rb)


@numba.njit
def capture_host(rp: np.ndarray, rb: np.ndarray) -> float:
    return up_capture_host(rp, rb) / down_capture_host(rp, rb)


def capture(rp: np.ndarray, rb: np.ndarray) -> float:
    """Returns the upside/downside capture ratio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        float: Upside/downside capture ratio
    """
    return capture_host(rp, rb)


@numba.njit(fastmath=True)
def tracking_error_host(rp: np.ndarray, rb: np.ndarray) -> float:
    return np.std(rp - rb)


def tracking_error(rp: np.ndarray, rb: np.ndarray) -> float:
    """Returns the tracking error of a portfolio

    Args:
        rp (np.ndarray): Portfolio returns
        rb (np.ndarray): Benchmark returns

    Returns:
        float: Tracking error
    """
    return tracking_error_host(rp, rb)
