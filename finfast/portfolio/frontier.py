import numpy as np
import dataclasses

from . import frontier_kernels as kernels


@dataclasses.dataclass
class LinearCombinationNormResult:
    """Normal distribution results from a linear combination of one or more
    other normal distributions

    Attributes:
        mean (np.ndarray): n-vector where the i-th entry corresponds to the i-th
          distribution mean
        var (np.ndarray): n-vector where the i-th entry corresponds to the i-th
          distribution variance
    """

    mean: np.ndarray
    var: np.ndarray


@dataclasses.dataclass
class LinearCombinationNormCovResult:
    """Normal distribution results from a linear combination of one or more
    other normal distributions

    Attributes:
        mean (np.ndarray): n-vector where the i-th entry corresponds to the i-th
          distribution mean
        cov (np.ndarray): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th linearly combined
          distributions
    """

    mean: np.ndarray
    cov: np.ndarray


def lincomb(rp: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Returns the normal distributions for w linear combinations of p
    portfolios

    Args:
        rp (np.ndarray): p-by-n matrix where the (i, j) entry corresponds to the
          j-th return of the i-th portfolio
        coeff (np.ndarray): w-by-p matrix where the (i, j) entry corresponds to
          the i-th set and j-th coefficient for the j-th portfolio

    Returns:
        np.ndarray: w-by-n matrix where the (i, j) entry corresponds to the
        j-th return for i-th portfolio from the linear combination
    """
    if not (
        rp.ndim == coeff.ndim == 2
        and rp.shape[0] == coeff.shape[1]
    ):
        raise RuntimeError
    return kernels.lincomb(rp, coeff)


def lincomb_norm(
    mean: np.ndarray, cov: np.ndarray, coeff: np.ndarray
) -> LinearCombinationNormResult:
    """Returns the normal distributions for w linear combinations of n normal
    distributions

    Args:
        mean (np.ndarray): n-vector where the i-th entry corresponds
          to the i-th distribution mean
        cov (np.ndarray): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th distributions
        coeff (np.ndarray): w-by-n matrix where the (i, j) entry corresponds
          to the i-th set and j-th coefficient for the j-th normal distribution

    Returns:
        LinearCombinationResult: w normal distributions from the linear
        combination
    """
    if not (
        mean.ndim == 1
        and cov.ndim == coeff.ndim == 2
        and mean.shape[0] == cov.shape[0] == cov.shape[1] == coeff.shape[1]
    ):
        raise RuntimeError
    return LinearCombinationNormResult(
        mean=kernels.lincomb_norm_mean(mean, coeff),
        var=kernels.lincomb_norm_var(cov, coeff),
    )


def lincomb_norm_cov(
    mean: np.ndarray, cov: np.ndarray, coeff: np.ndarray
) -> LinearCombinationNormCovResult:
    """Returns the normal distributions for w linear combinations of n normal
    distributions

    This function is similar to lincomb_norm, except it returns a covariance
    matrix instead of just the variances.

    Args:
        mean (np.ndarray): n-vector where the i-th entry corresponds to the i-th
          distribution mean
        cov (np.ndarray): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th distributions
        coeff (np.ndarray): w-by-n matrix where the (i, j) entry corresponds to
          the i-th set and j-th coefficient for the j-th normal distribution

    Returns:
        LinearCombinationCovResult: w normal distributions from the linear
        combination
    """
    if not (
        mean.ndim == 1
        and cov.ndim == coeff.ndim == 2
        and mean.shape[0] == cov.shape[0] == cov.shape[1] == coeff.shape[1]
    ):
        raise RuntimeError
    return LinearCombinationNormCovResult(
        mean=kernels.lincomb_norm_mean(mean, coeff),
        cov=kernels.lincomb_norm_cov(cov, coeff),
    )
