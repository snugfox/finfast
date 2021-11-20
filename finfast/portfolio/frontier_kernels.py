import numpy as np


def lincomb_norm_mean(mean: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Returns the means for w linear combinations of n normal distributions

    Args:
        mean (np.ndarray): n-vector where the i-th entry corresponds to the i-th
          distribution mean
        coeff (np.ndarray): w-by-n matrix where the (i, j) entry corresponds to
          the i-th set and j-th coefficient for the j-th distribution

    Returns:
        np.ndarray: w-vector where the i-th entry corresponds to the i-th
        distribution mean
    """
    return (mean[None, :] @ coeff.T)[0, :]


def lincomb_norm_var(cov: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    """Returns the variances for w linear combinations of n normal distributions

    Args:
        cov (np.ndarray): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th distributions
        coeff (np.ndarray): w-by-n matrix where the (i, j) entry corresponds to
          the i-th set and j-th coefficient for the j-th distribution

    Returns:
        np.ndarray: w-vector where the i-th entry corresponds to the i-th
        distribution variance
    """
    # The following is equivalent to torch.diag(weights.T @ cov @ weights)
    # except we only calculate the diagonal of the resulting covariance matrix.
    # We also use cov.T instead of cov since cov is symmetric and cov.T will use
    # coalesced reads.
    return np.sum(coeff @ cov.T * coeff, axis=1)


def lincomb_norm_cov(cov: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Returns the covariance matrix for w linear combinations of n normal
    distributions

    Args:
        cov (np.ndarray): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th distributions
        coeff (np.ndarray): w-by-n matrix where the (i, j) entry corresponds to
          the i-th set and j-th coefficient for the j-th distribution

    Returns:
        np.ndarray: w-by-w covariance matrix where the (i, j) entry corresponds
        to the covariance of the i-th and j-th distributions
    """
    return weights @ cov.T @ weights.T
