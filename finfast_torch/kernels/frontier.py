import torch


def lincomb_norm_mean(mean: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """Returns the means for w linear combinations of n normal distributions

    Args:
        mean (torch.Tensor): n-vector where the i-th entry corresponds to the
          i-th distribution mean
        coeff (torch.Tensor): w-by-n matrix where the (i, j) entry corresponds
          to the i-th set and j-th coefficient for the j-th distribution

    Returns:
        torch.Tensor: w-vector where the i-th entry corresponds to the i-th
        distribution mean
    """
    return (mean[None, :] @ coeff.T)[0, :]


def lincomb_norm_var(cov: torch.Tensor, coeff: torch.Tensor) -> torch.Tensor:
    """Returns the variances for w linear combinations of n normal distributions

    Args:
        cov (torch.Tensor): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th distributions
        coeff (torch.Tensor): w-by-n matrix where the (i, j) entry corresponds
          to the i-th set and j-th coefficient for the j-th distribution

    Returns:
        torch.Tensor: w-vector where the i-th entry corresponds to the i-th
        distribution variance
    """
    # The following is equivalent to torch.diag(weights.T @ cov @ weights)
    # except we only calculate the diagonal of the resulting covariance matrix.
    # We also use cov.T instead of cov since cov is symmetric and cov.T will use
    # coalesced reads.
    return torch.sum(coeff @ cov.T * coeff, dim=1)


def lincomb_norm_cov(cov: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Returns the covariance matrix for w linear combinations of n normal
    distributions

    Args:
        cov (torch.Tensor): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th distributions
        coeff (torch.Tensor): w-by-n matrix where the (i, j) entry corresponds
          to the i-th set and j-th coefficient for the j-th distribution

    Returns:
        torch.Tensor: w-by-w covariance matrix where the (i, j) entry
        corresponds to the covariance of the i-th and j-th distributions
    """
    return weights @ cov.T @ weights.T
