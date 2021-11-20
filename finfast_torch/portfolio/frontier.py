import torch
import dataclasses

from . import frontier_kernels as kernels


@dataclasses.dataclass
class LinearCombinationNormResult:
    """Normal distribution results from a linear combination of one or more
    other normal distributions

    Attributes:
        mean (torch.Tensor): n-vector where the i-th entry corresponds to the
          i-th distribution mean
        var (torch.Tensor): n-vector where the i-th entry corresponds to the
          i-th distribution variance
    """

    mean: torch.Tensor
    var: torch.Tensor


@dataclasses.dataclass
class LinearCombinationNormCovResult:
    """Normal distribution results from a linear combination of one or more
    other normal distributions

    Attributes:
        mean (torch.Tensor): n-vector where the i-th entry corresponds to the
          i-th distribution mean
        cov (torch.Tensor): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th linearly combined
          distributions
    """

    mean: torch.Tensor
    cov: torch.Tensor


def lincomb_norm(
    mean: torch.Tensor, cov: torch.Tensor, coeff: torch.Tensor
) -> LinearCombinationNormResult:
    """Returns the normal distributions for w linear combinations of n normal
    distributions

    Args:
        mean (torch.Tensor): n-vector where the i-th entry corresponds to the
          i-th distribution mean
        cov (torch.Tensor): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th distributions
        coeff (torch.Tensor): w-by-n matrix where the (i, j) entry corresponds
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
    mean: torch.Tensor, cov: torch.Tensor, coeff: torch.Tensor
) -> LinearCombinationNormCovResult:
    """Returns the normal distributions for w linear combinations of n normal
    distributions

    This function is similar to lincomb_norm, except it returns a covariance
    matrix instead of just the variances.

    Args:
        mean (torch.Tensor): n-vector where the i-th entry corresponds to the
          i-th distribution mean
        cov (torch.Tensor): n-by-n covariance matrix where the (i, j) entry
          corresponds to the covariance of the i-th and j-th distributions
        coeff (torch.Tensor): w-by-n matrix where the (i, j) entry corresponds
          to the i-th set and j-th coefficient for the j-th normal distribution

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
