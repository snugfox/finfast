import torch

from finfast_torch import kernels

from typing import NamedTuple


def linear_combination_norm_mean(
    means: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Returns the linear combination of multiple normal distributions

    Args:
        means (torch.Tensor): Batch of mean row vectors (B, 1, N)
        weights (torch.Tensor): Batch of weight row vectors (B, W, N)

    Returns:
        torch.Tensor: Batch of mean column vectors (B, W, 1)
    """
    if not (
        means.ndim == weights.ndim == 3
        and means.shape[0] == weights.shape[0]
        and means.shape[1] == 1
        and means.shape[2] == weights.shape[2]
    ):
        raise RuntimeError(
            f"mean and weights must be of the dimensions (B, 1, N) and (B, W, N); got {means.shape} and {weights.shape}, respectively"
        )
    return kernels.linear_combination_norm_mean(means, weights)


def linear_combination_norm_var(
    cov: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Returns the linear combination of multiple normal distributions

    Args:
        cov (torch.Tensor): Batch of covariance matricies (B, N, N); column
        major matricies are optimal
        weights (torch.Tensor): Batch of weight row vectors (B, W, N)

    Returns:
        torch.Tensor: Batch of variance column vectors (B, W, 1)
    """
    if not (
        cov.ndim == weights.ndim == 3
        and cov.shape[0] == weights.shape[0]
        and cov.shape[1] == cov.shape[2] == weights.shape[2]
    ):
        raise RuntimeError(
            f"cov and weights must be of the dimensions (B, N, N) and (B, W, N); got {cov.shape} and {weights.shape}, respectively"
        )
    return kernels.linear_combination_norm_var(cov, weights)


class EfficientFrontierResult(NamedTuple):
    """Mean and variance results for an efficient frontier

    Attributes:
        mean (torch.Tensor): Batch of mean column vectors (B, W, 1)
        var (torch.Tensor): Batch of variance column vectors (B, W, 1)
    """

    mean: torch.Tensor
    var: torch.Tensor


def linear_combination_norm(
    mean: torch.Tensor, cov: torch.Tensor, weights: torch.Tensor
) -> EfficientFrontierResult:
    """Generates an efficient frontier from a batch of normally distributed
    returns and possible weights

    Args:
        mean (torch.Tensor): Batch of mean vectors (B, 1, N)
        cov (torch.Tensor): Batch of covariance matricies (B, N, N); column
        major matricies are optimal
        weights (torch.Tensor): Batch of weight vectors (B, W, N)

    Returns:
        EfficientFrontierResult: Batches of mean and variance column vectors for
        the weighted portfolios
    """
    if not (
        mean.ndim == cov.ndim == weights.ndim == 3
        and mean.shape[0] == cov.shape[0] == weights.shape[0]
        and mean.shape[1] == 1
        and mean.shape[2] == cov.shape[1] == cov.shape[2] == weights.shape[2]
    ):
        raise RuntimeError(
            f"mean, cov, and weights must be of the dimensions (B, 1, N), (B, N, N), and (B, W, N); got {mean.shape}, {cov.shape}, and {weights.shape}, respectively"
        )
    return EfficientFrontierResult(
        kernels.linear_combination_norm_mean(mean, weights),
        kernels.linear_combination_norm_var(cov, weights),
    )
