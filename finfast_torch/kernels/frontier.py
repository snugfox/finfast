import torch


def linear_combination_norm_mean(
    means: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return weights @ torch.transpose(means, 1, 2)


def linear_combination_norm_var(
    cov: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    # The following is equivalent to torch.diag(weights.T @ cov @ weights)
    # except we only calculate the diagonal of the resulting covariance matrix
    return torch.unsqueeze(torch.sum(weights @ cov * weights, axis=2), 2)
