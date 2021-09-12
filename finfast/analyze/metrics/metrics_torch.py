import torch


def beta_torch(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    rp_centered = rp - torch.mean(rp, dim=1, keepdim=True)
    rb_centered = rb - torch.mean(rb, dim=1, keepdim=True)
    rb_var = torch.mean(torch.square(rb_centered), dim=1, keepdim=True)
    cov = torch.mean(rp_centered * rb_centered, dim=1, keepdim=True)
    return cov / rb_var


def beta(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the beta of a portfolio

    Args:
        rp (torch.Tensor): Portfolio returns
        rb (torch.Tensor): Benchmark returns

    Returns:
        torch.Tensor: Beta
    """
    return beta_torch(rp, rb)


def alpha_torch(rp: torch.Tensor, rb: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    return torch.mean(rp, dim=1, keepdim=True) - (
        rf + (torch.mean(rb, dim=1, keepdim=True) - rf) * beta_torch(rp, rb)
    )


def alpha(rp: torch.Tensor, rb: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Returns the alpha of a portfolio

    Args:
        rp (torch.Tensor): Portfolio returns
        rb (torch.Tensor): Benchmark returns
        rf (torch.Tensor): Risk-free rate

    Returns:
        torch.Tensor: Alpha
    """
    return alpha_torch(rp, rb, rf)


def sharpe_torch(rp: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    rp_std, rp_mean = torch.std_mean(rp, dim=1, keepdim=True)
    return (rp_mean - rf) / rp_std


def sharpe(rp: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Returns the Sharpe ratio of a portfolio

    Args:
        rp (torch.Tensor): Portfolio returns
        rf (torch.Tensor): Risk-free rate

    Returns:
        torch.Tensor: Sharpe ratio
    """
    return sharpe_torch(rp, rf)


def treynor_torch(rp: torch.Tensor, rb: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    return (torch.mean(rp, dim=1, keepdim=True) - rf) / beta_torch(rp, rb)


def treynor(rp: torch.Tensor, rb: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Returns the Treynor ratio of a portfolio

    Args:
        rp (torch.Tensor): Portfolio returns
        rb (torch.Tensor): Benchmark returns
        rf (torch.Tensor): Risk-free rate

    Returns:
        torch.Tensor: Treynor ratio
    """
    return treynor_torch(rp, rb, rf)


def sortino_torch(rp: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=rp.dtype, device=rp.device)
    return (torch.mean(rp, dim=1, keepdim=True) - rf) / torch.std(
        torch.minimum(rp, zero), dim=1, keepdim=True
    )


def sortino(rp: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Returns the Sortino ratio of a portfolio

    Args:
        rp (torch.Tensor): Portfolio returns
        rf (torch.Tensor): Risk-free rate

    Returns:
        torch.Tensor: Sortino ratio
    """
    return sortino_torch(rp, rf)


def information_torch(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(rp.dtype).tiny
    return (
        torch.mean(rp, dim=1, keepdim=True) - torch.mean(rb, dim=1, keepdim=True)
    ) / (tracking_error_torch(rp, rb) + eps)


def information(rp: torch.Tensor, rb: torch.Tensor) -> float:
    """Returns the information ratio of a portfolio

    Args:
        rp (torch.Tensor): Portfolio returns
        rb (torch.Tensor): Benchmark returns

    Returns:
        torch.Tensor: Information ratio
    """
    return information_torch(rp, rb)


def up_capture_torch(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    up_mask = rb > 0
    return torch.sum(up_mask * rp / rb, dim=1, keepdim=True) / torch.count_nonzero(
        up_mask, dim=1
    ).view(-1, 1)


def up_capture(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the upside capture ratio of a portfolio

    Args:
        rp (torch.Tensor): Portfolio returns
        rb (torch.Tensor): Benchmark returns

    Returns:
        torch.Tensor: Upside capture ratio
    """
    return up_capture_torch(rp, rb)


def down_capture_torch(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    down_mask = rb < 0
    return torch.sum(down_mask * rp / rb, dim=1, keepdim=True) / torch.count_nonzero(
        down_mask, dim=1
    ).view(-1, 1)


def down_capture(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the downside capture ratio of a portfolio

    Args:
        rp (torch.Tensor): Portfolio returns
        rb (torch.Tensor): Benchmark returns

    Returns:
        torch.Tensor: Downside capture ratio
    """
    return down_capture_torch(rp, rb)


def capture_torch(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    return up_capture_torch(rp, rb) / down_capture_torch(rp, rb)


def capture(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the upside/downside capture ratio

    Args:
        rp (torch.Tensor): Portfolio returns
        rb (torch.Tensor): Benchmark returns

    Returns:
        torch.Tensor: Upside/downside capture ratio
    """
    return capture_torch(rp, rb)


def tracking_error_torch(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    return torch.std(rp - rb, dim=1, keepdim=True)


def tracking_error(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the tracking error of a portfolio

    Args:
        rp (torch.Tensor): Portfolio returns
        rb (torch.Tensor): Benchmark returns

    Returns:
        torch.Tensor: Tracking error
    """
    return tracking_error_torch(rp, rb)
