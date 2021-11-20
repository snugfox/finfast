import torch

from . import metrics_kernels as kernels

from typing import Optional


def __expect_rp_rb_rf(
    rp: Optional[torch.Tensor], rb: Optional[torch.Tensor], rf: Optional[torch.Tensor]
) -> None:
    """Raises an exception if any of rp, rb, or rf have invalid dtypes or shapes

    It will raise a ValueError if any of the following conditions are not met:
        - rp and rb are both 2-D tensors
        - rp and rb have dimensions (P, N) and (B, N), respectively
        - rp and rb are both floating dtypes
        - rf is a 0-D tensor (scalar)
        - rf is a floating dtype

    If any arguments are omitted, the corresponding criteria are ignored.

    Args:
        rp (torch.Tensor): Portfolio returns matrix
        rb (torch.Tensor): Benchmark returns matrix
        rf (torch.Tensor): Scalar risk-free rate
    """
    if rp is not None:
        if rp.ndim != 2:
            raise ValueError("rp must be a 2-D tensor")
        if not torch.is_floating_point(rp):
            raise ValueError("rp must be a floating dtype")
    if rb is not None:
        if rb.ndim != 2:
            raise ValueError("rb must be a 2-D tensor")
        if not torch.is_floating_point(rb):
            raise ValueError("rb must be a floating dtype")
    if rf is not None:
        if rf.ndim != 0:
            raise ValueError("rf must be a 0-D tensor")
        if not torch.is_floating_point(rf):
            raise ValueError("rf must be a floating dtype")
    if (rp is not None) and (rb is not None) and rp.shape[1] != rb.shape[1]:
        raise ValueError("dimension 1 of rp and rb must be equal")


def beta(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the betas for all pairs of p portfolios and b benchmarks

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rb (torch.Tensor): b-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th benchmark

    Returns:
        torch.Tensor: p-by-b matrix where the (i, j) entry corresponds to the
        beta for the i-th portfolio and j-th benchmark
    """
    __expect_rp_rb_rf(rp, rb, None)
    return kernels.beta(rp, rb)


def alpha(rp: torch.Tensor, rb: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Returns the alphas for all pairs of p portfolios and b benchmarks

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rb (torch.Tensor): b-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th benchmark
        rf (torch.Tensor): Scalar risk-free rate (as a 0-D tensor)

    Returns:
        torch.Tensor: p-by-b matrix where the (i, j) entry corresponds to the
        alpha for the i-th portfolio and j-th benchmark
    """
    __expect_rp_rb_rf(rp, rb, rf)
    return kernels.alpha(rp, rb, rf)


def sharpe(rp: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Returns the sharpe ratios for p portfolios

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rf (torch.Tensor): Scalar risk-free rate (as a 0-D tensor)

    Returns:
        torch.Tensor: p-by-1 column vector where the i-th entry corresponds to
        the sharpe ratio for the i-th portfolio
    """
    __expect_rp_rb_rf(rp, None, rf)
    return kernels.sharpe(rp, rf)


def treynor(rp: torch.Tensor, rb: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Returns the treynor ratios for all pairs of p portfolios and b benchmarks

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rb (torch.Tensor): b-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th benchmark
        rf (torch.Tensor): Scalar risk-free rate (as a 0-D tensor)

    Returns:
        torch.Tensor: p-by-b matrix where the (i, j) entry corresponds to the
        treynor ratio for the i-th portfolio and j-th benchmark
    """
    __expect_rp_rb_rf(rp, rb, rf)
    return kernels.treynor(rp, rb, rf)


def sortino(rp: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    """Returns the sortino ratios for p portfolios

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rf (torch.Tensor): Scalar risk-free rate (as a 0-D tensor)

    Returns:
        torch.Tensor: p-by-1 column vector where the i-th entry corresponds to
        the sortino ratio for the i-th portfolio
    """
    __expect_rp_rb_rf(rp, None, rf)
    return kernels.sortino(rp, rf)


def tracking_error(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the tracking errors for all pairs of p portfolios and b
    benchmarks

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rb (torch.Tensor): b-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th benchmark

    Returns:
        torch.Tensor: p-by-b matrix where the (i, j) entry corresponds to the
        tracking error for the i-th portfolio and j-th benchmark
    """
    __expect_rp_rb_rf(rp, rb, None)
    return kernels.tracking_error(rp, rb)


def information(rp: torch.Tensor, rb: torch.Tensor) -> float:
    """Returns the information ratios for all pairs of p portfolios and b
    benchmarks

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rb (torch.Tensor): b-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th benchmark

    Returns:
        torch.Tensor: p-by-b matrix where the (i, j) entry corresponds to the
        information ratio for the i-th portfolio and j-th benchmark
    """
    __expect_rp_rb_rf(rp, rb, None)
    return kernels.information(rp, rb)


def up_capture(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the up-market capture ratios for all pairs of p portfolios and b
    benchmarks

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rb (torch.Tensor): b-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th benchmark

    Returns:
        torch.Tensor: p-by-b matrix where the (i, j) entry corresponds to the
        up-market capture ratio for the i-th portfolio and j-th benchmark
    """
    __expect_rp_rb_rf(rp, rb, None)
    return kernels.up_capture(rp, rb)


def down_capture(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the down-market capture ratios for all pairs of p portfolios and
    b benchmarks

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rb (torch.Tensor): b-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th benchmark

    Returns:
        torch.Tensor: p-by-b matrix where the (i, j) entry corresponds to the
        down-market capture ratio for the i-th portfolio and j-th benchmark
    """
    __expect_rp_rb_rf(rp, rb, None)
    return kernels.down_capture(rp, rb)


def capture(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    """Returns the capture ratios for all pairs of p portfolios and b benchmarks

    Args:
        rp (torch.Tensor): p-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th portfolio
        rb (torch.Tensor): b-by-n matrix where the (i, j) entry corresponds to
          the j-th return of the i-th benchmark

    Returns:
        torch.Tensor: p-by-b matrix where the (i, j) entry corresponds to the
        capture ratio for the i-th portfolio and j-th benchmark
    """
    __expect_rp_rb_rf(rp, rb, None)
    return kernels.capture(rp, rb)
