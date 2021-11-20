import torch


def beta(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    rp_cent = rp - torch.mean(rp, dim=1, keepdim=True)
    rb_cent = rb - torch.mean(rb, dim=1, keepdim=True)
    rb_var = torch.mean(torch.square(rb_cent), dim=1, keepdim=True)
    cov = (rp_cent @ rb_cent.T) / rp.shape[1]
    return cov / rb_var.T


def alpha(rp: torch.Tensor, rb: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    return (
        (rf - torch.mean(rb, dim=1, keepdim=True)).T * beta(rp, rb)
        + torch.mean(rp, dim=1, keepdim=True)
        - rf
    )


def sharpe(rp: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    rp_std, rp_mean = torch.std_mean(rp, dim=1, unbiased=False, keepdim=True)
    return (rp_mean - rf) / rp_std


def treynor(rp: torch.Tensor, rb: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    return (torch.mean(rp, dim=1, keepdim=True) - rf) / beta(rp, rb)


def sortino(rp: torch.Tensor, rf: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=rp.dtype, device=rp.device)
    return (torch.mean(rp, dim=1, keepdim=True) - rf) / torch.std(
        torch.minimum(rp, zero), dim=1, unbiased=False, keepdim=True
    )


def tracking_error(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    rp_expanded = torch.unsqueeze(rp, 1)
    rb_expanded = torch.unsqueeze(rb, 0)
    return torch.std(rp_expanded - rb_expanded, dim=2, unbiased=False)


def information(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(rp.dtype).tiny
    return (
        torch.mean(rp, dim=1, keepdim=True) - torch.mean(rb, dim=1, keepdim=True).T
    ) / (tracking_error(rp, rb) + eps)


def up_capture(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    rp_expanded = torch.unsqueeze(rp, 1)
    rb_expanded = torch.unsqueeze(rb, 0)
    up_mask = rb_expanded > 0
    return torch.sum((up_mask * rp_expanded) / rb_expanded, dim=2) / (
        torch.count_nonzero(up_mask, dim=2)
    )


def down_capture(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    rp_expanded = torch.unsqueeze(rp, 1)
    rb_expanded = torch.unsqueeze(rb, 0)
    down_mask = rb_expanded < 0
    return torch.sum((down_mask * rp_expanded) / rb_expanded, dim=2) / (
        torch.count_nonzero(down_mask, dim=2)
    )


def capture(rp: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    rp_expanded = torch.unsqueeze(rp, 1)
    rb_expanded = torch.unsqueeze(rb, 0)
    return torch.mean(rp_expanded / rb_expanded, dim=2)
