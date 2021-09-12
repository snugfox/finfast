import argparse
import numpy as np
from os import path
import torch

from finfast.analyze import metrics_numpy, metrics_torch

from typing import Mapping, Optional


def rel_curr_dir(p: str) -> str:
    return path.join(path.dirname(__file__), p)


def gen_metrics():
    rng = np.random.default_rng(19930405)
    rb_opts = (4.60074e-4, 9.29934e-3)  # $SPX 2010-2019
    rp_opts = (
        (1.03186e-3, 2.63412e-4, 8.97308e-5),  # AAPL 2010-2019
        (4.52055e-4, 1.47086e-4, 1.02036e-4),  # IWM  2010-2019
        (5.93643e-4, 2.48883e-4, 1.12394e-4),  # JPM  2010-2019
        (1.33507e-3, 6.52786e-4, 1.31599e-4),  # NVDA 2010-2019
        (6.64560e-4, 1.18405e-4, 9.37182e-5),  # QQQ  2010-2019
        (-0.5 * rb_opts[0], 0.25 * rb_opts[1], -0.5 * rb_opts[1]),  # rp == -0.5 * rb
        (rb_opts[0], rb_opts[1], 0.0),  # Xp ~ Xb
        (0.5 * rb_opts[0], 0.25 * rb_opts[1], 0.5 * rb_opts[1]),  # rp == 0.5 * rb
        (rb_opts[0], rb_opts[1], rb_opts[1]),  # rp == 1.0 * rb
        (2.0 * rb_opts[0], 4.0 * rb_opts[1], 2.0 * rb_opts[1]),  # rp == 2.0 * rb
    )
    rf = np.array([[0.0], [0.02 / 253.0]])

    # Sample benchmark returns from a normal distribution
    rb = rng.normal(loc=rb_opts[0], scale=np.sqrt(rb_opts[1]), size=(len(rp_opts), 253))
    rb = np.maximum(rb, -1.0)

    # Sample portfolio returns from several linear combinations of normal distributions
    rp = np.empty_like(rb)
    for mi, (rp_mean, rp_var, rp_rb_cov) in enumerate(rp_opts):
        rp_rb_beta = rp_rb_cov / rb_opts[1]
        diff_mean = rp_mean - rp_rb_beta * rb_opts[0]
        diff_var = (
            rp_var + np.square(rp_rb_beta) * rb_opts[1] - 2 * rp_rb_beta * rp_rb_cov
        )
        rp_m = rp_rb_beta * rb[mi, :] + rng.normal(
            loc=diff_mean, scale=np.sqrt(diff_var), size=(253,)
        )
        rp_m = np.maximum(rp_m, -1.0)
        rp[mi, :] = rp_m

    # Repeat each portfolio and benchmark returns vector for each risk-free rate
    rp = np.repeat(rp, rf.shape[0], axis=0)
    rb = np.repeat(rb, rf.shape[0], axis=0)
    rf = np.tile(rf, (len(rp_opts), 1))

    np.savez(rel_curr_dir("metrics.npz"), rp=rp, rb=rb, rf=rf)


def calc_metrics():
    rp: Optional[np.ndarray] = None
    rb: Optional[np.ndarray] = None
    rf: Optional[np.ndarray] = None
    with np.load(rel_curr_dir("metrics.npz")) as testdata:
        testdata: Mapping[str, np.ndarray]
        rp = testdata["rp"]
        rb = testdata["rb"]
        rf = testdata["rf"]

    expected_metrics: dict[str, np.ndarray] = {}
    for dtype, label in [(np.float64, "f64"), (np.float32, "f32")]:
        rp_, rb_, rf_ = rp.astype(dtype), rb.astype(dtype), rf.astype(dtype)
        expected_metrics[f"numpy/beta/{label}"] = metrics_numpy.beta(rp_, rb_)
        expected_metrics[f"numpy/alpha/{label}"] = metrics_numpy.alpha(rp_, rb_, rf_)
        expected_metrics[f"numpy/sharpe/{label}"] = metrics_numpy.sharpe(rp_, rf_)
        expected_metrics[f"numpy/treynor/{label}"] = metrics_numpy.treynor(
            rp_, rb_, rf_
        )
        expected_metrics[f"numpy/sortino/{label}"] = metrics_numpy.sortino(rp_, rf_)
        expected_metrics[f"numpy/information/{label}"] = metrics_numpy.information(
            rp_, rb_
        )
        expected_metrics[f"numpy/up_capture/{label}"] = metrics_numpy.up_capture(
            rp_, rb_
        )
        expected_metrics[f"numpy/down_capture/{label}"] = metrics_numpy.down_capture(
            rp_, rb_
        )
        expected_metrics[f"numpy/capture/{label}"] = metrics_numpy.capture(rp_, rb_)
        expected_metrics[
            f"numpy/tracking_error/{label}"
        ] = metrics_numpy.tracking_error(rp_, rb_)

    for dtype, label in [(torch.float64, "f64"), (torch.float32, "f32")]:
        rp_ = torch.from_numpy(rp).to(dtype)
        rb_ = torch.from_numpy(rb).to(dtype)
        rf_ = torch.from_numpy(rf).to(dtype)
        expected_metrics[f"torch/beta/{label}"] = metrics_torch.beta(rp_, rb_).numpy()
        expected_metrics[f"torch/alpha/{label}"] = metrics_torch.alpha(
            rp_, rb_, rf_
        ).numpy()
        expected_metrics[f"torch/sharpe/{label}"] = metrics_torch.sharpe(
            rp_, rf_
        ).numpy()
        expected_metrics[f"torch/treynor/{label}"] = metrics_torch.treynor(
            rp_, rb_, rf_
        ).numpy()
        expected_metrics[f"torch/sortino/{label}"] = metrics_torch.sortino(
            rp_, rf_
        ).numpy()
        expected_metrics[f"torch/information/{label}"] = metrics_torch.information(
            rp_, rb_
        ).numpy()
        expected_metrics[f"torch/up_capture/{label}"] = metrics_torch.up_capture(
            rp_, rb_
        ).numpy()
        expected_metrics[f"torch/down_capture/{label}"] = metrics_torch.down_capture(
            rp_, rb_
        ).numpy()
        expected_metrics[f"torch/capture/{label}"] = metrics_torch.capture(
            rp_, rb_
        ).numpy()
        expected_metrics[
            f"torch/tracking_error/{label}"
        ] = metrics_torch.tracking_error(rp_, rb_).numpy()

    np.savez(rel_curr_dir("metrics_results.npz"), **expected_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen",
        nargs="*",
        type=str,
        choices=["metrics"],
        help="Generate new test data",
    )
    parser.add_argument(
        "--calc",
        nargs="*",
        type=str,
        choices=["metrics"],
        help="Calculate expected outputs",
    )
    args = parser.parse_args()

    if args.gen is not None:
        for gen_arg in args.gen:
            if gen_arg == "metrics":
                gen_metrics()

    if args.calc is not None:
        for calc_arg in args.calc:
            if calc_arg == "metrics":
                calc_metrics()
