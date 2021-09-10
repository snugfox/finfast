import argparse
import numpy as np
from os import path

from finfast.analyze import metrics_numpy as metrics

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
    expected_metrics["beta"] = metrics.beta(rp, rb)
    expected_metrics["alpha"] = metrics.alpha(rp, rb, rf)
    expected_metrics["sharpe"] = metrics.sharpe(rp, rf)
    expected_metrics["treynor"] = metrics.treynor(rp, rb, rf)
    expected_metrics["sortino"] = metrics.sortino(rp, rf)
    expected_metrics["information"] = metrics.information(rp, rb)
    expected_metrics["up_capture"] = metrics.up_capture(rp, rb)
    expected_metrics["down_capture"] = metrics.down_capture(rp, rb)
    expected_metrics["capture"] = metrics.capture(rp, rb)
    expected_metrics["tracking_error"] = metrics.tracking_error(rp, rb)

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
