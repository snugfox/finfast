import numpy as np
from numpy.lib.arraysetops import isin
import torch
import pytest
import dataclasses

from finfast.analyze import metrics as metrics_numpy
from finfast_torch.analyze import metrics as metrics_torch

from typing import Generic, Optional, TypeVar

TensorT = TypeVar("TensorT")


@dataclasses.dataclass
class MetricsTestdata(Generic[TensorT]):
    rp: TensorT
    want: TensorT
    rb: Optional[TensorT] = None
    rf: Optional[TensorT] = None

    def as_numpy(self) -> "MetricsTestdata[np.ndarray]":
        return MetricsTestdata[np.ndarray](
            rp=self.rp.numpy(),
            want=self.want.numpy(),
            rb=(self.rb.numpy() if self.rb is not None else None),
            rf=(self.rf.numpy() if self.rf is not None else None),
        )

    def as_torch(self) -> "MetricsTestdata[torch.Tensor]":
        return MetricsTestdata[torch.Tensor](
            rp=torch.from_numpy(self.rp),
            want=torch.from_numpy(self.want),
            rb=(torch.from_numpy(self.rb) if self.rb is not None else None),
            rf=(torch.from_numpy(self.rf) if self.rf is not None else None),
        )


def beta_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [0.03, 0.05, -0.05, 0.09],
                [-0.005, 0.0, -0.025, 0.01],
            ],
            dtype=np.float64,
        ),
        rb=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [-0.01, -0.02, 0.03, -0.04],
            ],
            dtype=np.float64,
        ),
        want=np.asarray(
            [
                [1.0, -1.0],
                [2.0, -2.0],
                [0.5, -0.5],
            ],
            dtype=np.float64,
        ),
    )


def alpha_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [0.03, 0.05, -0.05, 0.09],
                [-0.005, 0.0, -0.025, 0.01],
            ],
            dtype=np.float64,
        ),
        rb=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [-0.01, -0.02, 0.03, -0.04],
            ],
            dtype=np.float64,
        ),
        rf=np.asarray(0.01, dtype=np.float64),
        want=np.asarray(
            [
                [0.0, -0.02],
                [0.02, -0.02],
                [-0.015, -0.025],
            ],
            dtype=np.float64,
        ),
    )


def sharpe_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [0.03, 0.05, -0.05, 0.09],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        rf=np.asarray(0.01, dtype=np.float64),
        want=np.asarray(
            [
                [0.0],
                [0.39223227027636803],
                [-np.inf],
            ],
            dtype=np.float64,
        ),
    )


def treynor_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [0.03, 0.05, -0.05, 0.09],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        rb=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [-0.01, -0.02, 0.03, -0.04],
            ],
            dtype=np.float64,
        ),
        rf=np.asarray(0.01, dtype=np.float64),
        want=np.asarray(
            [
                [0.0, 0.0],
                [0.01, -0.01],
                [-np.inf, -np.inf],
            ],
            dtype=np.float64,
        ),
    )


def sortino_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, -0.02, -0.03, 0.04],
                [0.03, 0.05, -0.05, 0.09],
                [0.0, 0.01, 0.0, 0.01],
            ],
            dtype=np.float64,
        ),
        rf=np.asarray(0.01, dtype=np.float64),
        want=np.asarray(
            [
                [-0.769800358919501],
                [0.923760430703401],
                [-np.inf],
            ],
            dtype=np.float64,
        ),
    )


def tracking_error_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [0.02, 0.03, -0.02, 0.05],
            ],
            dtype=np.float64,
        ),
        rb=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [-0.01, -0.02, 0.03, -0.04],
            ],
            dtype=np.float64,
        ),
        want=np.asarray(
            [
                [0.0, 0.05099019513592785],
                [0.0, 0.05099019513592785],
            ],
            dtype=np.float64,
        ),
    )


def information_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        rb=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [-0.01, -0.02, 0.03, -0.04],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        want=np.asarray(
            [
                [0.0, 0.3922322702763681, 0.3922322702763681],
                [-0.3922322702763681, 0.3922322702763681, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def up_capture_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [-0.02, -0.04, 0.06, -0.08],
            ],
            dtype=np.float64,
        ),
        rb=np.asarray(
            [
                [0.01, 0.02, -0.05, 0.04],
                [0.0, -0.01, 0.0, -0.01],
            ],
            dtype=np.float64,
        ),
        want=np.asarray(
            [
                [1.0, np.nan],
                [-2.0, np.nan],
            ],
            dtype=np.float64,
        ),
    )


def down_capture_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [-0.02, -0.04, 0.06, -0.08],
            ],
            dtype=np.float64,
        ),
        rb=np.asarray(
            [
                [0.03, 0.04, -0.03, 0.06],
                [0.0, 0.01, 0.0, 0.01],
            ],
            dtype=np.float64,
        ),
        want=np.asarray(
            [
                [1.0, np.nan],
                [-2.0, np.nan],
            ],
            dtype=np.float64,
        ),
    )


def capture_testdata() -> MetricsTestdata[np.ndarray]:
    return MetricsTestdata[np.ndarray](
        rp=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [-0.02, -0.04, 0.06, -0.08],
            ],
            dtype=np.float64,
        ),
        rb=np.asarray(
            [
                [0.01, 0.02, -0.03, 0.04],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        want=np.asarray(
            [
                [1.0, np.nan],
                [-2.0, -np.nan],
            ],
            dtype=np.float64,
        ),
    )


class TestMetricsNumpy:
    def test_beta(self) -> None:
        td = beta_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.beta(td.rp, td.rb)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want)

    def test_alpha(self) -> None:
        td = alpha_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.alpha(td.rp, td.rb, td.rf)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want)

    def test_sharpe(self) -> None:
        td = sharpe_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.sharpe(td.rp, td.rf)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want)

    def test_treynor(self) -> None:
        td = treynor_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.treynor(td.rp, td.rb, td.rf)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want)

    def test_sortino(self) -> None:
        td = sortino_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.sortino(td.rp, td.rf)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want)

    def test_tracking_error(self) -> None:
        td = tracking_error_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.tracking_error(td.rp, td.rb)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want, atol=1e-12)

    def test_information(self) -> None:
        td = information_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.information(td.rp, td.rb)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want)

    def test_up_capture(self) -> None:
        td = up_capture_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.up_capture(td.rp, td.rb)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want)

    def test_down_capture(self) -> None:
        td = down_capture_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.down_capture(td.rp, td.rb)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want)

    def test_capture(self) -> None:
        td = capture_testdata()
        with np.testing.assert_no_warnings():
            got = metrics_numpy.capture(td.rp, td.rb)
            assert got.dtype == td.want.dtype
            assert got.shape == td.want.shape
            np.testing.assert_allclose(got, td.want)


class TestMetricsTorch:
    def test_beta(self) -> None:
        td = beta_testdata().as_torch()
        got = metrics_torch.beta(td.rp, td.rb)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        torch.testing.assert_allclose(got, td.want)

    def test_alpha(self) -> None:
        td = alpha_testdata().as_torch()
        got = metrics_torch.alpha(td.rp, td.rb, td.rf)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        np.testing.assert_allclose(got, td.want)

    def test_sharpe(self) -> None:
        td = sharpe_testdata().as_torch()
        got = metrics_torch.sharpe(td.rp, td.rf)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        torch.testing.assert_allclose(got, td.want)

    def test_treynor(self) -> None:
        td = treynor_testdata().as_torch()
        got = metrics_torch.treynor(td.rp, td.rb, td.rf)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        torch.testing.assert_allclose(got, td.want)

    def test_sortino(self) -> None:
        td = sortino_testdata().as_torch()
        got = metrics_torch.sortino(td.rp, td.rf)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        torch.testing.assert_allclose(got, td.want)

    def test_tracking_error(self) -> None:
        td = tracking_error_testdata().as_torch()
        got = metrics_torch.tracking_error(td.rp, td.rb)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        torch.testing.assert_allclose(got, td.want)

    def test_information(self) -> None:
        td = information_testdata().as_torch()
        got = metrics_torch.information(td.rp, td.rb)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        torch.testing.assert_allclose(got, td.want)

    def test_up_capture(self) -> None:
        td = up_capture_testdata().as_torch()
        got = metrics_torch.up_capture(td.rp, td.rb)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        torch.testing.assert_allclose(got, td.want)

    def test_down_capture(self) -> None:
        td = down_capture_testdata().as_torch()
        got = metrics_torch.down_capture(td.rp, td.rb)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        torch.testing.assert_allclose(got, td.want)

    def test_capture(self) -> None:
        td = capture_testdata().as_torch()
        got = metrics_torch.capture(td.rp, td.rb)
        assert got.dtype == td.want.dtype
        assert got.shape == td.want.shape
        torch.testing.assert_allclose(got, td.want)


if __name__ == "__main__":
    pytest.main()
