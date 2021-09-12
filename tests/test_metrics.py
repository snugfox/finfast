import numpy as np
from os import path
import torch
import pytest

from finfast.analyze import metrics_numpy, metrics_torch

from typing import Any, Mapping, NamedTuple, Optional, Type, Union


class NumpyTestdata(NamedTuple):
    dtype: Type[np.dtype]
    rp: np.ndarray
    rb: np.ndarray
    rf: np.ndarray
    results: dict[str, np.ndarray]


class TorchTestdata(NamedTuple):
    dtype: torch.dtype
    rp: torch.Tensor
    rb: torch.Tensor
    rf: torch.Tensor
    results: dict[str, torch.Tensor]


def dtype_to_str(dtype: Union[Type[np.dtype], torch.dtype]) -> str:
    if dtype == np.float64 or dtype == torch.float64:
        return "f64"
    elif dtype == np.float32 or dtype == torch.float32:
        return "f32"
    raise ValueError("unrecognized dtype")


@pytest.fixture(scope="class", params=[np.float64, np.float32], ids=["f64", "f32"])
def numpy_testdata(request: Any) -> NumpyTestdata:
    dtype: Type[np.dtype] = request.param
    rp: Optional[np.ndarray] = None
    rb: Optional[np.ndarray] = None
    rf: Optional[np.ndarray] = None
    results: dict[str, np.ndarray] = None

    testdata_filename = path.join(path.dirname(__file__), "test_data", "metrics.npz")
    testdata_results_filename = path.join(
        path.dirname(__file__), "test_data", "metrics_results.npz"
    )
    with np.load(testdata_filename) as testdata:
        testdata: Mapping[str, np.ndarray]
        rp = testdata["rp"].astype(dtype)
        rb = testdata["rb"].astype(dtype)
        rf = testdata["rf"].astype(dtype)
    with np.load(testdata_results_filename) as testdata_results:
        testdata_results: Mapping[str, np.ndarray]
        results: dict[str, np.ndarray] = {}
        for key, result in testdata_results.items():
            module_key, key = key.split("/", 1)
            if module_key == "numpy":
                metric_key, dtype_key = key.split("/")
                if dtype_key == dtype_to_str(dtype):
                    results[metric_key] = result
    return NumpyTestdata(dtype, rp, rb, rf, results)


@pytest.fixture(
    scope="class", params=[torch.float64, torch.float32], ids=["f64", "f32"]
)
def torch_testdata(request: Any) -> TorchTestdata:
    dtype: torch.dtype = request.param
    rp: Optional[torch.Tensor] = None
    rb: Optional[torch.Tensor] = None
    rf: Optional[torch.Tensor] = None
    results: dict[str, torch.Tensor] = None

    testdata_filename = path.join(path.dirname(__file__), "test_data", "metrics.npz")
    testdata_results_filename = path.join(
        path.dirname(__file__), "test_data", "metrics_results.npz"
    )
    with np.load(testdata_filename) as testdata:
        testdata: Mapping[str, torch.Tensor]
        rp = torch.from_numpy(testdata["rp"]).to(dtype)
        rb = torch.from_numpy(testdata["rb"]).to(dtype)
        rf = torch.from_numpy(testdata["rf"]).to(dtype)
    with np.load(testdata_results_filename) as testdata_results:
        testdata_results: Mapping[str, torch.Tensor]
        results: dict[str, torch.Tensor] = {}
        for key, result in testdata_results.items():
            module_key, key = key.split("/", 1)
            if module_key == "torch":
                metric_key, dtype_key = key.split("/")
                if dtype_key == dtype_to_str(dtype):
                    results[metric_key] = result
    return TorchTestdata(dtype, rp, rb, rf, results)


class TestMetricsNumpy:
    def test_beta(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["beta"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.beta(
                numpy_testdata.rp,
                numpy_testdata.rb,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)

    def test_alpha(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["alpha"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.alpha(
                numpy_testdata.rp,
                numpy_testdata.rb,
                numpy_testdata.rf,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)

    def test_sharpe(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["sharpe"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.sharpe(
                numpy_testdata.rp,
                numpy_testdata.rf,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)

    def test_treynor(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["treynor"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.treynor(
                numpy_testdata.rp,
                numpy_testdata.rb,
                numpy_testdata.rf,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)

    def test_sortino(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["sortino"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.sortino(
                numpy_testdata.rp,
                numpy_testdata.rf,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)

    def test_information(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["information"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.information(
                numpy_testdata.rp,
                numpy_testdata.rb,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)

    def test_up_capture(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["up_capture"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.up_capture(
                numpy_testdata.rp,
                numpy_testdata.rb,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)

    def test_down_capture(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["down_capture"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.down_capture(
                numpy_testdata.rp,
                numpy_testdata.rb,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)

    def test_capture(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["capture"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.capture(
                numpy_testdata.rp,
                numpy_testdata.rb,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)

    def test_tracking_error(_, numpy_testdata: NumpyTestdata) -> None:
        want = numpy_testdata.results["tracking_error"]
        want_dtype = np.dtype(numpy_testdata.dtype)
        with np.testing.assert_no_warnings():
            got = metrics_numpy.tracking_error(
                numpy_testdata.rp,
                numpy_testdata.rb,
            )
            assert got.dtype == want_dtype
            assert got.flags.c_contiguous
            np.testing.assert_allclose(got, want)


class TestMetricsTorch:
    def test_beta(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["beta"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.beta(
            torch_testdata.rp,
            torch_testdata.rb,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)

    def test_alpha(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["alpha"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.alpha(
            torch_testdata.rp,
            torch_testdata.rb,
            torch_testdata.rf,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)

    def test_sharpe(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["sharpe"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.sharpe(
            torch_testdata.rp,
            torch_testdata.rf,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)

    def test_treynor(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["treynor"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.treynor(
            torch_testdata.rp,
            torch_testdata.rb,
            torch_testdata.rf,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)

    def test_sortino(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["sortino"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.sortino(
            torch_testdata.rp,
            torch_testdata.rf,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)

    def test_information(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["information"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.information(
            torch_testdata.rp,
            torch_testdata.rb,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)

    def test_up_capture(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["up_capture"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.up_capture(
            torch_testdata.rp,
            torch_testdata.rb,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)

    def test_down_capture(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["down_capture"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.down_capture(
            torch_testdata.rp,
            torch_testdata.rb,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)

    def test_capture(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["capture"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.capture(
            torch_testdata.rp,
            torch_testdata.rb,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)

    def test_tracking_error(_, torch_testdata: TorchTestdata) -> None:
        want = torch_testdata.results["tracking_error"]
        want_dtype = torch_testdata.dtype
        got = metrics_torch.tracking_error(
            torch_testdata.rp,
            torch_testdata.rb,
        )
        assert got.dtype == want_dtype
        assert got.is_contiguous()
        torch.testing.assert_allclose(got, want)


if __name__ == "__main__":
    pytest.main()
