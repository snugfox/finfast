import unittest
import numpy as np
from finfast.analyze import metrics_numpy, metrics_torch
import torch
from os import path

from typing import Mapping


class TestMetricsNumpy(unittest.TestCase):
    def setUp(self) -> None:
        testdata_filename = path.join(
            path.dirname(__file__), "test_data", "metrics.npz"
        )
        testdata_results_filename = path.join(
            path.dirname(__file__), "test_data", "metrics_results.npz"
        )
        with np.load(testdata_filename) as testdata:
            testdata: Mapping[str, np.ndarray]
            self.rp = testdata["rp"]
            self.rb = testdata["rb"]
            self.rf = testdata["rf"]
        with np.load(testdata_results_filename) as testdata_results:
            testdata_results: Mapping[str, np.ndarray]
            self.results = dict(testdata_results)

    def test_beta(self) -> None:
        want = self.results["beta"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.beta(
                    self.rp.astype(want_dtype),
                    self.rb.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )

    def test_alpha(self) -> None:
        want = self.results["alpha"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.alpha(
                    self.rp.astype(want_dtype),
                    self.rb.astype(want_dtype),
                    self.rf.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )

    def test_sharpe(self) -> None:
        want = self.results["sharpe"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.sharpe(
                    self.rp.astype(want_dtype),
                    self.rf.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )

    def test_treynor(self) -> None:
        want = self.results["treynor"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.treynor(
                    self.rp.astype(want_dtype),
                    self.rb.astype(want_dtype),
                    self.rf.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )

    def test_sortino(self) -> None:
        want = self.results["sortino"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.sortino(
                    self.rp.astype(want_dtype),
                    self.rf.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )

    def test_information(self) -> None:
        want = self.results["information"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.information(
                    self.rp.astype(want_dtype),
                    self.rb.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )

    def test_up_capture(self) -> None:
        want = self.results["up_capture"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.up_capture(
                    self.rp.astype(want_dtype),
                    self.rb.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )

    def test_down_capture(self) -> None:
        want = self.results["down_capture"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.down_capture(
                    self.rp.astype(want_dtype),
                    self.rb.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )

    def test_capture(self) -> None:
        want = self.results["capture"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.capture(
                    self.rp.astype(want_dtype),
                    self.rb.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )

    def test_tracking_error(self) -> None:
        want = self.results["tracking_error"]
        test_cases = [
            (np.float64, 1e-6, 1e-9),
            (np.float32, 1e-4, 1e-7),
        ]
        for want_dtype, rtol, atol in test_cases:
            want_dtype = np.dtype(want_dtype)
            with self.subTest(dtype=want_dtype.name), np.testing.assert_no_warnings():
                got = metrics_numpy.tracking_error(
                    self.rp.astype(want_dtype),
                    self.rb.astype(want_dtype),
                )
                self.assertEqual(got.dtype, want_dtype)
                self.assertTrue(got.flags.c_contiguous)
                np.testing.assert_allclose(
                    got, want.astype(want_dtype), rtol=rtol, atol=atol
                )


if __name__ == "__main__":
    unittest.main()
