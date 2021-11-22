import numpy as np
import pytest
import unittest

from finfast.analyze import indicators

from typing import Literal, NamedTuple, Optional


class TestIndicatorsNumpy(unittest.TestCase):
    def test_delta(self) -> None:
        class TestCase(NamedTuple):
            name: str
            x: np.ndarray
            interval: int
            want: np.ndarray

        tests = [
            TestCase(
                name="Int0",
                x=np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float64),
                interval=0,
                want=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            ),
            TestCase(
                name="Int1",
                x=np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=np.float64),
                interval=1,
                want=np.array([np.nan, 0.1, 0.1, 0.1, 0.1], dtype=np.float64),
            ),
            TestCase(
                name="NaN",
                x=np.array([1.0, 1.1, np.nan, 1.3, 1.4], dtype=np.float64),
                interval=1,
                want=np.array([np.nan, 0.1, np.nan, np.nan, 0.1], dtype=np.float64),
            ),
        ]
        for tc in tests:
            with self.subTest(msg=tc.name):
                got = indicators.delta(tc.x, tc.interval)
                self.assertEqual(got.dtype, tc.want.dtype)
                self.assertEqual(got.shape, tc.want.shape)
                np.testing.assert_allclose(got, tc.want, equal_nan=True)

    def test_roc(self) -> None:
        class TestCase(NamedTuple):
            name: str
            x: np.ndarray
            interval: int
            method: Literal["div", "log"]
            want: np.ndarray

        tests = [
            TestCase(
                name="Int0-Div",
                x=np.array([1.0, 1.2, 0.9, 1.8, 0.9], dtype=np.float64),
                interval=0,
                method="div",
                want=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            ),
            TestCase(
                name="Int1-Div",
                x=np.array([1.0, 1.2, 0.9, 1.8, 0.9], dtype=np.float64),
                interval=1,
                method="div",
                want=np.array([np.nan, 0.2, -0.25, 1.0, -0.5], dtype=np.float64),
            ),
            TestCase(
                name="NaN-Div",
                x=np.array([1.0, 1.0, np.nan, 1.0, 1.0], dtype=np.float64),
                interval=1,
                method="div",
                want=np.array([np.nan, 0.0, np.nan, np.nan, 0.0], dtype=np.float64),
            ),
            TestCase(
                name="Int0-Log",
                x=np.array([1.0, 1.2, 0.9, 1.8, 0.9], dtype=np.float64),
                interval=0,
                method="log",
                want=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            ),
            TestCase(
                name="Int1-Log",
                x=np.array([1.0, 1.2, 0.9, 1.8, 0.9], dtype=np.float64),
                interval=1,
                method="log",
                want=np.array([np.nan, 0.2, -0.25, 1.0, -0.5], dtype=np.float64),
            ),
            TestCase(
                name="NaN-Log",
                x=np.array([1.0, 1.0, np.nan, 1.0, 1.0], dtype=np.float64),
                interval=1,
                method="log",
                want=np.array([np.nan, 0.0, np.nan, np.nan, 0.0], dtype=np.float64),
            ),
        ]
        for tc in tests:
            with self.subTest(msg=tc.name):
                got = indicators.roc(tc.x, tc.interval, tc.method)
                self.assertEqual(got.dtype, tc.want.dtype)
                self.assertEqual(got.shape, tc.want.shape)
                np.testing.assert_allclose(got, tc.want, equal_nan=True)

    def test_returns(self) -> None:
        class TestCase(NamedTuple):
            name: str
            x: np.ndarray
            method: Literal["div", "log"]
            want: np.ndarray

        tests = [
            TestCase(
                name="Div",
                x=np.array([1.0, 1.2, 0.9, 1.8, 0.9], dtype=np.float64),
                method="div",
                want=np.array([np.nan, 0.2, -0.25, 1.0, -0.5], dtype=np.float64),
            ),
            TestCase(
                name="NaN-Div",
                x=np.array([1.0, 1.0, np.nan, 1.0, 1.0], dtype=np.float64),
                method="div",
                want=np.array([np.nan, 0.0, np.nan, np.nan, 0.0], dtype=np.float64),
            ),
            TestCase(
                name="Log",
                x=np.array([1.0, 1.2, 0.9, 1.8, 0.9], dtype=np.float64),
                method="log",
                want=np.array([np.nan, 0.2, -0.25, 1.0, -0.5], dtype=np.float64),
            ),
            TestCase(
                name="NaN-Log",
                x=np.array([1.0, 1.0, np.nan, 1.0, 1.0], dtype=np.float64),
                method="log",
                want=np.array([np.nan, 0.0, np.nan, np.nan, 0.0], dtype=np.float64),
            ),
        ]
        for tc in tests:
            with self.subTest(msg=tc.name):
                got = indicators.returns(tc.x, tc.method)
                self.assertEqual(got.dtype, tc.want.dtype)
                self.assertEqual(got.shape, tc.want.shape)
                np.testing.assert_allclose(got, tc.want, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
