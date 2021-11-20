import numpy as np
import pytest
import torch

from finfast.portfolio import frontier as frontier_numpy
from finfast_torch.portfolio import frontier as frontier_torch


class TestFrontierNumpy:
    def test_lincomb(self) -> None:
        rp = np.asarray(
            [
                [0.1, 0.3, 0.0],
                [-0.5, 0.1, -0.3],
                [0.2, 0.1, 0.1],
            ],
            dtype=np.float64,
        )
        coeff = np.asarray(
            [
                [0.3, 0.5, 0.2],
                [0.5, 0.3, 0.2],
            ],
            dtype=np.float64,
        )
        want = np.asarray(
            [
                [-0.18, 0.16, -0.13],
                [-0.06, 0.2, -0.07],
            ],
            dtype=np.float64,
        )
        got = frontier_numpy.lincomb(rp, coeff)
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        np.testing.assert_allclose(got, want)


    def test_lincomb_norm(self) -> None:
        mean = np.asarray([-0.1, 0.0, 0.1], dtype=np.float64)
        cov = np.asarray(
            [
                [0.1, -0.05, -0.025],
                [-0.05, 0.2, 0.05],
                [-0.025, 0.05, 0.3],
            ],
            dtype=np.float64,
        )
        coeff = np.asarray(
            [
                [0.3, 0.5, 0.2],
                [0.5, 0.3, 0.2],
            ],
            dtype=np.float64,
        )
        want = frontier_numpy.LinearCombinationNormResult(
            mean=np.asarray([-0.01, -0.03], dtype=np.float64),
            var=np.asarray([0.063, 0.041], dtype=np.float64),
        )

        got = frontier_numpy.lincomb_norm(mean, cov, coeff)
        assert got.mean.dtype == want.mean.dtype
        assert got.mean.shape == want.mean.shape
        np.testing.assert_allclose(got.mean, want.mean)
        assert got.var.dtype == want.var.dtype
        assert got.var.shape == want.var.shape
        np.testing.assert_allclose(got.var, want.var)

    def test_lincomb_norm_cov(self) -> None:
        mean = np.asarray([-0.1, 0.0, 0.1], dtype=np.float64)
        cov = np.asarray(
            [
                [0.1, -0.05, -0.025],
                [-0.05, 0.2, 0.05],
                [-0.025, 0.05, 0.3],
            ],
            dtype=np.float64,
        )
        coeff = np.asarray(
            [
                [0.3, 0.5, 0.2],
                [0.5, 0.3, 0.2],
            ],
            dtype=np.float64,
        )
        want = frontier_numpy.LinearCombinationNormCovResult(
            mean=np.asarray([-0.01, -0.03], dtype=np.float64),
            cov=np.asarray(
                [
                    [0.063, 0.044],
                    [0.044, 0.041],
                ],
                dtype=np.float64,
            ),
        )

        got = frontier_numpy.lincomb_norm_cov(mean, cov, coeff)
        assert got.mean.dtype == want.mean.dtype
        assert got.mean.shape == want.mean.shape
        np.testing.assert_allclose(got.mean, want.mean)
        assert got.cov.dtype == want.cov.dtype
        assert got.cov.shape == want.cov.shape
        np.testing.assert_allclose(got.cov, want.cov)


class TestFrontierTorch:
    def test_lincomb(self) -> None:
        rp = torch.tensor(
            [
                [0.1, 0.3, 0.0],
                [-0.5, 0.1, -0.3],
                [0.2, 0.1, 0.1],
            ],
            dtype=torch.float64,
        )
        coeff = torch.tensor(
            [
                [0.3, 0.5, 0.2],
                [0.5, 0.3, 0.2],
            ],
            dtype=torch.float64,
        )
        want = torch.tensor(
            [
                [-0.18, 0.16, -0.13],
                [-0.06, 0.2, -0.07],
            ],
            dtype=torch.float64,
        )
        got = frontier_torch.lincomb(rp, coeff)
        assert got.dtype == want.dtype
        assert got.shape == want.shape
        torch.testing.assert_allclose(got, want)

    def test_lincomb_norm(self) -> None:
        mean = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float64)
        cov = torch.tensor(
            [
                [0.1, -0.05, -0.025],
                [-0.05, 0.2, 0.05],
                [-0.025, 0.05, 0.3],
            ],
            dtype=torch.float64,
        )
        coeff = torch.tensor(
            [
                [0.3, 0.5, 0.2],
                [0.5, 0.3, 0.2],
            ],
            dtype=torch.float64,
        )
        want = frontier_torch.LinearCombinationNormResult(
            mean=torch.tensor([-0.01, -0.03], dtype=torch.float64),
            var=torch.tensor([0.063, 0.041], dtype=torch.float64),
        )

        got = frontier_torch.lincomb_norm(mean, cov, coeff)
        assert got.mean.dtype == want.mean.dtype
        assert got.mean.shape == want.mean.shape
        torch.testing.assert_allclose(got.mean, want.mean)
        assert got.var.dtype == want.var.dtype
        assert got.var.shape == want.var.shape
        torch.testing.assert_allclose(got.var, want.var)

    def test_lincomb_norm_cov(self) -> None:
        mean = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float64)
        cov = torch.tensor(
            [
                [0.1, -0.05, -0.025],
                [-0.05, 0.2, 0.05],
                [-0.025, 0.05, 0.3],
            ],
            dtype=torch.float64,
        )
        coeff = torch.tensor(
            [
                [0.3, 0.5, 0.2],
                [0.5, 0.3, 0.2],
            ],
            dtype=torch.float64,
        )
        want = frontier_torch.LinearCombinationNormCovResult(
            mean=torch.tensor([-0.01, -0.03], dtype=torch.float64),
            cov=torch.tensor(
                [
                    [0.063, 0.044],
                    [0.044, 0.041],
                ],
                dtype=torch.float64,
            ),
        )

        got = frontier_torch.lincomb_norm_cov(mean, cov, coeff)
        assert got.mean.dtype == want.mean.dtype
        assert got.mean.shape == want.mean.shape
        torch.testing.assert_allclose(got.mean, want.mean)
        assert got.cov.dtype == want.cov.dtype
        assert got.cov.shape == want.cov.shape
        torch.testing.assert_allclose(got.cov, want.cov)
