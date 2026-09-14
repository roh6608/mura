"""Exact Gaussian process inference on a full grid via Kronecker structure."""

from collections.abc import Iterator, Sequence
from string import ascii_letters
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shared.core.backend import array_module

from bragg.core.constants import DERIVATIVE_POLYORDER, DERIVATIVE_WINDOW, GRAM_JITTER, MAD_TO_SIGMA
from bragg.gp import kernels


def mode_dot(x: NDArray[np.float64], m: NDArray[np.float64], axis: int) -> NDArray[np.float64]:
    """Contract 'x' along 'axis' with matrix 'm'."""
    xp = array_module(x)
    n = x.shape[axis]
    if m.shape[1] != n:
        raise ValueError(f"matrix {m.shape} cannot contract axis {axis} of length {n}")
    return xp.moveaxis(xp.tensordot(m, x, axes=(1, axis)), 0, axis)


def kron_mv(x: NDArray[np.float64], mats: Sequence[NDArray[np.float64] | None]) -> NDArray[np.float64]:
    """Apply the per-axis matrices to a cube."""
    if len(mats) > x.ndim:
        raise ValueError("more matrices than cube axes")
    active_axes = tuple(axis for axis, matrix in enumerate(mats) if matrix is not None)
    if not active_axes:
        return x

    input_axes = ascii_letters[: x.ndim]
    new_axes = ascii_letters[x.ndim : 2 * x.ndim]
    if len(new_axes) != x.ndim:
        raise ValueError("too many cube axes for einsum")
    matrix_axes = (f"{new_axes[axis]}{input_axes[axis]}" for axis in active_axes)
    output_axes = "".join(new_axes[axis] if axis in active_axes else input_axes[axis] for axis in range(x.ndim))
    subscripts = f"{input_axes},{','.join(matrix_axes)}->{output_axes}"

    path = ["einsum_path", (0, 1), *((0, index) for index in range(len(active_axes) - 1, 0, -1))]
    return array_module(x).einsum(subscripts, x, *(mats[axis] for axis in active_axes), optimize=path)


def outer_product(vectors: Sequence[NDArray[np.float64]], xp: ModuleType = np) -> NDArray[np.float64]:
    """Eigenvalues of a Kronecker product: 'L[i,j,k] = L0[i] L1[j] L2[k]'."""
    if not vectors:
        raise ValueError("at least one vector is required")
    axes = ascii_letters[: len(vectors)]
    if len(axes) != len(vectors):
        raise ValueError("too many vectors for einsum")
    subscripts = f"{','.join(axes)}->{axes}"
    return xp.einsum(subscripts, *(xp.asarray(v, dtype=xp.float64) for v in vectors))


def noise_from_differences(y: NDArray[np.float64], axis: int = 0) -> float:
    """Robust scalar noise from the MAD of first differences along one axis."""
    xp = array_module(y)
    d = xp.diff(y, axis=axis)
    d = d[xp.isfinite(d)]
    if d.size == 0:
        return float("nan")
    mad = float(xp.median(xp.abs(d - xp.median(d))))
    return mad * MAD_TO_SIGMA / np.sqrt(2.0)


class KroneckerGP:
    """Exact GP on a full grid with a separable (product) prior."""

    def __init__(  # noqa: PLR0913
        self,
        coords: tuple[NDArray[np.float64], ...],
        length_scales: tuple[float, ...],
        kinds: tuple[str, ...],
        *,
        sigma_f: float = 1.0,
        sigma_n: float = 1.0,
        jitter: float = GRAM_JITTER,
        xp: ModuleType = np,
        work_dtype: type[np.floating[Any]] = np.float64,
    ) -> None:
        """Check that every per-axis sequence has one entry per axis, then eigendecompose each axis kernel."""
        n = len(coords)
        if len(length_scales) != n or len(kinds) != n:
            raise ValueError("coords, length_scales and kinds must have one entry per axis")
        self.coords = coords
        self.length_scales = length_scales
        self.kinds = kinds
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.jitter = jitter
        self.xp = xp
        self.work_dtype = work_dtype

        self.eigvals: list[NDArray[np.float64]] = []
        self.eigvecs: list[NDArray[np.float64]] = []
        for c, ls, kind in zip(coords, length_scales, kinds, strict=True):
            k = kernels.gram(c, ls, kind=kind, jitter=jitter, xp=xp)
            w, v = xp.linalg.eigh(k)
            self.eigvals.append(xp.clip(w, 0.0, None))  # clipping to positive
            self.eigvecs.append(v.astype(work_dtype))
        self.eigenvalue_spectrum: NDArray[np.float64] = outer_product(self.eigvals, xp=xp)

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape, one entry per axis."""
        return tuple(int(np.asarray(c).size) for c in self.coords)

    def _spectrum(self) -> NDArray[np.float64]:
        """'sigma_f^2 * Lambda + sigma_n^2' -- the eigenvalues of 'K + sn^2 I'."""
        return self.sigma_f**2 * self.eigenvalue_spectrum + self.sigma_n**2

    def _posterior_spectrum(self) -> NDArray[np.float64]:
        """Eigenvalues of the posterior covariance, 'sigma_f^2 Lambda sigma_n^2 / (sigma_f^2 Lambda + sigma_n^2)'."""
        return self.sigma_f**2 * self.eigenvalue_spectrum * self.sigma_n**2 / self._spectrum()

    def transform(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """'U^T y', rotate data into the eigenbasis."""
        xp = self.xp
        z = kron_mv(xp.asarray(y, dtype=self.work_dtype), [v.T for v in self.eigvecs])
        return z.astype(xp.float64)

    def untransform(
        self, z: NDArray[np.float64], mats: Sequence[NDArray[np.float64]] | None = None
    ) -> NDArray[np.float64]:
        """'U z', rotate back."""
        base = mats if mats is not None else self.eigvecs
        return kron_mv(self.xp.asarray(z, dtype=self.work_dtype), base)

    def nlml(self, y: NDArray[np.float64]) -> float:
        """Exact negative log marginal likelihood, every term a reduction over the eigenvalue cube."""
        xp = self.xp
        s = self._spectrum()
        yt = self.transform(y)
        n = float(yt.size)
        return 0.5 * float(xp.sum(yt * yt / s) + xp.sum(xp.log(s)) + n * np.log(2 * np.pi))

    def mean_coeff(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Eigen-coefficients of the posterior mean, 'sigma_f^2 L / (sigma_f^2 L + sn^2) * yt'."""
        return self.sigma_f**2 * self.eigenvalue_spectrum * (self.transform(y) / self._spectrum())

    def posterior_mean(self, y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the posterior mean cube, 'K (K + sn^2 I)^-1 y'."""
        return self.untransform(self.mean_coeff(y))

    def derivative_mats(self, axis: int, window: int, polyorder: int) -> list[NDArray[np.float64]]:
        """Return the eigenvector matrices with the derivative operator folded into 'axis'."""
        d = kernels.savgol_derivative_matrix(self.coords[axis], window, polyorder, xp=self.xp)
        mats = list(self.eigvecs)
        mats[axis] = d @ mats[axis]
        return mats

    def posterior_derivative(
        self,
        y: NDArray[np.float64],
        axis: int = 0,
        window: int = DERIVATIVE_WINDOW,
        polyorder: int = DERIVATIVE_POLYORDER,
    ) -> NDArray[np.float64]:
        """Posterior mean differentiated along one axis."""
        return self.untransform(self.mean_coeff(y), mats=self.derivative_mats(axis, window, polyorder))

    def posterior_var(self, mats: Sequence[NDArray[np.float64]] | None = None) -> NDArray[np.float64]:
        """Full posterior variance cube, at the same cost as the mean."""
        base = mats if mats is not None else self.eigvecs
        return kron_mv(self._posterior_spectrum(), [m * m for m in base])

    def sample(
        self, y: NDArray[np.float64], n_draws: int, seed: int, mats: Sequence[NDArray[np.float64]] | None = None
    ) -> Iterator[NDArray[np.float64]]:
        """Draw from the posterior: 'mean + U [ sqrt(s_post) xi ]', through 'mats' if given."""
        xp = self.xp
        root = xp.sqrt(self._posterior_spectrum())
        base = mats if mats is not None else self.eigvecs
        coeff = self.mean_coeff(y)
        rng = xp.random.default_rng(seed)
        for _ in range(n_draws):
            xi = rng.standard_normal(root.shape, dtype=root.dtype)
            yield self.untransform(coeff + root * xi, mats=base)
