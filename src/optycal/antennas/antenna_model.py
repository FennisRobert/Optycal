from __future__ import annotations
from .antenna import Antenna
from ..geo.cs import CoordinateSystem, GCS
import io
from dataclasses import dataclass
from typing import Optional, Sequence, Union, BinaryIO
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import msgpack
import msgpack_numpy
from pathlib import Path

"""
FarfieldDataset
===============

A container class for the far-field dataset dict produced by ``farfield_model``,
providing:

  * properties describing the supported frequency range / sample points
  * multi-dimensional (freq, theta, phi) interpolation of the E- and H-fields
  * linear superposition over excitations (since each stored solution is the
    field due to a single unit excitation, and the underlying EM problem is
    linear, an arbitrary excitation vector can be reconstructed by weighted
    summation of the per-excitation fields)
  * a stored, settable excitation vector (`set_excitation`) plus a
    `field(theta, phi, k0)` method that can be used as a drop-in
    replacement for analytic pattern functions like `dipole_pattern_ff`

Only numpy and scipy are required.
"""

@dataclass(frozen=True)
class PortModeExcitation:
    """Metadata describing a single port/mode excitation used when the
    far-field solution for a given column of `fields` was generated."""

    port_number: int
    mode_number: int
    smat_index: int
    Z0: complex
    beta: float

class FarfieldDataset:
    """
    Container / interpolator for a multi-frequency 3D far-field dataset of
    the form produced by ``farfield_model``:

        {
            "thetas": np.ndarray[n_theta],
            "phis": np.ndarray[n_phi],
            "solutions": [
                {
                    "freq": float,
                    "k0": float,
                    "n_excitations": int,
                    "excitations": [
                        {"port_number": int, "mode_number": int,
                         "smat_index": int, "Z0": complex, "beta": float},
                        ...
                    ],
                    "fields": [
                        {"E": ndarray[3, n_theta, n_phi] complex,
                         "H": ndarray[3, n_theta, n_phi] complex},
                        ...  # one entry per excitation
                    ],
                },
                ...  # one entry per frequency
            ],
        }

    Internally the data is reshaped into dense grids of shape
    ``(n_freq, n_theta, n_phi, n_excitations, 3)`` for E and H, and two
    `scipy.interpolate.RegularGridInterpolator` instances (one for E, one for
    H) are built over the ``(freq, theta, phi)`` axes. The excitation and
    field-component axes are carried through as extra "vector" output
    dimensions of the interpolator so a single call performs interpolation
    over all excitations/components simultaneously.
    """

    def __init__(self, data: dict, method: str = "linear"):
        """
        Args:
            data: dict as produced by ``farfield_model``.
            method: interpolation method passed to RegularGridInterpolator.
                Complex-valued fields are supported by "linear" and
                "nearest" (both are just weighted/nearest-neighbour
                combinations and work fine on complex numbers). Higher
                order spline methods ("cubic", "quintic") are not
                guaranteed to support complex data in scipy and are not
                recommended here.
        """
        self._raw = data
        self.thetas = np.asarray(data["thetas"], dtype=float)
        self.phis = np.asarray(data["phis"], dtype=float)

        solutions = sorted(data["solutions"], key=lambda s: s["freq"])
        if len(solutions) == 0:
            raise ValueError("Dataset contains no solutions (empty frequency list)")

        self._freqs = np.array([s["freq"] for s in solutions], dtype=float)
        self._k0s = np.array([s["k0"] for s in solutions], dtype=float)

        n_excitations = solutions[0]["n_excitations"]
        for s in solutions:
            if s["n_excitations"] != n_excitations:
                raise ValueError(
                    "All frequency solutions must share the same n_excitations"
                )
        self._n_excitations = n_excitations
        self._excitations = [
            PortModeExcitation(**e) for e in solutions[0]["excitations"]
        ]

        n_freq = len(solutions)
        n_theta = len(self.thetas)
        n_phi = len(self.phis)

        # Internal shape storing fields: (n_freq, n_excitations, 3, n_theta, n_phi)
        E = np.empty((n_freq, n_excitations, 3, n_theta, n_phi), dtype=complex)
        H = np.empty((n_freq, n_excitations, 3, n_theta, n_phi), dtype=complex)

        for fi, s in enumerate(solutions):
            if len(s["fields"]) != n_excitations:
                raise ValueError(
                    f"Solution at freq={s['freq']} has {len(s['fields'])} field "
                    f"entries, expected {n_excitations}"
                )
            for ei, fld in enumerate(s["fields"]):
                Earr = np.asarray(fld["E"])
                Harr = np.asarray(fld["H"])
                self._validate_field_shape(Earr, n_theta, n_phi, "E", s["freq"])
                self._validate_field_shape(Harr, n_theta, n_phi, "H", s["freq"])
                E[fi, ei, :, :, :] = Earr
                H[fi, ei, :, :, :] = Harr

        self._E = E
        self._H = H

        grid = (self._freqs, self.thetas, self.phis)

        # Transpose from (n_freq, n_excitations, 3, n_theta, n_phi)
        # to (n_freq, n_theta, n_phi, n_excitations, 3) for RegularGridInterpolator
        E_grid = np.transpose(E, (0, 3, 4, 1, 2))
        H_grid = np.transpose(H, (0, 3, 4, 1, 2))

        E_flat = E_grid.reshape(n_freq, n_theta, n_phi, n_excitations * 3)
        H_flat = H_grid.reshape(n_freq, n_theta, n_phi, n_excitations * 3)

        self._method = method
        self._E_interp = RegularGridInterpolator(
            grid, E_flat, method=method, bounds_error=False, fill_value=None
        )
        self._H_interp = RegularGridInterpolator(
            grid, H_flat, method=method, bounds_error=False, fill_value=None
        )

        # Currently active excitation vector, used by `field()` and as the
        # default superposition weights for interpolate_E/H/interpolate
        # when no explicit `excitation`/`excitation_amplitudes` is given.
        # Set via `set_excitation(...)`.
        self._excitation: Optional[np.ndarray] = None
        self.set_excitation(0)

    @staticmethod
    def _validate_field_shape(arr, n_theta, n_phi, name, freq):
        if arr.shape != (3, n_theta, n_phi):
            raise ValueError(
                f"{name}-field array at freq={freq} has shape {arr.shape}, "
                f"expected {(3, n_theta, n_phi)}"
            )

    # ------------------------------------------------------------------ #
    # Frequency / grid properties
    # ------------------------------------------------------------------ #

    @property
    def frequencies(self) -> np.ndarray:
        """Sorted array of frequencies (Hz) present in the dataset."""
        return self._freqs

    @property
    def n_frequencies(self) -> int:
        return len(self._freqs)

    @property
    def freq_min(self) -> float:
        return float(self._freqs[0])

    @property
    def freq_max(self) -> float:
        return float(self._freqs[-1])

    @property
    def k0_values(self) -> np.ndarray:
        """Free-space wavenumber k0 for each frequency sample."""
        return self._k0s

    def k0(self, freq: float) -> float:
        """Linearly interpolated free-space wavenumber at ``freq``."""
        return float(np.interp(freq, self._freqs, self._k0s))

    @property
    def n_theta(self) -> int:
        return len(self.thetas)

    @property
    def n_phi(self) -> int:
        return len(self.phis)

    @property
    def theta_range(self) -> tuple[float, float]:
        return float(self.thetas[0]), float(self.thetas[-1])

    @property
    def phi_range(self) -> tuple[float, float]:
        return float(self.phis[0]), float(self.phis[-1])

    @property
    def n_excitations(self) -> int:
        return self._n_excitations

    @property
    def excitations(self) -> list[PortModeExcitation]:
        """Port/mode metadata for each excitation column, in order."""
        return self._excitations

    @property
    def raw(self) -> dict:
        """The original dict this dataset was built from."""
        return self._raw

    # ------------------------------------------------------------------ #
    # Excitation vector
    # ------------------------------------------------------------------ #

    def set_excitations(
        self, *complex_amplitudes: Union[complex, Sequence[complex]]
    ) -> None:
        """
        Set the active excitation vector, used by `field()` and as the
        default weighting for `interpolate_E`/`interpolate_H`/`interpolate`
        when those are called without an explicit `excitation` or
        `excitation_amplitudes` argument.

        Can be called either with individual complex amplitudes::

            dataset.set_excitation(1 + 0j, 0.5j)

        or with a single iterable::

            dataset.set_excitation([1 + 0j, 0.5j])

        The amplitude at index ``i`` corresponds to ``dataset.excitations[i]``
        (i.e. the same ordering as the ``fields`` list per frequency in the
        source dataset).
        """
        if len(complex_amplitudes) == 1 and hasattr(complex_amplitudes[0], "__iter__"):
            amps = np.asarray(complex_amplitudes[0], dtype=complex)
        else:
            amps = np.asarray(complex_amplitudes, dtype=complex)

        if amps.shape != (self._n_excitations,):
            raise ValueError(
                f"Expected {self._n_excitations} excitation amplitude(s), "
                f"got {amps.shape[0] if amps.ndim else 1}"
            )
        self._excitation = amps

    def set_excitation(
        self, port: int
    ) -> None:
        """Excite a specific port."""
        amps = np.zeros((self.n_excitations,), dtype=np.complex128)
        amps[port] = 1.0
        self.set_excitations(*amps)

    @property
    def excitation(self) -> Optional[np.ndarray]:
        """The currently active excitation vector, or None if unset."""
        return self._excitation

    def clear_excitation(self) -> None:
        """Unset the active excitation vector."""
        self._excitation = None

    # ------------------------------------------------------------------ #
    # Interpolation
    # ------------------------------------------------------------------ #

    def _wrap_phi(self, phi: np.ndarray) -> np.ndarray:
        """Wrap phi values into the dataset's phi domain if that domain
        spans a full 2*pi period (so queries just outside [-pi, pi] etc.
        still resolve correctly instead of failing/extrapolating)."""
        lo, hi = self.phis[0], self.phis[-1]
        span = hi - lo
        if np.isclose(span, 2 * np.pi):
            return lo + np.mod(phi - lo, 2 * np.pi)
        return phi

    def _query_points(self, freq, theta, phi):
        scalar_input = np.isscalar(freq) and np.isscalar(theta) and np.isscalar(phi)
        freq_a = np.atleast_1d(np.asarray(freq, dtype=float))
        theta_a = np.atleast_1d(np.asarray(theta, dtype=float))
        phi_a = np.atleast_1d(np.asarray(phi, dtype=float))
        phi_a = self._wrap_phi(phi_a)

        freq_b, theta_b, phi_b = np.broadcast_arrays(freq_a, theta_a, phi_a)
        pts = np.stack([freq_b.ravel(), theta_b.ravel(), phi_b.ravel()], axis=-1)
        return pts, freq_b.shape, scalar_input

    def _interpolate(
        self,
        interpolator: RegularGridInterpolator,
        freq,
        theta,
        phi,
        excitation: Optional[int],
        excitation_amplitudes: Optional[Sequence[complex]],
    ) -> np.ndarray:
        if excitation is not None and excitation_amplitudes is not None:
            raise ValueError(
                "Specify only one of `excitation` or `excitation_amplitudes`"
            )

        pts, out_shape, scalar_input = self._query_points(freq, theta, phi)
        vals = interpolator(pts)  # shape (N, n_excitations * 3)
        vals = vals.reshape(-1, self._n_excitations, 3)

        # Fall back to the stored excitation vector (set via
        # `set_excitation`) if the caller didn't explicitly override it.
        if (
            excitation is None
            and excitation_amplitudes is None
            and self._excitation is not None
        ):
            excitation_amplitudes = self._excitation

        if excitation_amplitudes is not None:
            amps = np.asarray(excitation_amplitudes, dtype=complex)
            if amps.shape[0] != self._n_excitations:
                raise ValueError(
                    f"excitation_amplitudes must have length "
                    f"{self._n_excitations}, got {amps.shape[0]}"
                )
            # weighted superposition over excitations -> (N, 3)
            vals = np.tensordot(amps, vals, axes=([0], [1]))
            vals = vals.reshape(out_shape + (3,))
        elif excitation is not None:
            vals = vals[:, excitation, :]
            vals = vals.reshape(out_shape + (3,))
        else:
            vals = vals.reshape(out_shape + (self._n_excitations, 3))

        if scalar_input:
            vals = vals[0]

        return vals

    def interpolate_E(
        self,
        freq: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
        phi: Union[float, np.ndarray],
        excitation: Optional[int] = None,
        excitation_amplitudes: Optional[Sequence[complex]] = None,
    ) -> np.ndarray:
        """
        Interpolate the E-field (Ex, Ey, Ez) at arbitrary (freq, theta, phi)
        point(s).

        Args:
            freq, theta, phi: scalars or broadcastable arrays of query
                points (theta/phi in radians).
            excitation: if given, return only the field for this single
                excitation index (no superposition). Result shape (..., 3).
            excitation_amplitudes: if given, a complex vector of length
                n_excitations; the returned field is the linear
                superposition ``sum_i amplitudes[i] * field_i``. Result
                shape (..., 3).
            If neither `excitation` nor `excitation_amplitudes` is given,
            the currently active excitation set via `set_excitation()` is
            used (superposed) if one has been set; otherwise all
            excitations are returned unsuperposed: shape
            (..., n_excitations, 3).

        Returns:
            Complex ndarray as described above.
        """
        return self._interpolate(
            self._E_interp, freq, theta, phi, excitation, excitation_amplitudes
        )

    def interpolate_H(
        self,
        freq: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
        phi: Union[float, np.ndarray],
        excitation: Optional[int] = None,
        excitation_amplitudes: Optional[Sequence[complex]] = None,
    ) -> np.ndarray:
        """Same as `interpolate_E` but for the H-field (Hx, Hy, Hz)."""
        return self._interpolate(
            self._H_interp, freq, theta, phi, excitation, excitation_amplitudes
        )

    def interpolate(
        self,
        freq: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
        phi: Union[float, np.ndarray],
        excitation: Optional[int] = None,
        excitation_amplitudes: Optional[Sequence[complex]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper returning ``(E, H)`` interpolated fields."""
        E = self.interpolate_E(freq, theta, phi, excitation, excitation_amplitudes)
        H = self.interpolate_H(freq, theta, phi, excitation, excitation_amplitudes)
        return E, H

    def _freq_from_k0(self, k0: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Map free-space wavenumber k0 -> frequency using the dataset's
        stored (freq, k0) samples."""
        return np.interp(k0, self._k0s, self._freqs)

    def field(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        k0: Union[float, np.ndarray],
        r: Optional[np.ndarray] = None,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Drop-in replacement for analytic pattern functions such as
        ``dipole_pattern_ff(theta, phi, k0)`` / ``dipole_pattern_nf(theta,
        phi, r, k0)``, evaluated against the currently active excitation
        vector (see `set_excitation`).

        Returns:
            (ex, ey, ez, hx, hy, hz), each an ndarray with the broadcast
            shape of theta/phi/k0.
        """
        if self._excitation is None:
            raise RuntimeError(
                "No excitation set. Call `dataset.set_excitation(...)` before "
                "evaluating `field(...)`."
            )
        if r is not None:
            import warnings

            warnings.warn(
                "FarfieldDataset.field() ignores `r`: this dataset only "
                "contains far-field data with no radial dependence.",
                stacklevel=2,
            )
        print(f'Excitations = {self._excitation}')
        freq = self._freq_from_k0(k0)
        E, H = self.interpolate(
            freq, theta, phi, excitation_amplitudes=self._excitation
        )
        ex, ey, ez = E[..., 0], E[..., 1], E[..., 2]
        hx, hy, hz = H[..., 0], H[..., 1], H[..., 2]
        return ex, ey, ez, hx, hy, hz

    def calculate_ff(
        self, theta: np.ndarray, phi: np.ndarray, k0: Union[float, np.ndarray]
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        return self.field(theta, phi, k0)

    def calculate_nf(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        r: np.ndarray,
        k0: Union[float, np.ndarray],
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        return self.field(theta, phi, k0, r=r)

    @classmethod
    def from_dict(cls, data: dict, method: str = "linear") -> "FarfieldDataset":
        return cls(data, method=method)

    def __repr__(self) -> str:
        return (
            f"FarfieldDataset(n_freq={self.n_frequencies}, "
            f"freq_range=({self.freq_min:.4g}, {self.freq_max:.4g}), "
            f"n_theta={self.n_theta}, n_phi={self.n_phi}, "
            f"n_excitations={self.n_excitations})"
        )

    # ------------------------------------------------------------------ #
    # MsgPack Serialization & Deserialization
    # ------------------------------------------------------------------ #

    @staticmethod
    def _msgpack_default(obj):
        """Encoder fallback for msgpack. Supports Python complex types and numpy arrays."""
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        return msgpack_numpy.encode(obj)

    @staticmethod
    def _msgpack_object_hook(obj):
        """Decoder hook for msgpack. Restores numpy arrays and complex scalars."""
        decoded = msgpack_numpy.decode(obj)
        if isinstance(decoded, dict) and decoded.get("__complex__"):
            return complex(decoded["real"], decoded["imag"])
        return decoded

    def to_msgpack(
        self, path_or_stream: Union[str, Path, io.BufferedIOBase, BinaryIO]
    ) -> None:
        """
        Serialize the raw data dictionary (`self._raw`) to MsgPack format using msgpack-numpy.

        Args:
            path_or_stream: Filename (str or Path) or open binary file-like stream.
        """
        packed = msgpack.packb(
            self._raw, default=self._msgpack_default, use_bin_type=True
        )

        if isinstance(path_or_stream, (str, Path)):
            with open(path_or_stream, "wb") as f:
                f.write(packed)
        else:
            path_or_stream.write(packed)

    save_msgpack = to_msgpack  # Convenient alias

    @classmethod
    def from_msgpack(
        cls,
        path_or_stream: Union[str, Path, io.BufferedIOBase, BinaryIO],
        method: str = "linear",
    ) -> "FarfieldDataset":
        """
        Load a raw data dictionary from a MsgPack file or binary stream and construct a FarfieldDataset.

        Args:
            path_or_stream: Filename (str or Path) or open binary file-like stream.
            method: Interpolation method passed to FarfieldDataset.

        Returns:
            A restored FarfieldDataset instance.
        """
        if isinstance(path_or_stream, (str, Path)):
            with open(path_or_stream, "rb") as f:
                packed = f.read()
        else:
            packed = path_or_stream.read()

        data = msgpack.unpackb(
            packed, object_hook=cls._msgpack_object_hook, raw=False
        )

        return cls(data, method=method)

    load_msgpack = from_msgpack  # Convenient alias

class AntennaModel(Antenna):

    def __init__(self, 
                 x: float, 
                 y: float, 
                 z: float, 
                 frequency: float,  
                 dataset: dict | FarfieldDataset, 
                 cs: CoordinateSystem = None,
                 name: str = 'Antenna'):
        if isinstance(dataset,dict):
            self.ffset = FarfieldDataset(dataset)
        else:
            self.ffset = dataset
        
        super().__init__(x,y,z,frequency,cs, self.ffset.calculate_nf, self.ffset.calculate_ff)

        self.set_excitation = self.ffset.set_excitation
        self.set_excitations = self.ffset.set_excitations

    def receive_from(self, other) -> np.ndarray:
        N = self.ffset.n_excitations

        out = []
        for i in range(N):
            self.set_excitation(i)
            out.append(super().receive_from(other))
        return np.array(out)
        
    def save(self, filename: str) -> None:
        self.ffset.to_msgpack(filename)

    @staticmethod
    def load(x: float,
             y: float,
             z: float,
             frequency: float,
             filename: str,
             cs: CoordinateSystem = None) -> AntennaModel:
        dataset = FarfieldDataset.from_msgpack(filename)
        return AntennaModel(x,y,z,frequency, dataset, cs=cs)