"""Classes for basis."""

from abc import ABC, abstractmethod

from zernipax.backend import fori_loop, jit, jnp, np, sign
from zernipax.zernike import fourier, zernike_radial_cpu


def flatten_list(x, flatten_tuple=False):
    """Flatten a nested list.

    Parameters
    ----------
    x : list
        nested list of lists to flatten
    flatten_tuple : bool
        Whether to also flatten nested tuples.

    Returns
    -------
    x : list
        flattened input

    """
    types = (list,)
    if flatten_tuple:
        types += (tuple,)
    if isinstance(x, types):
        return [a for i in x for a in flatten_list(i, flatten_tuple)]
    else:
        return [x]


@jit
def copy_coeffs(c_old, modes_old, modes_new, c_new=None):
    """Copy coefficients from one resolution to another."""
    modes_old, modes_new = jnp.atleast_1d(modes_old), jnp.atleast_1d(modes_new)

    if modes_old.ndim == 1:
        modes_old = modes_old.reshape((-1, 1))
    if modes_new.ndim == 1:
        modes_new = modes_new.reshape((-1, 1))

    if c_new is None:
        c_new = jnp.zeros((modes_new.shape[0],))
    c_old, c_new = jnp.asarray(c_old), jnp.asarray(c_new)

    def body(i, c_new):
        mask = (modes_old[i, :] == modes_new).all(axis=1)
        c_new = jnp.where(mask, c_old[i], c_new)
        return c_new

    if c_old.size:
        c_new = fori_loop(0, modes_old.shape[0], body, c_new)
    return c_new


class _Basis(ABC):
    """Basis is an abstract base class for spectral basis sets."""

    def __init__(self):
        self._enforce_symmetry()
        self._sort_modes()
        self._create_idx()
        # ensure things that should be ints are ints
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)
        self._NFP = int(self._NFP)
        self._modes = self._modes.astype(int)

    def _set_up(self):
        """Do things after loading or changing resolution."""
        # Also recreates any attributes not in _io_attrs on load from input file.
        # See IOAble class docstring for more info.
        self._enforce_symmetry()
        self._sort_modes()
        self._create_idx()
        # ensure things that should be ints are ints
        self._L = int(self._L)
        self._M = int(self._M)
        self._N = int(self._N)
        self._NFP = int(self._NFP)
        self._modes = self._modes.astype(int)

    def _enforce_symmetry(self):
        """Enforce stellarator symmetry."""
        assert self.sym in [
            "sin",
            "sine",
            "cos",
            "cosine",
            "even",
            "cos(t)",
            False,
            None,
        ], f"Unknown symmetry type {self.sym}"
        if self.sym in ["cos", "cosine"]:  # cos(m*t-n*z) symmetry
            self._modes = self.modes[
                np.asarray(sign(self.modes[:, 1]) == sign(self.modes[:, 2]))
            ]
        elif self.sym in ["sin", "sine"]:  # sin(m*t-n*z) symmetry
            self._modes = self.modes[
                np.asarray(sign(self.modes[:, 1]) != sign(self.modes[:, 2]))
            ]
        elif self.sym == "even":  # even powers of rho
            self._modes = self.modes[np.asarray(self.modes[:, 0] % 2 == 0)]
        elif self.sym == "cos(t)":  # cos(m*t) terms only
            self._modes = self.modes[np.asarray(sign(self.modes[:, 1]) >= 0)]
        elif self.sym is None:
            self._sym = False

    def _sort_modes(self):
        """Sorts modes for use with FFT."""
        sort_idx = np.lexsort((self.modes[:, 1], self.modes[:, 0], self.modes[:, 2]))
        self._modes = self.modes[sort_idx]

    def _create_idx(self):
        """Create index for use with self.get_idx()."""
        self._idx = {}
        for idx, (L, M, N) in enumerate(self.modes):
            if L not in self._idx:
                self._idx[L] = {}
            if M not in self._idx[L]:
                self._idx[L][M] = {}
            self._idx[L][M][N] = idx

    def get_idx(self, L=0, M=0, N=0, error=True):
        """Get the index of the ``'modes'`` array corresponding to given mode numbers.

        Parameters
        ----------
        L : int
            Radial mode number.
        M : int
            Poloidal mode number.
        N : int
            Toroidal mode number.
        error : bool
            whether to raise exception if mode is not in basis, or return empty array

        Returns
        -------
        idx : ndarray of int
            Index of given mode numbers.

        """
        try:
            return self._idx[L][M][N]
        except KeyError as e:
            if error:
                raise ValueError(
                    "mode ({}, {}, {}) is not in basis {}".format(L, M, N, str(self))
                ) from e
            else:
                return np.array([]).astype(int)

    @abstractmethod
    def _get_modes(self):
        """ndarray: Mode numbers for the basis."""

    @abstractmethod
    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            node coordinates, in (rho,theta,zeta)
        derivatives : ndarray of int, shape(3,)
            order of derivatives to compute in (rho,theta,zeta)
        modes : ndarray of in, shape(num_modes,3), optional
            basis modes to evaluate (if None, full basis is used)
        unique : bool, optional
            whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            basis functions evaluated at nodes

        """

    @abstractmethod
    def change_resolution(self):
        """Change resolution of the basis to the given resolutions."""

    @property
    def L(self):
        """int: Maximum radial resolution."""
        return self.__dict__.setdefault("_L", 0)

    @L.setter
    def L(self, L):
        assert int(L) == L, "Basis Resolution must be an integer!"
        self._L = int(L)

    @property
    def M(self):
        """int:  Maximum poloidal resolution."""
        return self.__dict__.setdefault("_M", 0)

    @M.setter
    def M(self, M):
        assert int(M) == M, "Basis Resolution must be an integer!"
        self._M = int(M)

    @property
    def N(self):
        """int: Maximum toroidal resolution."""
        return self.__dict__.setdefault("_N", 0)

    @N.setter
    def N(self, N):
        assert int(N) == N, "Basis Resolution must be an integer!"
        self._N = int(N)

    @property
    def NFP(self):
        """int: Number of field periods."""
        return self.__dict__.setdefault("_NFP", 1)

    @property
    def sym(self):
        """str: {``'cos'``, ``'sin'``, ``False``} Type of symmetry."""
        return self.__dict__.setdefault("_sym", False)

    @property
    def modes(self):
        """ndarray: Mode numbers [l,m,n]."""
        return self.__dict__.setdefault("_modes", np.array([]).reshape((0, 3)))

    @modes.setter
    def modes(self, modes):
        self._modes = modes

    @property
    def num_modes(self):
        """int: Total number of modes in the spectral basis."""
        return self.modes.shape[0]

    @property
    def spectral_indexing(self):
        """str: Type of indexing used for the spectral basis."""
        return self.__dict__.setdefault("_spectral_indexing", "linear")

    def __repr__(self):
        """Get the string form of the object."""
        return (
            type(self).__name__
            + " at "
            + str(hex(id(self)))
            + " (L={}, M={}, N={}, NFP={}, sym={}, spectral_indexing={})".format(
                self.L, self.M, self.N, self.NFP, self.sym, self.spectral_indexing
            )
        )


class ZernikePolynomial(_Basis):
    """2D basis set for analytic functions in a unit disc.

    Parameters
    ----------
    L : int
        Maximum radial resolution. Use L=-1 for default based on M.
    M : int
        Maximum poloidal resolution.
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'ansi'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triangle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape.

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

    """

    def __init__(self, L, M, sym=False, spectral_indexing="ansi"):
        self.L = L
        self.M = M
        self.N = 0
        self._NFP = 1
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        self._modes = self._get_modes(
            L=self.L, M=self.M, spectral_indexing=self.spectral_indexing
        )

        super().__init__()

    def _get_modes(self, L=-1, M=0, spectral_indexing="ansi"):
        """Get mode numbers for Fourier-Zernike basis functions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        spectral_indexing : {``'ansi'``, ``'fringe'``}
            Indexing method, default value = ``'ansi'``

            For L=0, all methods are equivalent and give a "chevron" shaped
            basis (only the outer edge of the zernike pyramid of width M).
            For L>0, the indexing scheme defines order of the basis functions:

            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triangle shape. For L == M,
            the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
            to the bottom of the pyramid, increasing L while keeping M constant,
            giving a "house" shape.

            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape for L=2*M where
            the traditional fringe/U of Arizona indexing is recovered.
            For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

        """
        assert spectral_indexing in [
            "ansi",
            "fringe",
        ], "Unknown spectral_indexing: {}".format(spectral_indexing)
        default_L = {"ansi": M, "fringe": 2 * M}
        L = L if L >= 0 else default_L.get(spectral_indexing, M)
        self.L = L

        if spectral_indexing == "ansi":
            pol_posm = [
                [(m + d, m) for m in range(0, M + 1) if m + d < M + 1]
                for d in range(0, L + 1, 2)
            ]
            if L > M:
                pol_posm += [
                    (l, m)
                    for l in range(M + 1, L + 1)
                    for m in range(0, M + 1)
                    if (l - m) % 2 == 0
                ]

        elif spectral_indexing == "fringe":
            pol_posm = [
                [(m + d // 2, m - d // 2) for m in range(0, M + 1) if m - d // 2 >= 0]
                for d in range(0, L + 1, 2)
            ]
            if L > 2 * M:
                pol_posm += [
                    [(l - m, m) for m in range(0, M + 1)]
                    for l in range(2 * M, L + 1, 2)
                ]

        pol = [
            [(l, m), (l, -m)] if m != 0 else [(l, m)] for l, m in flatten_list(pol_posm)
        ]
        pol = np.array(flatten_list(pol))
        num_pol = len(pol)
        tor = np.zeros((num_pol, 1))

        return np.hstack([pol, tor])

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of int, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used).
        unique : bool, optional
            Whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff.

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            Basis functions evaluated at nodes.

        """
        if modes is None:
            modes = self.modes
        if derivatives[2] != 0:
            return jnp.zeros((nodes.shape[0], modes.shape[0]))
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        r, t, z = nodes.T
        l, m, n = modes.T
        lm = modes[:, :2]

        if unique:
            _, ridx, routidx = np.unique(
                r, return_index=True, return_inverse=True, axis=0
            )
            _, tidx, toutidx = np.unique(
                t, return_index=True, return_inverse=True, axis=0
            )
            _, lmidx, lmoutidx = np.unique(
                lm, return_index=True, return_inverse=True, axis=0
            )
            _, midx, moutidx = np.unique(
                m, return_index=True, return_inverse=True, axis=0
            )
            r = r[ridx]
            t = t[tidx]
            lm = lm[lmidx]
            m = m[midx]

        radial = zernike_radial_cpu(r, lm[:, 0], lm[:, 1], dr=derivatives[0])
        poloidal = fourier(t[:, np.newaxis], m, 1, derivatives[1])

        if unique:
            radial = radial[routidx][:, lmoutidx]
            poloidal = poloidal[toutidx][:, moutidx]

        return radial * poloidal

    def change_resolution(self, L, M, sym=None):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        sym : bool
            Whether to enforce stellarator symmetry.

        Returns
        -------
        None

        """
        if L != self.L or M != self.M or sym != self.sym:
            self.L = L
            self.M = M
            self._sym = sym if sym is not None else self.sym
            self._modes = self._get_modes(
                self.L, self.M, spectral_indexing=self.spectral_indexing
            )
            self._set_up()


class FourierZernikeBasis(_Basis):
    """3D basis set for analytic functions in a toroidal volume.

    Zernike polynomials in the radial & poloidal coordinates, and a Fourier
    series in the toroidal coordinate.

    Parameters
    ----------
    L : int
        Maximum radial resolution. Use L=-1 for default based on M.
    M : int
        Maximum poloidal resolution.
    N : int
        Maximum toroidal resolution.
    NFP : int
        Number of field periods.
    sym : {``'cos'``, ``'sin'``, ``False``}
        * ``'cos'`` for cos(m*t-n*z) symmetry
        * ``'sin'`` for sin(m*t-n*z) symmetry
        * ``False`` for no symmetry (Default)
    spectral_indexing : {``'ansi'``, ``'fringe'``}
        Indexing method, default value = ``'ansi'``

        For L=0, all methods are equivalent and give a "chevron" shaped
        basis (only the outer edge of the zernike pyramid of width M).
        For L>0, the indexing scheme defines order of the basis functions:

        ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
        decreasing size, ending in a triangle shape. For L == M,
        the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
        to the bottom of the pyramid, increasing L while keeping M constant,
        giving a "house" shape.

        ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
        decreasing size, ending in a diamond shape for L=2*M where
        the traditional fringe/U of Arizona indexing is recovered.
        For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

    """

    def __init__(self, L, M, N, NFP=1, sym=False, spectral_indexing="ansi"):
        self.L = L
        self.M = M
        self.N = N
        self._NFP = NFP
        self._sym = sym
        self._spectral_indexing = spectral_indexing

        self._modes = self._get_modes(
            L=self.L, M=self.M, N=self.N, spectral_indexing=self.spectral_indexing
        )

        super().__init__()

    def _get_modes(self, L=-1, M=0, N=0, spectral_indexing="ansi"):
        """Get mode numbers for Fourier-Zernike basis functions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        N : int
            Maximum toroidal resolution.
        spectral_indexing : {``'ansi'``, ``'fringe'``}
            Indexing method, default value = ``'ansi'``

            For L=0, all methods are equivalent and give a "chevron" shaped
            basis (only the outer edge of the zernike pyramid of width M).
            For L>0, the indexing scheme defines order of the basis functions:

            ``'ansi'``: ANSI indexing fills in the pyramid with triangles of
            decreasing size, ending in a triangle shape. For L == M,
            the traditional ANSI pyramid indexing is recovered. For L>M, adds rows
            to the bottom of the pyramid, increasing L while keeping M constant,
            giving a "house" shape.

            ``'fringe'``: Fringe indexing fills in the pyramid with chevrons of
            decreasing size, ending in a diamond shape for L=2*M where
            the traditional fringe/U of Arizona indexing is recovered.
            For L > 2*M, adds chevrons to the bottom, making a hexagonal diamond.

        Returns
        -------
        modes : ndarray of int, shape(num_modes,3)
            Array of mode numbers [l,m,n].
            Each row is one basis function with modes (l,m,n).

        """
        assert spectral_indexing in [
            "ansi",
            "fringe",
        ], "Unknown spectral_indexing: {}".format(spectral_indexing)
        default_L = {"ansi": M, "fringe": 2 * M}
        L = L if L >= 0 else default_L.get(spectral_indexing, M)
        self.L = L

        if spectral_indexing == "ansi":
            pol_posm = [
                [(m + d, m) for m in range(0, M + 1) if m + d < M + 1]
                for d in range(0, L + 1, 2)
            ]
            if L > M:
                pol_posm += [
                    (l, m)
                    for l in range(M + 1, L + 1)
                    for m in range(0, M + 1)
                    if (l - m) % 2 == 0
                ]

        elif spectral_indexing == "fringe":
            pol_posm = [
                [(m + d // 2, m - d // 2) for m in range(0, M + 1) if m - d // 2 >= 0]
                for d in range(0, L + 1, 2)
            ]
            if L > 2 * M:
                pol_posm += [
                    [(l - m, m) for m in range(0, M + 1)]
                    for l in range(2 * M, L + 1, 2)
                ]

        pol = [
            [(l, m), (l, -m)] if m != 0 else [(l, m)] for l, m in flatten_list(pol_posm)
        ]
        pol = np.array(flatten_list(pol))
        num_pol = len(pol)

        pol = np.tile(pol, (2 * N + 1, 1))
        tor = np.atleast_2d(
            np.tile(np.arange(-N, N + 1), (num_pol, 1)).flatten(order="f")
        ).T
        return np.unique(np.hstack([pol, tor]), axis=0)

    def evaluate(
        self, nodes, derivatives=np.array([0, 0, 0]), modes=None, unique=False
    ):
        """Evaluate basis functions at specified nodes.

        Parameters
        ----------
        nodes : ndarray of float, size(num_nodes,3)
            Node coordinates, in (rho,theta,zeta).
        derivatives : ndarray of int, shape(num_derivatives,3)
            Order of derivatives to compute in (rho,theta,zeta).
        modes : ndarray of int, shape(num_modes,3), optional
            Basis modes to evaluate (if None, full basis is used).
        unique : bool, optional
            Whether to workload by only calculating for unique values of nodes, modes
            can be faster, but doesn't work with jit or autodiff.

        Returns
        -------
        y : ndarray, shape(num_nodes,num_modes)
            Basis functions evaluated at nodes.

        """
        if modes is None:
            modes = self.modes
        if not len(modes):
            return np.array([]).reshape((len(nodes), 0))

        # TODO: avoid duplicate calculations when mixing derivatives
        r, t, z = nodes.T
        l, m, n = modes.T
        lm = modes[:, :2]

        if unique:
            # TODO: can avoid this here by using grid.unique_idx etc
            # and adding unique_modes attributes to basis
            _, ridx, routidx = np.unique(
                r, return_index=True, return_inverse=True, axis=0
            )
            _, tidx, toutidx = np.unique(
                t, return_index=True, return_inverse=True, axis=0
            )
            _, zidx, zoutidx = np.unique(
                z, return_index=True, return_inverse=True, axis=0
            )
            _, lmidx, lmoutidx = np.unique(
                lm, return_index=True, return_inverse=True, axis=0
            )
            _, midx, moutidx = np.unique(
                m, return_index=True, return_inverse=True, axis=0
            )
            _, nidx, noutidx = np.unique(
                n, return_index=True, return_inverse=True, axis=0
            )
            r = r[ridx]
            t = t[tidx]
            z = z[zidx]
            lm = lm[lmidx]
            m = m[midx]
            n = n[nidx]

        radial = zernike_radial_cpu(r, lm[:, 0], lm[:, 1], dr=derivatives[0])
        poloidal = fourier(t[:, np.newaxis], m, dt=derivatives[1])
        toroidal = fourier(z[:, np.newaxis], n, NFP=self.NFP, dt=derivatives[2])
        if unique:
            radial = radial[routidx][:, lmoutidx]
            poloidal = poloidal[toutidx][:, moutidx]
            toroidal = toroidal[zoutidx][:, noutidx]

        return radial * poloidal * toroidal

    def change_resolution(self, L, M, N, NFP=None, sym=None):
        """Change resolution of the basis to the given resolutions.

        Parameters
        ----------
        L : int
            Maximum radial resolution.
        M : int
            Maximum poloidal resolution.
        N : int
            Maximum toroidal resolution.
        NFP : int
            Number of field periods.
        sym : bool
            Whether to enforce stellarator symmetry.

        Returns
        -------
        None

        """
        self._NFP = NFP if NFP is not None else self.NFP
        if L != self.L or M != self.M or N != self.N or sym != self.sym:
            self.L = L
            self.M = M
            self.N = N
            self._sym = sym if sym is not None else self.sym
            self._modes = self._get_modes(
                self.L, self.M, self.N, spectral_indexing=self.spectral_indexing
            )
            self._set_up()
