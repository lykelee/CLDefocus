"""Functions for evaluating Zernike polynomials and their derivatives."""

import functools
from math import factorial

from zernipax.backend import (
    cond,
    custom_jvp,
    custom_jvp_with_jit,
    execute_on_cpu,
    fori_loop,
    jit,
    jnp,
    np,
)


def jacobi_poly_single(x, n, alpha, beta=0, P_n1=0, P_n2=0):
    """Evaluate Jacobi for single alpha and n pair."""
    c = 2 * n + alpha + beta
    a1 = 2 * n * (c - n) * (c - 2)
    a2 = (c - 1) * (c * (c - 2) * x + (alpha - beta) * (alpha + beta))
    a3 = 2 * (n + alpha - 1) * (n + beta - 1) * c

    a1 = jnp.where(a1 == 0, 1e-6, a1)
    P_n = (a2 * P_n1 - a3 * P_n2) / a1
    # Checks for special cases
    P_n = jnp.where(n < 0, 0, P_n)
    P_n = jnp.where(n == 0, 1, P_n)
    P_n = jnp.where(n == 1, (alpha + 1) + (alpha + beta + 2) * (x - 1) / 2, P_n)
    return P_n


@execute_on_cpu
@custom_jvp_with_jit
def zernike_radial_unique(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes. This version
    is almost the same as zernike_radial_old function but way faster and more
    accurate.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if dr > 4:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]
        P_n1 = P_past[1]
        P_n = jnp.zeros((dr + 1, r.size))

        def find_inter_jacobi(dx, args):
            N, alpha, P_n1, P_n2, P_n = args
            P_n = P_n.at[dx, :].set(
                jacobi_poly_single(r_jacobi, N - dx, alpha + dx, dx, P_n1[dx], P_n2[dx])
            )
            return (N, alpha, P_n1, P_n2, P_n)

        # Calculate Jacobi polynomial and derivatives for (m,n)
        _, _, _, _, P_n = fori_loop(
            0, dr + 1, find_inter_jacobi, (N, alpha, P_n1, P_n2, P_n)
        )

        coef = jnp.array(
            [
                1,
                (alpha + N + 1) / 2,
                (alpha + N + 2) * (alpha + N + 1) / 4,
                (alpha + N + 3) * (alpha + N + 2) * (alpha + N + 1) / 8,
                (alpha + N + 4)
                * (alpha + N + 3)
                * (alpha + N + 2)
                * (alpha + N + 1)
                / 16,
            ]
        )
        # TODO: A version without if statements are possible?
        if dr == 0:
            result = (-1) ** N * r**alpha * P_n[0]
        elif dr == 1:
            result = (-1) ** N * (
                alpha * r ** jnp.maximum(alpha - 1, 0) * P_n[0]
                - coef[1] * 4 * r ** (alpha + 1) * P_n[1]
            )
        elif dr == 2:
            result = (-1) ** N * (
                (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n[0]
                - coef[1] * 4 * (2 * alpha + 1) * r**alpha * P_n[1]
                + coef[2] * 16 * r ** (alpha + 2) * P_n[2]
            )
        elif dr == 3:
            result = (-1) ** N * (
                (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 3, 0)
                * P_n[0]
                - coef[1] * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * P_n[1]
                + coef[2] * 48 * (alpha + 1) * r ** (alpha + 1) * P_n[2]
                - coef[3] * 64 * r ** (alpha + 3) * P_n[3]
            )
        elif dr == 4:
            result = (-1) ** N * (
                (alpha - 3)
                * (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 4, 0)
                * P_n[0]
                - coef[1]
                * 8
                * alpha
                * (2 * alpha**2 - 3 * alpha + 1)
                * r ** jnp.maximum(alpha - 2, 0)
                * P_n[1]
                + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * P_n[2]
                - coef[3] * 128 * (2 * alpha + 3) * r ** (alpha + 2) * P_n[3]
                + coef[4] * 256 * r ** (alpha + 4) * P_n[4]
            )
        index = jnp.where(
            jnp.logical_and(m == alpha, n == N),
            jnp.arange(1, m.size + 1),
            0,
        )
        idx = jnp.sum(index)
        # needed for proper index
        idx -= 1
        out = out.at[:, idx].set(jnp.where(idx >= 0, result, out.at[:, idx].get()))

        # Shift past values if needed
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask[:, None], P_n1, P_n2)
        P_n1 = jnp.where(mask[:, None], P_n, P_n1)
        P_past = P_past.at[0, :, :].set(P_n2)
        P_past = P_past.at[1, :, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        def find_init_jacobi(dx, args):
            alpha, P_past = args
            P_past = P_past.at[0, dx, :].set(
                jacobi_poly_single(r_jacobi, 0, alpha + dx, beta=dx)
            )
            P_past = P_past.at[1, dx, :].set(
                jacobi_poly_single(r_jacobi, 1, alpha + dx, beta=dx)
            )
            return (alpha, P_past)

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, dr + 1, r.size))
        _, P_past = fori_loop(0, dr + 1, find_init_jacobi, (alpha, P_past))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    r = jnp.atleast_1d(r)
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = int(dr)

    out = jnp.zeros((r.size, m.size))
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)
    dxs = jnp.arange(0, dr + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@custom_jvp_with_jit
def zernike_radial_cpu(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Calculates Radial part of Zernike Polynomials using Jacobi recursion relation
    by getting rid of the redundant calculations for appropriate modes. This version
    is almost the same as zernike_radial_gpu function but way faster and more
    accurate on CPU.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    out : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if dr > 4:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )

    def update(x, args):
        alpha, N, result, out = args
        idx = jnp.where(jnp.logical_and(m[x] == alpha, n[x] == N), x, -1)

        def falseFun(args):
            _, _, out = args
            return out

        def trueFun(args):
            idx, result, out = args
            out = out.at[:, idx].set(result)
            return out

        out = cond(idx >= 0, trueFun, falseFun, (idx, result, out))
        return (alpha, N, result, out)

    def body_inner(N, args):
        alpha, out, P_past = args
        P_n2 = P_past[0]
        P_n1 = P_past[1]
        P_n = jnp.zeros((dr + 1, r.size))

        def find_inter_jacobi(dx, args):
            N, alpha, P_n1, P_n2, P_n = args
            P_n = P_n.at[dx, :].set(
                jacobi_poly_single(r_jacobi, N - dx, alpha + dx, dx, P_n1[dx], P_n2[dx])
            )
            return (N, alpha, P_n1, P_n2, P_n)

        # Calculate Jacobi polynomial and derivatives for (m,n)
        _, _, _, _, P_n = fori_loop(
            0, dr + 1, find_inter_jacobi, (N, alpha, P_n1, P_n2, P_n)
        )

        coef = jnp.array(
            [
                1,
                (alpha + N + 1) / 2,
                (alpha + N + 2) * (alpha + N + 1) / 4,
                (alpha + N + 3) * (alpha + N + 2) * (alpha + N + 1) / 8,
                (alpha + N + 4)
                * (alpha + N + 3)
                * (alpha + N + 2)
                * (alpha + N + 1)
                / 16,
            ]
        )
        # TODO: A version without if statements are possible?
        if dr == 0:
            result = (-1) ** N * r**alpha * P_n[0]
        elif dr == 1:
            result = (-1) ** N * (
                alpha * r ** jnp.maximum(alpha - 1, 0) * P_n[0]
                - coef[1] * 4 * r ** (alpha + 1) * P_n[1]
            )
        elif dr == 2:
            result = (-1) ** N * (
                (alpha - 1) * alpha * r ** jnp.maximum(alpha - 2, 0) * P_n[0]
                - coef[1] * 4 * (2 * alpha + 1) * r**alpha * P_n[1]
                + coef[2] * 16 * r ** (alpha + 2) * P_n[2]
            )
        elif dr == 3:
            result = (-1) ** N * (
                (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 3, 0)
                * P_n[0]
                - coef[1] * 12 * alpha**2 * r ** jnp.maximum(alpha - 1, 0) * P_n[1]
                + coef[2] * 48 * (alpha + 1) * r ** (alpha + 1) * P_n[2]
                - coef[3] * 64 * r ** (alpha + 3) * P_n[3]
            )
        elif dr == 4:
            result = (-1) ** N * (
                (alpha - 3)
                * (alpha - 2)
                * (alpha - 1)
                * alpha
                * r ** jnp.maximum(alpha - 4, 0)
                * P_n[0]
                - coef[1]
                * 8
                * alpha
                * (2 * alpha**2 - 3 * alpha + 1)
                * r ** jnp.maximum(alpha - 2, 0)
                * P_n[1]
                + coef[2] * 48 * (2 * alpha**2 + 2 * alpha + 1) * r**alpha * P_n[2]
                - coef[3] * 128 * (2 * alpha + 3) * r ** (alpha + 2) * P_n[3]
                + coef[4] * 256 * r ** (alpha + 4) * P_n[4]
            )
        _, _, _, out = fori_loop(0, m.size, update, (alpha, N, result, out))

        # Shift past values if needed
        mask = N >= 2 + dxs
        P_n2 = jnp.where(mask[:, None], P_n1, P_n2)
        P_n1 = jnp.where(mask[:, None], P_n, P_n1)
        P_past = P_past.at[0, :, :].set(P_n2)
        P_past = P_past.at[1, :, :].set(P_n1)

        return (alpha, out, P_past)

    def body(alpha, out):
        # find l values with m values equal to alpha
        l_alpha = jnp.where(m == alpha, l, 0)
        # find the maximum among them
        L_max = jnp.max(l_alpha)
        # Maximum possible value for n for loop bound
        N_max = (L_max - alpha) // 2

        def find_init_jacobi(dx, args):
            alpha, P_past = args
            P_past = P_past.at[0, dx, :].set(
                jacobi_poly_single(r_jacobi, 0, alpha + dx, beta=dx)
            )
            P_past = P_past.at[1, dx, :].set(
                jacobi_poly_single(r_jacobi, 1, alpha + dx, beta=dx)
            )
            return (alpha, P_past)

        # First 2 Jacobi Polynomials (they don't need recursion)
        # P_past stores last 2 Jacobi polynomials (and required derivatives)
        # evaluated at given r points
        P_past = jnp.zeros((2, dr + 1, r.size))
        _, P_past = fori_loop(0, dr + 1, find_init_jacobi, (alpha, P_past))

        # Loop over every n value
        _, out, _ = fori_loop(
            0, (N_max + 1).astype(int), body_inner, (alpha, out, P_past)
        )
        return out

    r = jnp.atleast_1d(r)
    m = jnp.atleast_1d(m)
    l = jnp.atleast_1d(l)
    dr = int(dr)

    out = jnp.zeros((r.size, m.size))
    r_jacobi = 1 - 2 * r**2
    m = jnp.abs(m)
    n = ((l - m) // 2).astype(int)
    dxs = jnp.arange(0, dr + 1)

    M_max = jnp.max(m)
    # Loop over every different m value. There is another nested
    # loop which will execute necessary n values.
    out = fori_loop(0, (M_max + 1).astype(int), body, (out))
    return out


@functools.partial(jit, static_argnums=3)
def zernike_radial_gpu(r, l, m, dr=0):
    """Radial part of zernike polynomials.

    Evaluates basis functions using JAX and a stable
    evaluation scheme based on jacobi polynomials and
    binomial coefficients. Generally faster for L>24
    and differentiable, but slower for low resolution.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    m = jnp.abs(m).astype(float)
    alpha = m
    beta = 0
    n = (l - m) // 2
    s = (-1) ** n
    jacobi_arg = 1 - 2 * r**2
    if dr == 0:
        out = r**m * _jacobi(n, alpha, beta, jacobi_arg, 0)
    elif dr == 1:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        out = m * r ** jnp.maximum(m - 1, 0) * f - 4 * r ** (m + 1) * df
    elif dr == 2:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        out = (
            (m - 1) * m * r ** jnp.maximum(m - 2, 0) * f
            - 4 * (2 * m + 1) * r**m * df
            + 16 * r ** (m + 2) * d2f
        )
    elif dr == 3:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        d3f = _jacobi(n, alpha, beta, jacobi_arg, 3)
        out = (
            (m - 2) * (m - 1) * m * r ** jnp.maximum(m - 3, 0) * f
            - 12 * m**2 * r ** jnp.maximum(m - 1, 0) * df
            + 48 * (m + 1) * r ** (m + 1) * d2f
            - 64 * r ** (m + 3) * d3f
        )
    elif dr == 4:
        f = _jacobi(n, alpha, beta, jacobi_arg, 0)
        df = _jacobi(n, alpha, beta, jacobi_arg, 1)
        d2f = _jacobi(n, alpha, beta, jacobi_arg, 2)
        d3f = _jacobi(n, alpha, beta, jacobi_arg, 3)
        d4f = _jacobi(n, alpha, beta, jacobi_arg, 4)
        out = (
            (m - 3) * (m - 2) * (m - 1) * m * r ** jnp.maximum(m - 4, 0) * f
            - 8 * m * (2 * m**2 - 3 * m + 1) * r ** jnp.maximum(m - 2, 0) * df
            + 48 * (2 * m**2 + 2 * m + 1) * r**m * d2f
            - 128 * (2 * m + 3) * r ** (m + 2) * d3f
            + 256 * r ** (m + 4) * d4f
        )
    else:
        raise NotImplementedError(
            "Analytic radial derivatives of Zernike polynomials for order>4 "
            + "have not been implemented."
        )
    return s * jnp.where((l - m) % 2 == 0, out, 0)


@jit
def fourier(theta, m, NFP=1, dt=0):
    """Fourier series.

    Parameters
    ----------
    theta : ndarray, shape(N,)
        poloidal/toroidal coordinates to evaluate basis
    m : ndarray of int, shape(K,)
        poloidal/toroidal mode number(s)
    NFP : int
        number of field periods (Default = 1)
    dt : int
        order of derivative (Default = 0)

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    theta, m, NFP, dt = map(jnp.asarray, (theta, m, NFP, dt))
    m_pos = (m >= 0).astype(int)
    m_abs = jnp.abs(m) * NFP
    shift = m_pos * jnp.pi / 2 + dt * jnp.pi / 2
    return m_abs**dt * jnp.sin(m_abs * theta + shift)


@functools.partial(custom_jvp, nondiff_argnums=(4,))
@jit
@jnp.vectorize
def _jacobi(n, alpha, beta, x, dx=0):
    """Jacobi polynomial evaluation.

    Implementation is only correct for non-negative integer coefficients,
    returns 0 otherwise.

    Parameters
    ----------
    n : int, array_like
        Degree of the polynomial.
    alpha : int, array_like
        Parameter
    beta : int, array_like
        Parameter
    x : float, array_like
        Points at which to evaluate the polynomial

    Returns
    -------
    P : ndarray
        Values of the Jacobi polynomial
    """
    # adapted from scipy:
    # https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/
    # scipy/special/orthogonal_eval.pxd#L144

    def _jacobi_body_fun(kk, d_p_a_b_x):
        d, p, alpha, beta, x = d_p_a_b_x
        k = kk + 1.0
        t = 2 * k + alpha + beta
        d = (
            (t * (t + 1) * (t + 2)) * (x - 1) * p + 2 * k * (k + beta) * (t + 2) * d
        ) / (2 * (k + alpha + 1) * (k + alpha + beta + 1) * t)
        p = d + p
        return (d, p, alpha, beta, x)

    n, alpha, beta, x = map(jnp.asarray, (n, alpha, beta, x))

    # coefficient for derivative
    coeffs = jnp.array(
        [
            1,
            (alpha + n + 1) / 2,
            (alpha + n + 2) * (alpha + n + 1) / 4,
            (alpha + n + 3) * (alpha + n + 2) * (alpha + n + 1) / 8,
            (alpha + n + 4) * (alpha + n + 3) * (alpha + n + 2) * (alpha + n + 1) / 16,
        ]
    )
    c = coeffs[dx]
    # taking derivative is same as c*jacobi but for shifted n,a,b
    n -= dx
    alpha += dx
    beta += dx

    d = (alpha + beta + 2) * (x - 1) / (2 * (alpha + 1))
    p = d + 1
    d, p, alpha, beta, x = fori_loop(
        0, jnp.maximum(n - 1, 0).astype(int), _jacobi_body_fun, (d, p, alpha, beta, x)
    )
    out = _binom(n + alpha, n) * p
    # should be complex for n<0, but it gets replaced elsewhere so just return 0 here
    out = jnp.where(n < 0, 0, out)
    # other edge cases
    out = jnp.where(n == 0, 1.0, out)
    out = jnp.where(n == 1, 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (x - 1)), out)
    return c * out


@_jacobi.defjvp
def _jacobi_jvp(dx, x, xdot):
    (n, alpha, beta, x) = x
    (ndot, alphadot, betadot, xdot) = xdot
    f = _jacobi(n, alpha, beta, x, dx)
    df = _jacobi(n, alpha, beta, x, dx + 1)
    return f, df * xdot


@jit
@jnp.vectorize
def _binom(n, k):
    """Binomial coefficient.

    Implementation is only correct for positive integer n,k and n>=k

    Parameters
    ----------
    n : int, array-like
        number of things to choose from
    k : int, array-like
        number of things chosen

    Returns
    -------
    val : int, float, array-like
        number of possible combinations
    """
    # adapted from scipy:
    # https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/
    # scipy/special/orthogonal_eval.pxd#L68

    n, k = map(jnp.asarray, (n, k))

    def _binom_body_fun(i, b_n):
        b, n = b_n
        num = n + 1 - i
        den = i
        return (b * num / den, n)

    kx = k.astype(int)
    b, n = fori_loop(1, 1 + kx, _binom_body_fun, (1.0, n))
    return b


def polyder_vec(p, m, exact=False):
    """Vectorized version of polyder.

    For differentiating multiple polynomials of the same degree

    Parameters
    ----------
    p : ndarray, shape(N,M)
        polynomial coefficients. Each row is 1 polynomial, in descending powers of x,
        each column is a power of x
    m : int >=0
        order of derivative
    exact : bool
        Whether to use exact integer arithmetic (not compatible with JAX, but may be
        needed for very high degree polynomials)

    Returns
    -------
    der : ndarray, shape(N,M)
        polynomial coefficients for derivative in descending order

    """
    if exact:
        return _polyder_exact(p, m)
    else:
        return _polyder_jax(p, m)


def _polyder_exact(p, m):
    m = np.asarray(m, dtype=int)  # order of derivative
    p = np.atleast_2d(p)
    order = p.shape[1] - 1

    D = np.arange(order, -1, -1)
    num = np.array([factorial(i) for i in D], dtype=object)
    den = np.array([factorial(max(i - m, 0)) for i in D], dtype=object)
    D = (num // den).astype(p.dtype)

    p = np.roll(D * p, m, axis=1)
    idx = np.arange(p.shape[1])
    p = np.where(idx < m, 0, p)
    return p


@jit
def _polyder_jax(p, m):
    p = jnp.atleast_2d(jnp.asarray(p))
    m = jnp.asarray(m).astype(int)
    order = p.shape[1] - 1
    D = jnp.arange(order, -1, -1)

    def body(i, Di):
        return Di * jnp.maximum(D - i, 1)

    D = fori_loop(0, m, body, jnp.ones_like(D))

    p = jnp.roll(D * p, m, axis=1)
    idx = jnp.arange(p.shape[1])
    p = jnp.where(idx < m, 0, p)

    return p


def polyval_vec(p, x, prec=None):
    """Evaluate a polynomial at specific values.

    Vectorized for evaluating multiple polynomials of the same degree.

    Parameters
    ----------
    p : ndarray, shape(N,M)
        Array of coefficient for N polynomials of order M.
        Each row is one polynomial, given in descending powers of x.
    x : ndarray, shape(K,)
        A number, or 1d array of numbers at
        which to evaluate p. If greater than 1d it is flattened.
    prec : int, optional
        precision to use, in number of decimal places. Default is
        double precision (~16 decimals) which should be enough for
        most cases with L <= 24

    Returns
    -------
    y : ndarray, shape(N,K)
        polynomials evaluated at x.
        Each row corresponds to a polynomial, each column to a value of x

    """
    if prec is not None and prec > 18:
        return _polyval_exact(p, x, prec)
    else:
        return _polyval_jax(p, x)


def _polyval_exact(p, x, prec):
    import mpmath

    p = np.atleast_2d(p)
    x = np.atleast_1d(x).flatten()
    # TODO: possibly multithread this bit
    mpmath.mp.dps = prec
    y = np.array([np.asarray(mpmath.polyval(list(pi), x)) for pi in p])
    return y.astype(float)


@jit
def _polyval_jax(p, x):
    p = jnp.atleast_2d(jnp.asarray(p))
    x = jnp.atleast_1d(jnp.asarray(x)).flatten()
    npoly = p.shape[0]  # number of polynomials
    order = p.shape[1]  # order of polynomials
    nx = len(x)  # number of coordinates
    y = jnp.zeros((npoly, nx))

    def body(k, y):
        return y * x + jnp.atleast_2d(p[:, k]).T

    y = fori_loop(0, order, body, y)

    return y.astype(float)


def zernike_radial_coeffs(l, m, exact=True):
    """Polynomial coefficients for radial part of zernike basis.

    The for loop ranges from m to l+1 in steps of 2, as opposed to the
    formula in the zernike_eval notebook. This is to make the coeffs array in
    ascending powers of r, which is more natural for polynomial evaluation.
    So, one should substitute s=(l-k)/s in the formula in the notebook to get
    the coding implementation below.

                                 (-1)^((l-k)/2) * ((l+k)/2)!
    R_l^m(r) = sum_{k=m}^l  -------------------------------------
                             ((l-k)/2)! * ((k+m)/2)! * ((k-m)/2)!

    Parameters
    ----------
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    exact : bool
        whether to return exact coefficients with `object` dtype
        or return integer or floating point approximation

    Returns
    -------
    coeffs : ndarray
        Polynomial coefficients for Zernike polynomials, in descending powers of r.

    Notes
    -----
    Integer representation is exact up to l~54, so leaving `exact` arg as False
    can speed up evaluation with no loss in accuracy
    """
    l = np.atleast_1d(l).astype(int)
    m = np.atleast_1d(np.abs(m)).astype(int)
    lm = np.vstack([l, m]).T
    # for modest to large arrays, faster to find unique values and
    # only evaluate those
    lms, idx = np.unique(lm, return_inverse=True, axis=0)

    npoly = len(lms)
    lmax = np.max(lms[:, 0])
    coeffs = np.zeros((npoly, lmax + 1), dtype=object)
    lm_even = ((lms[:, 0] - lms[:, 1]) % 2 == 0)[:, np.newaxis]
    for ii in range(npoly):
        ll = lms[ii, 0]
        mm = lms[ii, 1]
        for s in range(mm, ll + 1, 2):
            # Zernike polynomials can also be written in the form of [1] which
            # states that the coefficients are given by the binomial coefficients
            # hence they are all integers. So, we can use exact arithmetic with integer
            # division instead of floating point division.
            # [1]https://en.wikipedia.org/wiki/Zernike_polynomials#Other_representations
            coeffs[ii, s] = ((-1) ** ((ll - s) // 2) * factorial((ll + s) // 2)) // (
                factorial((ll - s) // 2)
                * factorial((s + mm) // 2)
                * factorial((s - mm) // 2)
            )
    c = np.fliplr(np.where(lm_even, coeffs, 0))
    if not exact:
        try:
            c = c.astype(int)
        except OverflowError:
            c = c.astype(float)
    c = c[idx]
    return c


def zernike_radial_poly(r, l, m, dr=0, exact="auto"):
    """Radial part of zernike polynomials.

    Evaluates basis functions using numpy to
    exactly compute the polynomial coefficients
    and Horner's method for low resolution,
    or extended precision arithmetic for high resolution.
    Faster for low resolution, but not differentiable.

    Parameters
    ----------
    r : ndarray, shape(N,)
        radial coordinates to evaluate basis
    l : ndarray of int, shape(K,)
        radial mode number(s)
    m : ndarray of int, shape(K,)
        azimuthal mode number(s)
    dr : int
        order of derivative (Default = 0)
    exact : {"auto", True, False}
        Whether to use exact/extended precision arithmetic. Slower but more accurate.
        "auto" will use higher accuracy when needed.

    Returns
    -------
    y : ndarray, shape(N,K)
        basis function(s) evaluated at specified points

    """
    if exact == "auto":
        exact = np.max(l) > 54
    if exact:
        # this should give accuracy of ~1e-10 in the eval'd polynomials
        lmax = np.max(l)
        prec = int(0.4 * lmax + 8.4)
    else:
        prec = None
    coeffs = zernike_radial_coeffs(l, m, exact=exact)
    coeffs = polyder_vec(coeffs, dr, exact=exact)
    return polyval_vec(coeffs, r, prec=prec).T
