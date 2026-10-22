"""Restarted, right-preconditioned GMRES with classical Gram-Schmidt reorthogonalisation."""
import numpy as np
from scipy.sparse.linalg import LinearOperator


def givens(a: float, b: float) -> tuple[float, float]:
    if b == 0.0:
        return 1.0, 0.0
    r = np.hypot(a, b)
    return a / r, b / r


def cgs2(V: np.ndarray, w: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Two passes of classical Gram-Schmidt against the first k+1 basis vectors."""
    h = V[:, : k + 1].T @ w
    w = w - V[:, : k + 1] @ h
    correction = V[:, : k + 1].T @ w
    w = w - V[:, : k + 1] @ correction
    return w, h + correction


def gmres(A: LinearOperator, b: np.ndarray, M: LinearOperator, x0=None,
          tol: float = 1e-10, restart: int = 50, maxiter: int = 1000, callback=None):
    n = b.shape[0]
    x = np.zeros(n) if x0 is None else x0.copy()
    b_norm = np.linalg.norm(b)
    if b_norm == 0.0:
        return x, 0

    total = 0
    while total < maxiter:
        r = b - A @ x
        beta = np.linalg.norm(r)
        if beta / b_norm < tol:
            return x, total

        V = np.zeros((n, restart + 1))
        H = np.zeros((restart + 1, restart))
        cs = np.zeros(restart)
        sn = np.zeros(restart)
        g = np.zeros(restart + 1)
        V[:, 0] = r / beta
        g[0] = beta

        for k in range(restart):
            z = M @ V[:, k]
            w, h = cgs2(V, A @ z, k)
            H[: k + 1, k] = h
            H[k + 1, k] = np.linalg.norm(w)
            if H[k + 1, k] > 1e-14:
                V[:, k + 1] = w / H[k + 1, k]

            for j in range(k):
                temp = cs[j] * H[j, k] + sn[j] * H[j + 1, k]
                H[j + 1, k] = -sn[j] * H[j, k] + cs[j] * H[j + 1, k]
                H[j, k] = temp

            cs[k], sn[k] = givens(H[k, k], H[k + 1, k])
            H[k, k] = cs[k] * H[k, k] + sn[k] * H[k + 1, k]
            H[k + 1, k] = 0.0
            g[k + 1] = -sn[k] * g[k]
            g[k] = cs[k] * g[k]

            total += 1
            residual = abs(g[k + 1]) / b_norm
            if callback is not None:
                callback(total, residual)
            if residual < tol or total >= maxiter:
                k += 1
                break
        else:
            k = restart

        y = np.linalg.solve(np.triu(H[:k, :k]), g[:k])
        x = x + M @ (V[:, :k] @ y)

    return x, total


def amg_vcycle(levels, b: np.ndarray, level: int = 0, sweeps: int = 2) -> np.ndarray:
    A, P, smoother = levels[level]
    if level == len(levels) - 1:
        return np.linalg.solve(A.toarray(), b)
    x = smoother(A, b, np.zeros_like(b), sweeps)
    coarse_residual = P.T @ (b - A @ x)
    x = x + P @ amg_vcycle(levels, coarse_residual, level + 1, sweeps)
    return smoother(A, b, x, sweeps)
