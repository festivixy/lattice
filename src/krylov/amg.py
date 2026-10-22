"""Smoothed-aggregation algebraic multigrid setup: strength, aggregation, prolongation."""
import numpy as np
import scipy.sparse as sp


def strength(A: sp.csr_matrix, theta: float = 0.08) -> sp.csr_matrix:
    """Keep the off-diagonal entries that are strong relative to the diagonal."""
    diag = np.abs(A.diagonal())
    coo = A.tocoo()
    keep = (coo.row != coo.col) & (
        np.abs(coo.data) >= theta * np.sqrt(diag[coo.row] * diag[coo.col])
    )
    return sp.csr_matrix((np.ones(keep.sum()), (coo.row[keep], coo.col[keep])), shape=A.shape)


def aggregate(S: sp.csr_matrix) -> np.ndarray:
    n = S.shape[0]
    labels = -np.ones(n, dtype=np.int64)
    count = 0
    for i in range(n):  # pass 1: seed aggregates from untouched neighbourhoods
        neighbours = S.indices[S.indptr[i]:S.indptr[i + 1]]
        if labels[i] == -1 and np.all(labels[neighbours] == -1):
            labels[i] = count
            labels[neighbours] = count
            count += 1
    for i in np.flatnonzero(labels == -1):  # pass 2: attach leftovers to a neighbour
        neighbours = S.indices[S.indptr[i]:S.indptr[i + 1]]
        assigned = neighbours[labels[neighbours] >= 0]
        labels[i] = labels[assigned[0]] if assigned.size else count
        count += int(not assigned.size)
    return labels


def prolongator(A: sp.csr_matrix, labels: np.ndarray, omega: float = 4 / 3) -> sp.csr_matrix:
    n, m = A.shape[0], labels.max() + 1
    tentative = sp.csr_matrix((np.ones(n), (np.arange(n), labels)), shape=(n, m))
    norms = np.sqrt(np.asarray(tentative.multiply(tentative).sum(axis=0))).ravel()
    tentative = tentative @ sp.diags(1 / norms)
    dinv = sp.diags(1 / A.diagonal())
    rho = spectral_radius(dinv @ A)
    return (sp.eye(n) - (omega / rho) * dinv @ A) @ tentative


def spectral_radius(M: sp.spmatrix, iterations: int = 20) -> float:
    v = np.random.default_rng(0).standard_normal(M.shape[0])
    for _ in range(iterations):
        v = M @ v
        v /= np.linalg.norm(v)
    return float(v @ (M @ v))


def jacobi(A, b, x, sweeps: int, omega: float = 2 / 3):
    dinv = 1 / A.diagonal()
    for _ in range(sweeps):
        x = x + omega * dinv * (b - A @ x)
    return x


def setup(A: sp.csr_matrix, max_levels: int = 12, coarse_size: int = 500):
    levels = []
    while len(levels) < max_levels - 1 and A.shape[0] > coarse_size:
        P = prolongator(A, aggregate(strength(A)))
        levels.append((A, P, jacobi))
        A = (P.T @ A @ P).tocsr()
    levels.append((A, None, None))
    return levels
