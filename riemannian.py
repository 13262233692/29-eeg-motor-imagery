import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _matrix_inv_sqrt(A: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _matrix_log(A: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T


def _matrix_exp(A: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(A)
    return eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T


def _symmat_to_vec(S: np.ndarray) -> np.ndarray:
    n = S.shape[0]
    rows, cols = np.triu_indices(n)
    vec = S[rows, cols].copy()
    off_diag = rows != cols
    vec[off_diag] *= np.sqrt(2)
    return vec


def _vec_to_symmat(vec: np.ndarray, n: int) -> np.ndarray:
    S = np.zeros((n, n))
    rows, cols = np.triu_indices(n)
    v = vec.copy()
    off_diag = rows != cols
    v[off_diag] /= np.sqrt(2)
    S[rows, cols] = v
    S[cols, rows] = v
    return S


def riemannian_distance(P: np.ndarray, Q: np.ndarray) -> float:
    eigvals = np.linalg.eigvalsh(_matrix_inv_sqrt(P) @ Q @ _matrix_inv_sqrt(P))
    eigvals = np.maximum(eigvals, 1e-12)
    return np.sqrt(np.sum(np.log(eigvals) ** 2))


def frechet_mean(
    covs: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    M = np.mean(covs, axis=0).copy()
    M = _ensure_spd(M)

    for _ in range(max_iter):
        M_inv_sqrt = _matrix_inv_sqrt(M)
        M_sqrt = _matrix_sqrt(M)

        S_sum = np.zeros_like(M)
        for C in covs:
            W = M_inv_sqrt @ C @ M_inv_sqrt
            S_sum += _matrix_log(W)
        S_mean = S_sum / len(covs)

        M_new = M_sqrt @ _matrix_exp(S_mean) @ M_sqrt
        M_new = _ensure_spd(M_new)

        dist = riemannian_distance(M, M_new)
        M = M_new

        if dist < tol:
            break

    return M


def _ensure_spd(C: np.ndarray) -> np.ndarray:
    C = (C + C.T) / 2
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 1e-10)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def tangent_space_project(
    covs: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    n_trials = len(covs)
    n_channels = covs.shape[1]
    ref_inv_sqrt = _matrix_inv_sqrt(reference)

    n_ts = n_channels * (n_channels + 1) // 2
    features = np.zeros((n_trials, n_ts))

    for i, C in enumerate(covs):
        W = ref_inv_sqrt @ C @ ref_inv_sqrt
        S = _matrix_log(W)
        features[i] = _symmat_to_vec(S)

    return features


def parallel_transport(
    C: np.ndarray,
    M_source: np.ndarray,
    M_target: np.ndarray,
) -> np.ndarray:
    M_s_inv_sqrt = _matrix_inv_sqrt(M_source)
    M_t_sqrt = _matrix_sqrt(M_target)
    return M_t_sqrt @ M_s_inv_sqrt @ C @ M_s_inv_sqrt @ M_t_sqrt


def batch_parallel_transport(
    covs: np.ndarray,
    M_source: np.ndarray,
    M_target: np.ndarray,
) -> np.ndarray:
    M_s_inv_sqrt = _matrix_inv_sqrt(M_source)
    M_t_sqrt = _matrix_sqrt(M_target)
    transported = np.zeros_like(covs)
    for i, C in enumerate(covs):
        transported[i] = M_t_sqrt @ M_s_inv_sqrt @ C @ M_s_inv_sqrt @ M_t_sqrt
    transported = (transported + np.transpose(transported, (0, 2, 1))) / 2
    return transported


def compute_subject_means(
    covs: np.ndarray,
    subject_ids: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict:
    unique_subjects = np.unique(subject_ids)
    subject_means = {}
    for s in unique_subjects:
        mask = subject_ids == s
        subject_means[s] = frechet_mean(covs[mask], max_iter=max_iter, tol=tol)
    return subject_means


def cross_subject_align(
    covs: np.ndarray,
    subject_ids: np.ndarray,
    subject_means: dict,
    reference_mean: np.ndarray,
) -> np.ndarray:
    aligned = np.zeros_like(covs)
    for i, (C, s) in enumerate(zip(covs, subject_ids)):
        if s in subject_means:
            aligned[i] = parallel_transport(C, subject_means[s], reference_mean)
        else:
            aligned[i] = C
    aligned = (aligned + np.transpose(aligned, (0, 2, 1))) / 2
    return aligned


class SPDCovariance(BaseEstimator, TransformerMixin):

    def __init__(self, shrinkage: float = 1e-6):
        self.shrinkage = shrinkage

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_trials, n_channels, n_samples), "
                f"got shape {X.shape}"
            )
        n_trials, n_channels, _ = X.shape
        covs = np.zeros((n_trials, n_channels, n_channels))
        for i in range(n_trials):
            C = np.cov(X[i])
            C /= np.trace(C)
            C = (C + C.T) / 2
            C += self.shrinkage * np.eye(n_channels)
            covs[i] = C
        return covs


class TangentSpace(BaseEstimator, TransformerMixin):

    def __init__(self, max_iter: int = 50, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y=None):
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_trials, n_channels, n_channels), "
                f"got shape {X.shape}"
            )
        self.reference_ = frechet_mean(X, max_iter=self.max_iter, tol=self.tol)
        self.n_channels_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "reference_"):
            raise RuntimeError("TangentSpace has not been fitted. Call fit() first.")
        return tangent_space_project(X, self.reference_)

    def get_reference(self) -> np.ndarray:
        if not hasattr(self, "reference_"):
            raise RuntimeError("TangentSpace has not been fitted. Call fit() first.")
        return self.reference_
