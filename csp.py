import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):

    def __init__(self, n_components: int = 4, log: bool = True):
        self.n_components = n_components
        self.log = log

    def _compute_normalized_covariance(self, epoch_data: np.ndarray) -> np.ndarray:
        n_trials, n_channels, n_samples = epoch_data.shape
        cov_sum = np.zeros((n_channels, n_channels))
        for trial in epoch_data:
            cov = np.cov(trial)
            cov_sum += cov / np.trace(cov)
        return cov_sum / n_trials

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_trials, n_channels, n_samples), "
                f"got shape {X.shape}"
            )
        if len(X) != len(y):
            raise ValueError(
                f"Number of trials ({len(X)}) does not match "
                f"number of labels ({len(y)})"
            )

        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(
                f"CSP requires exactly 2 classes, got {len(unique_labels)}: "
                f"{unique_labels}"
            )

        n_channels = X.shape[1]

        data_class_0 = X[y == unique_labels[0]]
        data_class_1 = X[y == unique_labels[1]]

        sigma_0 = self._compute_normalized_covariance(data_class_0)
        sigma_1 = self._compute_normalized_covariance(data_class_1)

        eigenvalues, eigenvectors = eigh(sigma_0, sigma_0 + sigma_1)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_idx]
        eigenvalues = eigenvalues[sorted_idx]

        n_pick = min(self.n_components, n_channels)
        half = n_pick // 2
        pick_indices = np.concatenate(
            [np.arange(0, half), np.arange(n_channels - (n_pick - half), n_channels)]
        )

        self.filters_ = eigenvectors[:, pick_indices].T
        self.patterns_ = np.linalg.pinv(self.filters_).T
        self.eigenvalues_ = eigenvalues
        self.eigenvalues_picked_ = eigenvalues[pick_indices]
        self.n_components_ = n_pick
        self.pick_indices_ = pick_indices
        self.n_channels_ = n_channels
        self.classes_ = unique_labels

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "filters_"):
            raise RuntimeError("CSP has not been fitted. Call fit() first.")
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_trials, n_channels, n_samples), "
                f"got shape {X.shape}"
            )
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                f"Channel mismatch: data has {X.shape[1]} channels, "
                f"but CSP was fitted on {self.n_channels_} channels"
            )

        transformed = np.einsum("ij,kjl->kil", self.filters_, X)

        variances = np.var(transformed, axis=2)

        if self.log:
            features = np.log(variances + 1e-10)
        else:
            features = variances

        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_spatial_filters(self) -> np.ndarray:
        if not hasattr(self, "filters_"):
            raise RuntimeError("CSP has not been fitted. Call fit() first.")
        return self.filters_

    def get_spatial_patterns(self) -> np.ndarray:
        if not hasattr(self, "patterns_"):
            raise RuntimeError("CSP has not been fitted. Call fit() first.")
        return self.patterns_
