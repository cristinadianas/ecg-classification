"""Fit one binary SVM on a precomputed kernel (worker for src/training/train_svm.py).

Kept in its own module with light imports: train_svm runs it in worker processes,
which then only load scikit-learn and NumPy, not PyTorch.
"""

import time

from sklearn.svm import SVC


def fit_binary(k_train, y, k_val, k_test, c: float, cache_mb: int):
    """-> (validation scores, test scores, number of support vectors, fit seconds)."""
    svm = SVC(kernel="precomputed", C=c, class_weight="balanced", cache_size=cache_mb)
    t0 = time.time()
    svm.fit(k_train, y)
    return svm.decision_function(k_val), svm.decision_function(k_test), int(svm.n_support_.sum()), time.time() - t0
