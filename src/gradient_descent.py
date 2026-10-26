"""Matrix-form gradient descent for the linear model ``y_hat = X @ m + b``.

The implementation is divided into four parts so that each mathematical step
can be explained and tested independently:

1. prediction and mean squared error,
2. the analytical gradient,
3. an independent SciPy numerical-gradient check, and
4. the gradient-descent update loop with a complete history.
"""

import numpy as np
from scipy.differentiate import derivative


def predict(X, m, b):
    """Return predictions for all rows of ``X`` using weights ``m`` and bias ``b``."""
    # The scalar bias is added to every sample, which is b * 1 in matrix notation.
    return X @ m + b * np.ones(X.shape[0])


def mse(X, y, m, b):
    """Calculate J(m, b) = (1/n) * sum((y_hat - y)^2)."""
    error = predict(X, m, b) - y
    return float(error @ error / len(y))

def analytic_gradient(X, y, m, b):
    """Return dJ/dm and dJ/db for the mean squared-error cost."""
    n = len(y)
    error = predict(X, m, b) - y

    # These are the required matrix derivatives:
    # dJ/dm = (2/n) X.T @ error and dJ/db = (2/n) 1.T @ error.
    grad_m = (2 / n) * X.T @ error
    grad_b = (2 / n) * float(np.ones(n) @ error)
    return grad_m, grad_b
