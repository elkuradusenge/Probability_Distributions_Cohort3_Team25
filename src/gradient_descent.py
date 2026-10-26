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

def scipy_gradient(cost, params):
    """Differentiate a flat-parameter cost function with SciPy.

    ``cost`` receives one parameter vector, for example::

        cost = lambda p: mse(X, y, p[:-1], p[-1])

    The result contains one numerical partial derivative per parameter.
    """
    params = np.asarray(params, dtype=float)
    grad = np.zeros_like(params)

    for i in range(len(params)):
        # Vary only parameter i while keeping all other parameters fixed.
        def one_dim(t, i=i):
            parameter_matrix = np.tile(params[:, None], (1, np.size(t))).astype(float)
            parameter_matrix[i, :] = t
            return np.array(
                [cost(parameter_matrix[:, j]) for j in range(parameter_matrix.shape[1])]
            ).reshape(np.shape(t))

        grad[i] = derivative(one_dim, params[i], initial_step=0.5).df

    return grad


def gradient_descent(X, y, m0, b0, lr, n_iter):
    """Run gradient descent and record every intermediate value.

    History entry 0 is the starting point. Each later entry stores the
    gradient used for that update, the new parameters, predictions, errors,
    and cost, making the calculation easy to inspect or plot.
    """
    m = np.asarray(m0, dtype=float).copy()
    b = float(b0)
    y_hat = predict(X, m, b)
    history = [
        dict(
            k=0,
            m=m.copy(),
            b=b,
            y_hat=y_hat,
            error=y_hat - y,
            cost=mse(X, y, m, b),
            grad_m=None,
            grad_b=None,
        )
    ]

    for k in range(1, n_iter + 1):
        grad_m, grad_b = analytic_gradient(X, y, m, b)

        # Move parameters opposite to the gradient by the learning rate.
        m = m - lr * grad_m
        b = b - lr * grad_b

        y_hat = predict(X, m, b)
        history.append(
            dict(
                k=k,
                m=m.copy(),
                b=b,
                y_hat=y_hat,
                error=y_hat - y,
                cost=mse(X, y, m, b),
                grad_m=grad_m,
                grad_b=grad_b,
            )
        )

    return m, b, history