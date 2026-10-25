"""Two-component Gaussian Mixture Model fitted with Expectation-Maximization, from scratch (NumPy only)."""
import numpy as np


def gaussian_pdf(x, mu, var):
    """Univariate normal density N(x | mu, var)."""
    return np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def weighted_densities(x, mu, var, pi):
    """Matrix (n, K) of pi_k * N(x_i | mu_k, var_k)."""
    return np.column_stack([pi[k] * gaussian_pdf(x, mu[k], var[k]) for k in range(len(mu))])


def log_likelihood(x, mu, var, pi):
    """sum_i log( sum_k pi_k N(x_i | mu_k, var_k) )."""
    return float(np.sum(np.log(weighted_densities(x, mu, var, pi).sum(axis=1))))


def e_step(x, mu, var, pi):
    """Responsibilities r_ik = pi_k N(x_i|k) / sum_j pi_j N(x_i|j)."""
    w = weighted_densities(x, mu, var, pi)
    return w / w.sum(axis=1, keepdims=True)


def m_step(x, resp):
    """Update (mu, var, pi) using the responsibilities as soft counts."""
    n_k = resp.sum(axis=0)
    mu = (resp * x[:, None]).sum(axis=0) / n_k
    var = (resp * (x[:, None] - mu) ** 2).sum(axis=0) / n_k
    pi = n_k / len(x)
    return mu, var, pi


def fit_em(x, mu_init, var_init, pi_init, tol=1e-6, max_iter=500):
    """Run EM until the log-likelihood changes by less than `tol`.
    Returns final (mu, var, pi) and a history list of dicts (iteration 0 = initial state)."""
    mu, var, pi = (np.asarray(a, dtype=float) for a in (mu_init, var_init, pi_init))
    history = [dict(iter=0, mu=mu.copy(), var=var.copy(), pi=pi.copy(),
                    ll=log_likelihood(x, mu, var, pi))]
    for it in range(1, max_iter + 1):
        resp = e_step(x, mu, var, pi)
        mu, var, pi = m_step(x, resp)
        ll = log_likelihood(x, mu, var, pi)
        history.append(dict(iter=it, mu=mu.copy(), var=var.copy(), pi=pi.copy(), ll=ll))
        if abs(ll - history[-2]["ll"]) < tol:
            break
    return mu, var, pi, history


def posterior(h, mu, var, pi):
    """Exact posterior P(component k | height h) via Bayes' rule."""
    return e_step(np.atleast_1d(np.asarray(h, dtype=float)), mu, var, pi)[0]
