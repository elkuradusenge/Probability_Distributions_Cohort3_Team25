"""Reusable plotting helpers (DRY): one function per figure type."""
import numpy as np
import matplotlib.pyplot as plt
from em_gmm import gaussian_pdf


def plot_mixture(x, mu, var, pi, labels, title, test_height=None, path=None):
    grid = np.linspace(x.min() - 2, x.max() + 2, 400)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(x, bins=30, density=True, alpha=0.35, color="grey", label="all heights (unlabelled)")
    for k, c in enumerate(["tab:blue", "tab:red"]):
        ax.plot(grid, pi[k] * gaussian_pdf(grid, mu[k], var[k]), c=c, lw=2, label=f"{labels[k]}  μ={mu[k]:.2f}")
    mix = sum(pi[k] * gaussian_pdf(grid, mu[k], var[k]) for k in range(2))
    ax.plot(grid, mix, "k--", lw=1.5, label="mixture")
    ax.axvline(x.mean(), c="green", ls=":", label=f"global mean {x.mean():.2f}")
    if test_height is not None:
        ax.axvline(test_height, c="purple", lw=2, label=f"test height {test_height}")
    ax.set(title=title, xlabel="height (inches)", ylabel="density"); ax.legend(fontsize=8)
    return _finish(fig, path)


def plot_lines(xs, series, labels, title, ylabel, xlabel="iteration", path=None, logy=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    for ys, lab in zip(series, labels):
        ax.plot(xs, ys, marker="o", ms=3, label=lab)
    if logy:
        ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=0.3); ax.legend()
    return _finish(fig, path)


def plot_bars(labels, values, title, ylabel, colors, path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color=colors); ax.axhline(0.5, c="k", ls="--", lw=1)
    ax.set(title=title, ylabel=ylabel, ylim=(0, 1)); plt.setp(ax.get_xticklabels(), rotation=30)
    return _finish(fig, path)


def _finish(fig, path):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=140)
    plt.show()
    return fig
