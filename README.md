# Formative 3 — Probability Distributions, Bayes & Gradient Descent

**Group:** _Formative 3 Group 25_  <br> 
**Group Members:**  <br> 
 _Teta Nelly_ ,  <br>
 _Elie Kuradusenge_ ,  <br>
_Fabrice Mbarushimana_

## Repo layout
| Path | Purpose |
|---|---|
| `Formative3_Notebook.ipynb` | Everything, executed, with figures (open this during the presentation) |
| `src/em_gmm.py` | Part 1: EM for a 2-component Gaussian mixture (NumPy only) |
| `src/bayes.py` | Part 2: Bayes' theorem on IMDb (stdlib only: `csv`, `re`) |
| `src/gradient_descent.py` | Parts 3–4: matrix-form GD, SciPy derivative function |
| `src/plotting.py` | Reusable plot helpers (DRY) |
| `part3_gradient_descent_calculations.pdf` | Part 3 manual calculations (3 updates, one per member) |
| `data/` | `GaltonFamilies.csv` (included); `IMDB_Dataset.csv` from Kaggle goes here (not pushed, 66 MB) |
 
Run: `pip install numpy pandas scipy matplotlib jupyter` → `jupyter notebook Formative3_Notebook.ipynb`.
Download the IMDb file from Kaggle, rename it `IMDB_Dataset.csv`, and place it in `data/` before running Part 2.


## Part 1 — EM (talking points)
* **Data:** 934 child heights + 205 mother heights (one per family), labels hidden.
* **Why not split at the global mean (66.25)?** Groups overlap, so a hard cut misclassifies points on the wrong side, biases both
  pile means (63.58 / 69.47 vs EM 64.30 / 69.79), ignores unequal spreads/sizes, and gives no uncertainty. EM gives soft
  responsibilities.
* **E-step:** r_ik = π_k N(x_i|μ_k,σ_k²) / Σ_j π_j N(x_i|μ_j,σ_j²).
  **M-step:** N_k = Σ r_ik; μ_k = Σ r_ik x_i / N_k; σ_k² = Σ r_ik (x_i−μ_k)² / N_k; π_k = N_k / N.
* **Stop** when the log-likelihood changes by < 1e-6 (376 iterations here). Log-likelihood never decreases.
* **Init:** μ = mean ∓ 1 SD, σ² = data variance, π = 0.5/0.5.

| Iter | μ1 | μ2 | σ1² | σ2² | π1 | π2 | Log-lik |
|---|---|---|---|---|---|---|---|
| 0 | 62.7052 | 69.7989 | 12.5802 | 12.5802 | 0.5000 | 0.5000 | −3190.3325 |
| 1 | 64.0693 | 68.5283 | 6.5015 | 8.7695 | 0.5105 | 0.4895 | −3041.3709 |
| 2 | 64.0394 | 68.5861 | 6.0192 | 8.8886 | 0.5133 | 0.4867 | −3039.7574 |

* **Be upfront:** EM finds *shorter* vs *taller* groups (sons are ~69 in, daughters ~64 in like mothers), not exactly
  mothers vs children: 195/205 mothers land in the shorter component along with 421 daughters.
* **Live demo:** set `TEST_HEIGHT` in the notebook and run the cell → prints P(Component 1) and P(Component 2). Example 66.0 in → 0.835 / 0.165.
  The assignment text says "Child vs Basketball Player"; with this dataset the two classes are the shorter and taller components.

## Part 2 — Bayes
Computed **P(Positive | keyword)** only. Prior 0.5. Posterior = likelihood × prior / marginal.

| keyword | P(kw\|Pos) | P(kw) | P(Pos\|kw) |
|---|---|---|---|
| excellent | 0.1146 | 0.0710 | 0.8073 |
| wonderful | 0.0902 | 0.0555 | 0.8120 |
| amazing | 0.0671 | 0.0431 | 0.7783 |
| perfect | 0.0815 | 0.0532 | 0.7664 |
| worst | 0.0164 | 0.0886 | 0.0924 |
| awful | 0.0112 | 0.0575 | 0.0977 |
| boring | 0.0236 | 0.0610 | 0.1938 |
| waste | 0.0070 | 0.0507 | 0.0687 |

Counting rule: a review counts once if the word appears at least once. Posterior matches the direct frequency (asserted in the notebook).

## Parts 3–4 — Gradient descent
ŷ = Xm + b·1, J = (1/n)eᵀe, ∇ₘJ = (2/n)Xᵀe, ∂J/∂b = (2/n)1ᵀe. m₀=[−1,2], b₀=1, α=0.01 (given b=[1,1] is read as bias 1 on both samples).
First step: ŷ=[6,17], J=61, ∇ₘ=[45,113], ∂b=12 → m=[−1.45,0.87], b=0.88, J=6.845. `scipy_gradient` (SciPy `differentiate.derivative`) matches the analytic gradient.
Plots: parameters vs iteration, MSE vs iteration (log scale).

### Part 3 — who did which update
Handwritten in `part3_gradient_descent_calculations.pdf`, one update per member (each starts from the previous result):

| Update | Member | Pages | Result |
|---|---|---|---|
| 1 | Teta Nelly | 1–2 | m=[−1.45, 0.87], b=0.88, J=6.845 |
| 2 | Elie Kuradusenge | 3–5 | m=[−1.3316, 1.1808], b=0.9318, J=2.7238 |
| 3 | Fabrice Mbarushimana | 6–8 | m=[−1.369562, 1.095182], b=0.93624, J=2.3904 |

* **Trend:** J falls every update (61 → 6.845 → 2.7238 → 2.3904), so m and b move in a direction that reduces the error.
