"""
Default Probability Simulation — Single-Factor Gaussian Copula
==============================================================
Simulates:  X_i = r_i * Y + sqrt(1 - r_i^2) * eps_i

Where:
  Y     ~ N(0,1)     common / systematic factor
  eps_i ~ N(0,1)     idiosyncratic noise for firm i
  r_i                factor loading (correlation with systematic factor)
  X_i                latent asset value — firm i defaults when X_i < barrier
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ────────────────────────────────────────────────────
NUM_ASSETS      = 5
NUM_SIMULATIONS = 50_000
DEFAULT_BARRIER = -1.5          # Φ(barrier) ≈ individual default prob
FIRM_NAMES      = [f"Firm {chr(65 + i)}" for i in range(NUM_ASSETS)]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Core functions ───────────────────────────────────────────────────
def generate_corr_normals(omega):
    """Cholesky factor of the correlation matrix."""
    L = np.linalg.cholesky(omega)
    return L


def generate_omega(size):
    """
    Generate a positive-definite correlation matrix.
    Uses eigenvalue clipping to guarantee Cholesky will succeed.
    """
    omega = np.diag(np.ones(size))
    for i in range(size):
        for j in range(i + 1, size):
            omega[i, j] = np.random.uniform(-0.3, 0.8)
            omega[j, i] = omega[i, j]

    # Nearest PD correction
    eigvals, eigvecs = np.linalg.eigh(omega)
    eigvals = np.maximum(eigvals, 1e-6)
    omega = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d_inv = np.diag(1.0 / np.sqrt(np.diag(omega)))
    omega = d_inv @ omega @ d_inv
    return omega


def generate_systematic_factor(num_sims):
    """Draw the common systematic factor Y ~ N(0,1)."""
    return np.random.normal(0, 1, size=num_sims)


def idiosyncratic_noise(num_sims, num_assets):
    """Independent noise eps_i ~ N(0,1) for each firm."""
    return np.random.normal(0, 1, size=(num_sims, num_assets))


def generate_factor_loadings(num_assets):
    """Random factor loadings r_i ∈ (0.2, 0.8) for each firm."""
    return np.random.uniform(0.2, 0.8, size=num_assets)


def simulate_latent_variables(r, Y, eps):
    """
    X_i = r_i * Y + sqrt(1 - r_i^2) * eps_i

    Parameters
    ----------
    r   : (num_assets,)       factor loadings
    Y   : (num_sims,)         systematic factor
    eps : (num_sims, num_assets)  idiosyncratic noise

    Returns
    -------
    X   : (num_sims, num_assets)  latent asset values
    """
    systematic  = r[None, :] * Y[:, None]                # (num_sims, num_assets)
    idio        = np.sqrt(1 - r**2)[None, :] * eps       # (num_sims, num_assets)
    X = systematic + idio
    return X


# ── Main ─────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    r   = generate_factor_loadings(NUM_ASSETS)
    Y   = generate_systematic_factor(NUM_SIMULATIONS)
    eps = idiosyncratic_noise(NUM_SIMULATIONS, NUM_ASSETS)

    X = simulate_latent_variables(r, Y, eps)             # (num_sims, num_assets)
    defaults = (X < DEFAULT_BARRIER).astype(int)         # 1 = default

    # ── Print results ────────────────────────────────────────────────
    print("Factor loadings r_i:")
    for name, ri in zip(FIRM_NAMES, r):
        print(f"  {name}: r = {ri:.3f}")
    print()

    marginal_pd = defaults.mean(axis=0)
    print(f"Default barrier:  {DEFAULT_BARRIER}")
    print(f"Simulations:      {NUM_SIMULATIONS:,}\n")
    print("Marginal default probabilities:")
    for name, pd in zip(FIRM_NAMES, marginal_pd):
        print(f"  {name}: {pd:.4f}")

    joint_pd = np.all(defaults, axis=1).mean()
    print(f"\nJoint default prob (all {NUM_ASSETS} firms): {joint_pd:.6f}")

    # ── Plotting ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colours = plt.cm.Set2(np.linspace(0, 1, NUM_ASSETS))

    # 1) Latent variable distributions
    ax = axes[0, 0]
    for i, name in enumerate(FIRM_NAMES):
        ax.hist(X[:, i], bins=80, alpha=0.45, color=colours[i],
                label=name, density=True)
    ax.axvline(DEFAULT_BARRIER, color="red", ls="--", lw=1.5,
               label="Default barrier")
    ax.set_title("Latent Variable Distributions  X_i", fontweight="bold")
    ax.set_xlabel("X_i")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2) Marginal default probabilities
    ax = axes[0, 1]
    bars = ax.bar(FIRM_NAMES, marginal_pd, color=colours[:NUM_ASSETS],
                  edgecolor="black", linewidth=0.5)
    ax.set_title("Marginal Default Probability", fontweight="bold")
    ax.set_ylabel("P(Default)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, pd in zip(bars, marginal_pd):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{pd:.3f}", ha="center", va="bottom", fontsize=9)

    # 3) Systematic vs idiosyncratic decomposition for one firm
    ax = axes[1, 0]
    firm_idx = 0
    sys_part  = r[firm_idx] * Y
    idio_part = np.sqrt(1 - r[firm_idx]**2) * eps[:, firm_idx]
    ax.hist(sys_part, bins=80, alpha=0.5, color="steelblue", density=True,
            label=f"Systematic  r·Y  (r={r[firm_idx]:.2f})")
    ax.hist(idio_part, bins=80, alpha=0.5, color="coral", density=True,
            label=f"Idiosyncratic  √(1−r²)·ε")
    ax.hist(X[:, firm_idx], bins=80, alpha=0.35, color="black", density=True,
            label=f"Combined  X ({FIRM_NAMES[firm_idx]})")
    ax.set_title(f"Decomposition — {FIRM_NAMES[firm_idx]}", fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4) Scatter: two firms showing correlation via shared Y
    ax = axes[1, 1]
    i, j = 0, 2
    both_default = (defaults[:, i] == 1) & (defaults[:, j] == 1)
    ax.scatter(X[~both_default, i], X[~both_default, j],
               s=2, alpha=0.1, color="steelblue", label="No joint default")
    ax.scatter(X[both_default, i], X[both_default, j],
               s=8, alpha=0.6, color="red", label="Both default")
    ax.axhline(DEFAULT_BARRIER, color="grey", ls="--", lw=0.8)
    ax.axvline(DEFAULT_BARRIER, color="grey", ls="--", lw=0.8)
    ax.set_xlabel(FIRM_NAMES[i])
    ax.set_ylabel(FIRM_NAMES[j])
    implied_corr = r[i] * r[j]
    ax.set_title(f"Joint Defaults  (implied ρ = {implied_corr:.2f})",
                 fontweight="bold")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)

    plt.suptitle("X_i = r_i · Y + √(1 − r_i²) · ε_i", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join(SCRIPT_DIR, "default_sim_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()