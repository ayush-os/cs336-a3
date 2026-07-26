"""Section 2: IsoFLOPs scaling laws (chinchilla_isoflops).

Fits N_opt(C) and D_opt(C) power laws from data/isoflops_curves.json using the
Chinchilla IsoFLOPs method (Hoffmann et al. 2022), taking the lowest-loss run
per compute budget as the optimum rather than fitting a quadratic.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "isoflops_curves.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "data"

TARGET_BUDGETS = [1e23, 1e24]


def fit_power_law(c: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit y = a * C^b via linear regression in log-log space. Returns (a, b)."""
    log_c, log_y = np.log(c), np.log(y)
    b, log_a = np.polyfit(log_c, log_y, 1)
    return float(np.exp(log_a)), float(b)


def main() -> None:
    runs = json.loads(DATA_PATH.read_text())

    budgets = sorted({r["compute_budget"] for r in runs})
    c_points, n_opt_points, d_opt_points = [], [], []
    for c in budgets:
        runs_c = [r for r in runs if r["compute_budget"] == c]
        best = min(runs_c, key=lambda r: r["final_loss"])
        n_opt = best["parameters"]
        d_opt = c / (6 * n_opt)
        c_points.append(c)
        n_opt_points.append(n_opt)
        d_opt_points.append(d_opt)

    c_points = np.array(c_points)
    n_opt_points = np.array(n_opt_points)
    d_opt_points = np.array(d_opt_points)

    a_n, b_n = fit_power_law(c_points, n_opt_points)
    a_d, b_d = fit_power_law(c_points, d_opt_points)

    print(f"N_opt(C) = {a_n:.4g} * C^{b_n:.4f}")
    print(f"D_opt(C) = {a_d:.4g} * C^{b_d:.4f}")
    print(f"Check: a_n * a_d ~= 1/6 -> {a_n * a_d:.4g} (should be close to {1 / 6:.4g})")
    print(f"Check: b_n + b_d ~= 1 -> {b_n + b_d:.4f}")

    for c_target in TARGET_BUDGETS:
        n_pred = a_n * c_target**b_n
        d_pred = a_d * c_target**b_d
        print(
            f"C={c_target:.0e}: N_opt≈{n_pred:.3e} params, D_opt≈{d_pred:.3e} tokens "
            f"(check: 6*N*D={6 * n_pred * d_pred:.3e})"
        )

    extrap_c = np.geomspace(c_points.min(), max(TARGET_BUDGETS) * 2, 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(c_points, n_opt_points, "o", label="Observed $N_{opt}(C_i)$")
    ax.loglog(extrap_c, a_n * extrap_c**b_n, "-", label=f"Fit: $N \\propto C^{{{b_n:.3f}}}$")
    for c_target in TARGET_BUDGETS:
        ax.axvline(c_target, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Compute budget C (FLOPs)")
    ax.set_ylabel("Optimal model size N (params)")
    ax.set_title("Compute-optimal model size vs. compute budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "isoflops_n_opt.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(c_points, d_opt_points, "o", label="Observed $D_{opt}(C_i)$")
    ax.loglog(extrap_c, a_d * extrap_c**b_d, "-", label=f"Fit: $D \\propto C^{{{b_d:.3f}}}$")
    for c_target in TARGET_BUDGETS:
        ax.axvline(c_target, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("Compute budget C (FLOPs)")
    ax.set_ylabel("Optimal dataset size D (tokens)")
    ax.set_title("Compute-optimal dataset size vs. compute budget")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "isoflops_d_opt.png", dpi=150)

    print(f"\nSaved plots to {OUT_DIR / 'isoflops_n_opt.png'} and {OUT_DIR / 'isoflops_d_opt.png'}")


if __name__ == "__main__":
    main()
