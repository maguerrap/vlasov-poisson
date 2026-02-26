from typing import Callable

import jax
import jax.numpy as jnp

import matplotlib
from matplotlib import figure, axes
import matplotlib.pyplot as plt

from .jax_vp_solver import Mesh, VlasovPoissonSolver


matplotlib.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})


Array = jax.Array


# ===== Auxiliary Functions =====

def external_electric_field(
    ak: Array,
    mesh: Mesh,
    k_0: float
) -> Array:
    """
    Compute the external electric field H(x):

        H(x; a_k, b_k) = ∑_k a_k cos(k₀ k x) + b_k sin(k₀ k x).
    """
    N = ak.shape[1]
    k = jnp.arange(1, N + 1)
    cos_term = ak[0, :] @ jnp.cos(k_0 * k[:, None] * mesh.xs)
    sin_term = ak[1, :] @ jnp.sin(k_0 * k[:, None] * mesh.xs)
    return cos_term + sin_term


# ===== Cost Functions =====

# --- KL Divergence ---

def kl_divergence(
    f_T: Array,
    solver: VlasovPoissonSolver,
    eps: float = 1e-12
) -> Array:
    """
    Compute the KL divergence:

        KL(f_T || f_eq) = ∫∫ f_T log(f_T / f_eq) dx dv.
    """
    norm_final = jnp.trapezoid(jnp.trapezoid(f_T, solver.mesh.vs, axis=1),
                               solver.mesh.xs) + eps
    norm_eq = jnp.trapezoid(jnp.trapezoid(solver.f_eq, solver.mesh.vs, axis=1),
                            solver.mesh.xs) + eps

    f_final = f_T / norm_final
    f_eq = solver.f_eq / norm_eq

    kl_div = jnp.trapezoid(jnp.trapezoid(
        jax.scipy.special.rel_entr(f_final, f_eq + eps),
        solver.mesh.vs,
        axis=1
    ), solver.mesh.xs)
    return kl_div


def make_cost_function_kl(
    solver: VlasovPoissonSolver,
    solver_jit: Callable[
        [Array, Array, float], tuple[Array, Array, Array]
    ],
    f_iv: Array,
    k_0: float,
    t_final: float
) -> Callable[[Array], Array]:
    """Generate KL cost function."""
    @jax.jit
    def cost_function_kl(a_k: Array) -> Array:
        H = external_electric_field(a_k, solver.mesh, k_0)
        f_array, _, _, _ = solver_jit(f_iv, H, t_final)
        return kl_divergence(f_array, solver)

    return cost_function_kl


# --- Final Electric Energy ---

def make_cost_function_ee(
    solver: VlasovPoissonSolver,
    solver_jit: Callable[
        [Array, Array, float], tuple[Array, Array, Array]
    ],
    f_iv: Array,
    k_0: float,
    t_final: float
) -> Callable[[Array], Array]:
    """
    Generate electric energy cost function.

        EE = ∫ [E(T, x)]² dx
    """
    @jax.jit
    def cost_function_ee(a_k: Array) -> Array:
        H = external_electric_field(a_k, solver.mesh, k_0)
        _, _, _, ee_array = solver_jit(f_iv, H, t_final)
        return ee_array[-1]

    return cost_function_ee


# --- Total Electric Energy Over Time ---

def electric_energy_in_time(
    ee_array: Array,
    solver: VlasovPoissonSolver
) -> Array:
    """
    Compute the total electric energy over time:

        EE = ∫∫ [E(t, x)]² dx dt
    """
    return jnp.trapezoid(ee_array, dx=solver.dt)


def make_cost_function_eet(
    solver: VlasovPoissonSolver,
    solver_jit: Callable[
        [Array, Array, float], tuple[Array, Array, Array]
    ],
    f_iv: Array,
    k_0: float,
    t_final: float
) -> Callable[[Array], Array]:
    """Generate total-time electric energy cost function."""
    @jax.jit
    def cost_function_eet(a_k: Array) -> Array:
        H = external_electric_field(a_k, solver.mesh, k_0)
        _, _, _, ee_array = solver_jit(f_iv, H, t_final)
        return electric_energy_in_time(ee_array, solver)

    return cost_function_eet



# ===== Plotting Functions =====

def plot_feq_distribution(
    fig: figure.Figure,
    ax: axes.Axes,
    f_eq: Array,
    title: str,
    mesh: Mesh,
    sci: bool = False
) -> None:
    """
    Plot the equilibrium distribution f_eq(x, v).
    """
    im = ax.imshow(
        f_eq.T,
        extent=(float(mesh.xs[0]), float(mesh.xs[-1]),
                float(mesh.vs[0]), float(mesh.vs[-1])),
        aspect="auto",
        cmap="plasma",
    )
    ax.set_title(title)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$v$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if sci:
        cbar.ax.set_yscale("log")


def plot_distribution(
    fig: figure.Figure,
    ax: axes.Axes,
    data: Array,
    title: str,
    time: float,
    mesh: Mesh,
    sci: bool = False
) -> None:
    """
    Plot the plasma distribution f(x, v) at a given time.
    """
    im = ax.imshow(
        data.T,
        extent=(float(mesh.xs[0]),float(mesh.xs[-1]),
                float(mesh.vs[0]), float(mesh.vs[-1])),
        aspect="auto",
        cmap="plasma",
    )
    ax.set_title(f"{title} (T={time:.0f})")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$v$")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if sci:
        cbar.ax.set_yscale("log")


def plot_inital_solve(
    fig: figure.Figure,
    axs: list[axes.Axes],
    f_eq: Array,
    f_array_1: Array,
    ee_array_1: Array,
    f_array_2: Array,
    ee_array_2: Array,
    mesh: Mesh,
    t_values: Array,
    sci: bool = False,
) -> None:
    """
    Plot solution comparison between H ≡ 0 and a good initial guess.
    """
    plot_feq_distribution(
        fig, axs[0], f_eq, "Distribution of $f_{eq}$", mesh, sci
    )
    plot_distribution(
        fig, axs[1], f_array_1,
        "Distribution of $f[H\\equiv 0]$",
        float(t_values[-1]), mesh, sci
    )
    plot_distribution(
        fig, axs[2], f_array_2,
        "Distribution of $f[H]$",
        float(t_values[-1]), mesh, sci
    )

    axs[3].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[3].plot(t_values, ee_array_1, label="No $H$")
    axs[3].plot(t_values, ee_array_2, label="Good initial $H$")
    axs[3].set_xlabel("$t$")
    axs[3].set_title("$\\mathcal{E}_{f}(t)$")
    axs[3].legend()


def plot_results_TS(
    fig: figure.Figure,
    axs: list[axes.Axes],
    f_final: Array,
    E_array: Array,
    H: Array,
    ee_array: Array,
    objective_values: Array,
    t_values: Array,
    mesh: Mesh,
) -> None:
    """
    Plot results of optimization for Two-Stream equilibrium.
    """
    dt = t_values[1] - t_values[0]

    plot_distribution(
        fig, axs[0], f_final, "Distribution of $f[H]$",
        float(t_values[-1]), mesh
    )

    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[1].plot(mesh.xs, H, label="$H(x)$")
    axs[1].plot(mesh.xs, E_array[0] - H, label=f"$E(t={0*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[99] - H, label=f"$E(t={100*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[199] - H, label=f"$E(t={200*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[299] - H, label=f"$E(t={300*dt:.0f},x)$")
    axs[1].set_xlabel("$x$")
    axs[1].set_title("Electric fields")
    axs[1].legend(loc="upper right")

    axs[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[2].plot(t_values, ee_array)
    axs[2].set_xlabel("$t$")
    axs[2].set_title("$\\mathcal{E}_{f}(t)$")

    axs[3].plot(objective_values)
    axs[3].set_yscale("log")
    axs[3].set_xlabel("Iteration")
    axs[3].set_title("Convergence of Objective")


def plot_results_BoT(
    fig: figure.Figure,
    axs: list[axes.Axes],
    f_final: Array,
    E_array: Array,
    H: Array,
    ee_array: Array,
    objective_values: Array,
    t_values: Array,
    mesh: Mesh,
) -> None:
    """
    Plot results of optimization for Bump-on-Tail equilibrium.
    """
    dt = t_values[1] - t_values[0]

    plot_distribution(
        fig, axs[0], f_final, "Distribution of $f[H]$",
        float(t_values[-1]), mesh
    )

    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[1].plot(mesh.xs, H, label="$H(x)$")
    axs[1].plot(mesh.xs, E_array[0] - H, label=f"$E(t={0*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[199] - H, label=f"$E(t={200*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[299] - H, label=f"$E(t={300*dt:.0f},x)$")
    axs[1].plot(mesh.xs, E_array[399] - H, label=f"$E(t={400*dt:.0f},x)$")
    axs[1].set_xlabel("$x$")
    axs[1].set_title("Electric fields")
    axs[1].legend(loc="upper right")

    axs[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[2].plot(t_values, ee_array)
    axs[2].set_xlabel("$t$")
    axs[2].set_title("$\\mathcal{E}_{f}(t)$")

    axs[3].plot(objective_values)
    axs[3].set_yscale("log")
    axs[3].set_xlabel("Iteration")
    axs[3].set_title("Convergence of Objective")