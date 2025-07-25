import jax
import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})


# ===== Auxiliary Functions =====

def external_electric_field(ak, mesh, k_0):
    N = ak.shape[1]
    k = jnp.arange(1, N + 1)
    cos_term = ak[0, :] @ jnp.cos(k_0 * k[:, None] * mesh.xs)
    sin_term = ak[1, :] @ jnp.sin(k_0 * k[:, None] * mesh.xs)
    return cos_term + sin_term


# ===== Cost Functions =====

# --- KL Divergence ---

def kl_divergence(f_final, solver, eps=1e-12):
    dx_dv = solver.mesh.dx * solver.mesh.dv
    norm_final = jnp.sum(f_final) * dx_dv + eps
    norm_eq = jnp.sum(solver.f_eq) * dx_dv + eps

    f_final = f_final / norm_final
    f_eq = solver.f_eq / norm_eq

    kl_div = jnp.sum(
        jax.scipy.special.rel_entr(f_final, f_eq + eps) * dx_dv
    )
    return kl_div


def make_cost_function_kl(solver, solver_jit, f_iv, k_0, t_final):
    @jax.jit
    def cost_function_kl(a_k):
        H = external_electric_field(a_k, solver.mesh, k_0)
        f_array, E_array, ee_array = solver_jit(f_iv, H, t_final)
        cost = kl_divergence(f_array, solver)
        return cost

    return cost_function_kl


# --- Final Electric Energy ---

def make_cost_function_ee(solver, solver_jit, f_iv, k_0, t_final):
    @jax.jit
    def cost_function_ee(a_k):
        H = external_electric_field(a_k, solver.mesh, k_0)
        f_array, E_array, ee_array = solver_jit(f_iv, H, t_final)
        return ee_array[-1]

    return cost_function_ee


# --- Total Electric Energy Over Time ---

def electric_energy_in_time(ee_array, solver):
    return jnp.sum(ee_array) * solver.dt


def make_cost_function_eet(solver, solver_jit, f_iv, k_0, t_final):
    @jax.jit
    def cost_function_eet(a_k):
        H = external_electric_field(a_k, solver.mesh, k_0)
        f_array, E_array, ee_array = solver_jit(f_iv, H, t_final)
        return electric_energy_in_time(ee_array, solver)

    return cost_function_eet


# ===== Plotting Functions =====

def plot_feq_distribution(fig, ax, f_eq, title, mesh, sci=False):
    im = ax.imshow(
        f_eq.T,
        extent=[mesh.xs[0], mesh.xs[-1], mesh.vs[0], mesh.vs[-1]],
        aspect='auto',
        cmap='plasma'
    )
    ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v$')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if sci:
        cbar.ax.set_yscale('log')


def plot_distribution(fig, ax, data, title, time, mesh, sci=False):
    im = ax.imshow(
        data.T,
        extent=[mesh.xs[0], mesh.xs[-1], mesh.vs[0], mesh.vs[-1]],
        aspect='auto',
        cmap='plasma'
    )
    ax.set_title(f'{title} (T={time:.0f})')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$v$')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if sci:
        cbar.ax.set_yscale('log')


def plot_inital_solve(
    fig, axs, f_eq, f_array_1, ee_array_1, f_array_2, ee_array_2,
    mesh, t_values, sci=False
):
    plot_feq_distribution(
        fig, axs[0], f_eq, 'Distribution of $f_{eq}$', mesh, sci
    )
    plot_distribution(
        fig, axs[1], f_array_1, 'Distribution of $f[H\\equiv 0]$',
        t_values[-1], mesh, sci
    )
    plot_distribution(
        fig, axs[2], f_array_2, 'Distribution of $f[H]$',
        t_values[-1], mesh, sci
    )

    axs[3].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    axs[3].plot(t_values, ee_array_1, label="No $H$")
    axs[3].plot(t_values, ee_array_2, label="Good initial $H$")
    axs[3].set_xlabel('$t$')
    axs[3].set_title('$\\mathcal{E}_{f}(t)$')
    axs[3].legend()


def plot_results_TS(
    fig, axs, f_final, E_array, H, ee_array, objective_values,
    t_values, mesh
):
    dt = t_values[1] - t_values[0]

    plot_distribution(
        fig, axs[0], f_final, 'Distribution of $f[H]$',
        t_values[-1], mesh
    )

    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    axs[1].plot(mesh.xs, H, label='$H(x)$')
    axs[1].plot(mesh.xs, E_array[0] - H, label='$E(t=0,x)$')
    axs[1].plot(mesh.xs, E_array[99] - H, label='$E(t=100,x)$')
    axs[1].plot(mesh.xs, E_array[199] - H, label='$E(t=200,x)$')
    axs[1].plot(mesh.xs, E_array[299] - H, label='$E(t=300,x)$')
    axs[1].set_xlabel('$x$')
    axs[1].set_title('Electric fields')
    axs[1].legend(loc='upper right')

    axs[2].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    axs[2].plot(t_values, ee_array)
    axs[2].set_xlabel('$t$')
    axs[2].set_title('$\\mathcal{E}_{f}(t)$')

    axs[3].plot(objective_values)
    axs[3].set_yscale('log')
    axs[3].set_xlabel("Iteration")
    axs[3].set_title("Convergence of Objective")


def plot_results_BoT(
    fig, axs, f_final, E_array, H, ee_array, objective_values,
    t_values, mesh
):
    dt = t_values[1] - t_values[0]

    plot_distribution(
        fig, axs[0], f_final, 'Distribution of $f[H]$',
        t_values[-1], mesh
    )

    axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    axs[1].plot(mesh.xs, H, label='$H(x)$')
    axs[1].plot(mesh.xs, E_array[0] - H, label='$E(t=0,x)$')
    axs[1].plot(mesh.xs, E_array[199] - H, label='$E(t=200,x)$')
    axs[1].plot(mesh.xs, E_array[299] - H, label='$E(t=300,x)$')
    axs[1].plot(mesh.xs, E_array[399] - H, label='$E(t=400,x)$')
    axs[1].set_xlabel('$x$')
    axs[1].set_title('Electric fields')
    axs[1].legend(loc='upper right')

    axs[2].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    axs[2].plot(t_values, ee_array)
    axs[2].set_xlabel('$t$')
    axs[2].set_title('$\\mathcal{E}_{f}(t)$')

    axs[3].plot(objective_values)
    axs[3].set_yscale('log')
    axs[3].set_xlabel("Iteration")
    axs[3].set_title("Convergence of Objective")
