import jax
import jax.numpy as jnp

# Function to compute the external field H
def external_electric_field(ak, mesh, k_0):
    N = ak.shape[1]
    k = jnp.arange(1, N+1)
    H = ak[0,:] @ jnp.cos(k_0 * k[:,None] * mesh.xs) + ak[1,:] @ jnp.sin(k_0 * k[:,None] * mesh.xs)
    return H




#### Cost functions #####

#### KL ####

def kl_divergence(f_final, solver, eps=1e-12):
    # Normalize the distributions so they sum to 1
    f_final = f_final / (jnp.sum(f_final) * solver.mesh.dx * solver.mesh.dv + eps)
    f_eq = solver.f_eq / (jnp.sum(solver.f_eq) * solver.mesh.dx * solver.mesh.dv + eps)
    # Compute KL divergence, adding eps to avoid log(0)
    kl_div = jnp.sum(jax.scipy.special.rel_entr(f_final, f_eq + eps) * solver.mesh.dx * solver.mesh.dv)
    return kl_div


def make_cost_function_kl(solver, solver_jit, f_iv, k_0, t_final):
    @jax.jit
    def cost_function_kl(a_k):
        H = external_electric_field(a_k, solver.mesh, k_0)
        f_array, E_array, ee_array = solver_jit(f_iv, H, t_final)
        cost = kl_divergence(f_array, solver)
        return cost

    return cost_function_kl


### EE ###

def make_cost_function_ee(solver, solver_jit, f_iv, k_0, t_final):
    @jax.jit
    def cost_function_ee(a_k):
        H = external_electric_field(a_k, solver.mesh, k_0)
        f_array, E_array, ee_array = solver_jit(f_iv, H, t_final)
        cost = ee_array[-1]
        return cost

    return cost_function_ee


### EET ###

def electric_energy_in_time(ee_array, solver):
    return jnp.sum(ee_array)*solver.dt


def make_cost_function_eet(solver, solver_jit, f_iv, k_0, t_final):
    @jax.jit
    def cost_function_eet(a_k):
        H = external_electric_field(a_k, solver.mesh, k_0)
        f_array, E_array, ee_array = solver_jit(f_iv, H, t_final)
        cost = electric_energy_in_time(ee_array, solver)
        return cost

    return cost_function_eet
