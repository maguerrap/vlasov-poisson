import jax
Array = jax.Array
import jax.numpy as jnp
import dataclasses
from functools import partial
from typing import Callable, Union



@dataclasses.dataclass
class Mesh:
    xs: Array
    dx: float
    vs: Array
    dv: float
    V: Array
    X: Array
    period_x: float
    period_v: float
    nx: int
    nv: int
  
def make_mesh(length_x: float, length_v: float, nx: int, nv: int) -> Mesh:
    xs = jnp.linspace(0., length_x, nx, endpoint=False)
    dx = xs[1] - xs[0]
    vs = jnp.linspace(-length_v, length_v, nv, endpoint=False)
    dv = vs[1] - vs[0]
    V, X = jnp.meshgrid(vs, xs)
    return Mesh(xs=xs, dx=dx, vs=vs, dv=dv, X=X, V=V, period_x=length_x, period_v=length_v, nx=nx, nv=nv)

@dataclasses.dataclass(frozen=True)
class VlasovPoissonSolver:
    mesh: Mesh
    dt: float
    f_eq: Array


    def build_semilag_x(self) -> Callable[[Array], Array]:
        def interp_jax_x(f, v):
            return jnp.interp(self.mesh.xs - 0.5 * v * self.dt, self.mesh.xs, f, period=self.mesh.period_x)
        return jax.vmap(interp_jax_x, in_axes=(1,0), out_axes=1)

    def build_semilag_v(self) -> Callable[[Array, Array], Array]:
        def interp_jax_v(f: Array, E: Array) -> Array:
            return jnp.interp(self.mesh.vs - E * self.dt, self.mesh.vs, f, period=2 * self.mesh.period_v)
        return jax.vmap(interp_jax_v, in_axes=(0, 0), out_axes=0)

    def compute_rho(self, f: Array) -> Array:
        return self.mesh.dv * jnp.sum(self.f_eq - f, axis=1)

    def compute_E_from_rho(self, rho: Array) -> Array:
        rho_hat = jnp.fft.fft(rho)
        E_hat = jnp.zeros_like(rho_hat)
        inv_multiplier = -1.0 / (1j * 2 * jnp.pi * jnp.fft.fftfreq(self.mesh.nx, d=self.mesh.period_x / self.mesh.nx)[1:])
        E_hat = E_hat.at[1:].set(inv_multiplier * rho_hat[1:])
        return jnp.real(jnp.fft.ifft(E_hat))

    def compute_E(self, f: Array) -> Array:
        return self.compute_E_from_rho(self.compute_rho(f))

    def compute_electric_energy(self, E: Array) -> Array:
        return 0.5 * jnp.sum(jnp.square(E)) * self.mesh.dx

    @jax.jit
    def time_step(self, f: Array, H: Array) -> tuple[Array, Array, Array]:
        f_half = self.semilag_x(f)
        E = self.compute_E(f_half)
        E_total = E + H
        f = self.semilag_v(f_half, E_total)
        f = self.semilag_x(f)
        return f, f_half, E_total

    def run_forward_jax_scan(self, f_iv: Array, H: Array, t_final: float) -> tuple:
        num_steps = int(t_final / self.dt)
        f = f_iv.copy()
        tspan = self.dt * jnp.linspace(0, t_final, num_steps)

        semilag_x = self.build_semilag_x()
        semilag_v = self.build_semilag_v()
        compute_E_jax = jax.jit(self.compute_E)
        compute_energy_jax = jax.jit(self.compute_electric_energy)

        @jax.jit
        def time_step_jax(f, t):
            f_half = semilag_x(f, self.mesh.vs)
            E = compute_E_jax(f_half)
            E_total = E + H
            ee = compute_energy_jax(E)
            f = semilag_v(f_half, E_total)
            f = semilag_x(f, self.mesh.vs)
            return f, (E_total, ee)

        # Accumulate the final f, E, and electric energy ee
        f_array, (E_array, ee_array) = jax.lax.scan(time_step_jax, f, tspan)

        return f_array, E_array, ee_array