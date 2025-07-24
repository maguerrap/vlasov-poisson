import jax
Array = jax.Array
import jax.numpy as jnp
import interpax
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

    def semilag_x(self, f: Array) -> Array:
        out = jnp.zeros_like(f)
        for i, v in enumerate(self.mesh.vs):
            #out = out.at[:, i].set(jnp.interp(self.mesh.xs - 0.5 * v * self.dt,
            #                                  self.mesh.xs, f[:, i], period=self.mesh.period_x))
            out = out.at[:,i].set(interpax.interp1d(self.mesh.xs - 0.5 * v * self.dt,
                                              self.mesh.xs, f[:, i], method='cubic2', period=self.mesh.period_x))
        return out

    def semilag_v(self, f: Array, E: Array, s: Union[Array, None] = None):
        out = jnp.zeros_like(f)
        if jnp.sum((jnp.real(E) * self.dt % self.mesh.dv) == 0) != 0.0:
            raise ValueError("E*dt shouldn't be a multiple of dv.")
        for i in range(self.mesh.nx):
            #out = out.at[i, :].set(jnp.interp(self.mesh.vs - E[i] * self.dt,
            #                                  self.mesh.vs, f[i, :], period=2.0 * self.mesh.period_v))
            out = out.at[:,i].set(interpax.interp1d(self.mesh.vs - E[i] * self.dt,
                                              self.mesh.vs, f[i, :], method='cubic2', period=2 * self.mesh.period_v))
            if s is not None:
                out = out.at[i, :].add(self.dt * s[i])
        return out

    def build_semilag_x(self) -> Callable[[Array], Array]:
        def interp_jax_x(f, v):
            #return jnp.interp(self.mesh.xs - 0.5 * v * self.dt, self.mesh.xs, f, period=self.mesh.period_x)
            return interpax.interp1d(self.mesh.xs - 0.5 * v * self.dt, self.mesh.xs, f, method='cubic2',
                                     period=self.mesh.period_x)
        interp_vmap = jax.vmap(interp_jax_x, in_axes=(1, 0), out_axes=1)
        def semilag_x(f):
            return interp_vmap(f, self.mesh.vs)
        return semilag_x

    def build_semilag_v(self) -> Callable[[Array, Array], Array]:
        def interp_jax_v(f: Array, E: Array) -> Array:
            #return jnp.interp(self.mesh.vs - E * self.dt, self.mesh.vs, f, period=2 * self.mesh.period_v)
            return interpax.interp1d(self.mesh.vs - E * self.dt, self.mesh.vs, f, method='cubic2',
                                     period=2 * self.mesh.period_v)
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
            f_half = semilag_x(f)
            E = compute_E_jax(f_half)
            E_total = E + H
            ee = compute_energy_jax(E)
            f = semilag_v(f_half, E_total)
            f = semilag_x(f)
            return f, (E_total, ee)

        # Accumulate the final f, E, and electric energy ee
        f_array, (E_array, ee_array) = jax.lax.scan(time_step_jax, f, tspan)

        return f_array, E_array, ee_array
    
    @jax.jit
    def shift_by_n(self, g, EH):
        hv = self.mesh.dv
        dt = self.dt
        n = jnp.floor(-dt * EH / hv).astype(int)  # Shape (nx,)

        def shift_row(row, shift):
            return jnp.roll(row, -shift)

        return jax.vmap(shift_row)(g, n)

    @partial(jax.jit, static_argnums=0)
    def time_step_backward(self, f, f_star, EH_star):
        dt = self.dt
        hv = self.mesh.dv

        f = self.semilag_x(f)
        g_starstar = f
        g_ss_s = self.shift_by_n(f, EH_star)
        g_ss_d = (g_ss_s - jnp.roll(g_ss_s, 1, axis=1)) / hv
        rho = jnp.sum(g_ss_d * f_star, axis=1) * hv
        s = self.compute_E_from_rho(rho)
        f = self.semilag_v(f, EH_star, s)
        f = self.semilag_x(f)

        return g_starstar, f

    # Modified run_backward function
    @partial(jax.jit, static_argnums=0)
    def run_backward(self, gT, f_stars, E_stars):
        dt = self.dt
        hv = self.mesh.dv

        f_stars_rev = f_stars[::-1]
        E_stars_rev = E_stars[::-1]
        f_init = gT

        def backward_step(f_prev, inputs):
            f_star, EH_star = inputs
            g_starstar, f_next = self.time_step_backward(f_prev, f_star, EH_star)
            return f_next, (g_starstar, f_next)

        inputs = (f_stars_rev, E_stars_rev)
        f_final, outputs = jax.lax.scan(backward_step, f_init, inputs)
        g_starstars, fs = outputs

        # Instead of using concatenate, include gT as part of the iteration
        fs = jax.vmap(lambda f, g: g, in_axes=(0, None))(fs, gT)
        
        return fs[::-1], g_starstars[::-1]
    
    @partial(jax.jit, static_argnums=0)
    def calc_cost_gradient(self, H: Array, f_eq: Array):
        f_array, E_array, ee_array = self.run_forward_jax_scan(H, f_eq, t_final=self.dt * len(H))
        cost = jnp.sum((f_array[-1] - f_eq) ** 2) * self.mesh.dx * self.mesh.dv

        gT = 2 * (f_array[-1] - f_eq) / self.dt
        f_back_array, g_starstars = self.run_backward(gT, f_array, E_array)

        # Vectorize gradient calculations
        f_ss_s = jax.vmap(self.shift_by_n, in_axes=(0, 0))(f_array, E_array)
        f_ss_d = jax.vmap(lambda f: (f - jnp.roll(f, -1, axis=1)) / self.mesh.dv)(f_ss_s)
        gradient = jax.vmap(lambda fd, g: jnp.sum(fd * g, axis=1) * self.mesh.dv)(f_ss_d, g_starstars).sum(axis=0) * self.dt ** 2 * self.mesh.dx

        return cost, gradient, f_array, ee_array

    @partial(jax.jit, static_argnums=0)
    def calc_cost_only(self, H: Array, f_eq: Array):
        f_array, _, _ = self.run_forward_jax_scan(H, f_eq, t_final=self.dt * len(H))
        return jnp.sum((f_array[-1] - f_eq) ** 2) * self.mesh.dx * self.mesh.dv
