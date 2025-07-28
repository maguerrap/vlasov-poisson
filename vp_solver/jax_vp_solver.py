import dataclasses
from typing import Callable

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclasses.dataclass
class Mesh:
    """Mesh object."""
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
    """Generates mesh object."""
    xs = jnp.linspace(0.0, length_x, nx, endpoint=False)
    dx = float(xs[1] - xs[0])
    vs = jnp.linspace(-length_v, length_v, nv, endpoint=False)
    dv = float(vs[1] - vs[0])
    V, X = jnp.meshgrid(vs, xs)
    return Mesh(
        xs=xs,
        dx=dx,
        vs=vs,
        dv=dv,
        X=X,
        V=V,
        period_x=length_x,
        period_v=length_v,
        nx=nx,
        nv=nv,
    )


@dataclasses.dataclass(frozen=True)
class VlasovPoissonSolver:
    """
    Vlasov–Poisson semi-Lagrangian solver with operator splitting.

    The Vlasov–Poisson system is given by

        ∂_t f(t, x, v)
        + v ∂_x f(t, x, v)
        - (E(t, x) + H(x)) ∂_v f(t, x, v) = 0,

    where f(t, x, v) is the particle distribution function.

    The electric field E(t, x) is determined by Poisson's equation:

        E(t, x) = ∂_x V(t, x)

        ∂_xx V(t, x) = 1 - ρ(t, x),

    with charge density

        ρ(t, x) = ∫ f(t, x, v) dv.

    We split our operator into
        ∂_t f(t, x, v) + v ∂_x f(t, x, v) = 0  (1),
    and
        ∂_t f(t, x, v) - (E(t, x) + H(x)) ∂_v f(t, x, v) = 0    (2).
    """

    mesh: Mesh
    dt: float
    f_eq: Array

    def build_semilag_x(self) -> Callable[[Array, Array], Array]:
        """
        Builds semi-Lagrangian to solve (1) via linear interpolation.
        """
        def interp_jax_x(f: Array, v: Array) -> Array:
            return jnp.interp(
                self.mesh.xs - 0.5 * v * self.dt,
                self.mesh.xs,
                f,
                period=self.mesh.period_x,
            )

        return jax.vmap(interp_jax_x, in_axes=(1, 0), out_axes=1)

    def build_semilag_v(self) -> Callable[[Array, Array], Array]:
        """
        Builds semi-Lagrangian to solve (2) via linear interpolation.
        """
        def interp_jax_v(f: Array, E: Array) -> Array:
            return jnp.interp(
                self.mesh.vs - E * self.dt,
                self.mesh.vs,
                f,
                period=2 * self.mesh.period_v,
            )

        return jax.vmap(interp_jax_v, in_axes=(0, 0), out_axes=0)

    def compute_rho(self, f: Array) -> Array:
        """Compute value of ρ(t, x)."""
        return self.mesh.dv * jnp.sum(self.f_eq - f, axis=1)

    def compute_E_from_rho(self, rho: Array) -> Array:
        """Compute value of E(t, x) from ρ(t, x)."""

        rho_hat = jnp.fft.fft(rho)
        E_hat = jnp.zeros_like(rho_hat)

        inv_multiplier = -1.0 / (
            1j
            * 2
            * jnp.pi
            * jnp.fft.fftfreq(
                self.mesh.nx,
                d=self.mesh.period_x / self.mesh.nx,
            )[1:]
        )

        E_hat = E_hat.at[1:].set(inv_multiplier * rho_hat[1:])
        return jnp.real(jnp.fft.ifft(E_hat))

    def compute_E(self, f: Array) -> Array:
        """Compute E(t, x)."""
        return self.compute_E_from_rho(self.compute_rho(f))

    def compute_electric_energy(self, E: Array) -> Array:
        """Compute electric energy from E(t, x)."""
        return 0.5 * jnp.sum(jnp.square(E)) * self.mesh.dx


    def run_forward_jax_scan(
        self, f_iv: Array, H: Array, t_final: float
    ) -> tuple:
        
        """
        Compute time integration for the time derivative.
        """

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

        f_array, (E_array, ee_array) = jax.lax.scan(
            time_step_jax, f, tspan
        )

        return f_array, E_array, ee_array
