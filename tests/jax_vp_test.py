import jax
import jax.numpy as jnp
import pytest

from vp_solver.jax_vp_solver import (
    Mesh,
    make_mesh,
    VlasovPoissonSolver,
)


# ----------------------------
# Fixtures
# ----------------------------

@pytest.fixture
def mesh():
    return make_mesh(
        length_x=2.0 * jnp.pi,
        length_v=6.0,
        nx=16,
        nv=32,
    )


@pytest.fixture
def f_eq(mesh):
    """Simple Maxwellian equilibrium."""
    return jnp.exp(-0.5 * mesh.V**2)


@pytest.fixture
def solver(mesh, f_eq):
    return VlasovPoissonSolver(
        mesh=mesh,
        dt=0.01,
        f_eq=f_eq,
    )


@pytest.fixture
def f0(mesh, f_eq):
    """Small perturbation around equilibrium."""
    perturb = 0.01 * jnp.cos(mesh.X)
    return f_eq * (1.0 + perturb)


# ----------------------------
# Mesh tests
# ----------------------------

def test_make_mesh_shapes(mesh):
    assert mesh.X.shape == (mesh.nx, mesh.nv)
    assert mesh.V.shape == (mesh.nx, mesh.nv)
    assert mesh.xs.shape == (mesh.nx,)
    assert mesh.vs.shape == (mesh.nv,)


# ----------------------------
# Semi-Lagrangian operators
# ----------------------------

def test_semilag_x_shape(solver, f0):
    semilag_x = solver.build_semilag_x()
    f1 = semilag_x(f0, solver.mesh.vs)
    assert f1.shape == f0.shape


def test_semilag_v_shape(solver, f0):
    semilag_v = solver.build_semilag_v()
    E = jnp.zeros(solver.mesh.nx)
    f1 = semilag_v(f0, E)
    assert f1.shape == f0.shape


# ----------------------------
# Field computations
# ----------------------------

def test_compute_rho_shape(solver, f0):
    rho = solver.compute_rho(f0)
    assert rho.shape == (solver.mesh.nx,)


def test_compute_E_shape(solver, f0):
    E = solver.compute_E(f0)
    assert E.shape == (solver.mesh.nx,)


def test_electric_energy_positive(solver, f0):
    E = solver.compute_E(f0)
    ee = solver.compute_electric_energy(E)
    assert ee >= 0.0


# ----------------------------
# Fourier Poisson solver sanity
# ----------------------------

def test_zero_density_gives_zero_field(solver, f_eq):
    """If f == f_eq, rho = 0 => E = 0."""
    rho = solver.compute_rho(f_eq)
    E = solver.compute_E_from_rho(rho)
    assert jnp.allclose(E, 0.0, atol=1e-12)


# ----------------------------
# Time integration
# ----------------------------

def test_one_time_step_forward(solver, f0):
    H = jnp.zeros(solver.mesh.nx)
    t_final = solver.dt  # one step
    num_steps = int(t_final/solver.dt)

    f_array, f_total, E_array, ee_array = solver.run_forward_jax_scan(
        f0, H, t_final
    )

    assert f_array.ndim == 2
    assert f_array.shape == (solver.mesh.nx, solver.mesh.nv)
    assert f_total.shape == (num_steps, solver.mesh.nx, solver.mesh.nv)
    assert E_array.shape == (num_steps, solver.mesh.nx)
    assert ee_array.shape == (num_steps,)


def test_scan_is_jittable(solver, f0):
    """Ensure no tracer errors under jit."""
    H = jnp.zeros(solver.mesh.nx)

    @jax.jit
    def run():
        return solver.run_forward_jax_scan(f0, H, solver.dt)

    run()  # should not raise

# ----------------------------
# Physics aware tests
# ----------------------------

def test_mass_conservation(solver, f0):
    H = jnp.zeros(solver.mesh.nx)
    t_final = 10 * solver.dt

    f_array, _, _, _ = solver.run_forward_jax_scan(
        f0, H, t_final
    )

    def mass(f):
        return jnp.trapezoid(jnp.trapezoid(f, solver.mesh.xs, axis=0),
                             solver.mesh.vs, axis=0)

    mass0 = mass(f0)
    massT = mass(f_array)

    assert jnp.isclose(
        mass0, massT, rtol=1e-6, atol=1e-8
    )


def test_zero_mode_filtered(solver, f0):
    rho = solver.compute_rho(f0)
    E = solver.compute_E_from_rho(rho)

    E_hat = jnp.fft.fft(E)

    assert jnp.isclose(E_hat[0], 0.0, atol=1e-12)


def test_linear_landau_damping(solver, f0):
    """
    Qualitative Landau damping test:
    electric energy should decay at early times.
    """
    H = jnp.zeros(solver.mesh.nx)
    t_final = 20 * solver.dt

    _, _, _, ee_array = solver.run_forward_jax_scan(
        f0, H, t_final
    )

    # ignore first step (transient)
    ee = ee_array[1:]

    # energy should decrease overall
    assert ee[-1] < ee[0]

    # no explosive growth
    assert jnp.max(ee) < 2.0 * ee[0]
