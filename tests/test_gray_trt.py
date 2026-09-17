import jax
import jax.numpy as jnp
import numpy as np

from trt_jfnk.discretization.boundary import BoundaryCondition2D, PERIODIC
from trt_jfnk.discretization.grid import Grid2D
from trt_jfnk.models.gray_trt import GrayTRTModel
from trt_jfnk.models.material import PolynomialMaterial
from trt_jfnk.models.opacity import PowerLawOpacity
from trt_jfnk.solvers.jvp import JVPOptions, build_jacobian_action


def problem():
    grid = Grid2D(7, 5, periodic_x=True, periodic_y=True)
    bc = BoundaryCondition2D(PERIODIC, PERIODIC)
    model = GrayTRTModel(
        opacity=PowerLawOpacity(sigma_a0=0.7, exponent=2.5, temperature_floor=0.1),
        material=PolynomialMaterial(cv0=0.3, cv1=0.8, exponent=3),
    )
    x, y = grid.mesh
    temperature = 0.5 + 0.03 * jnp.cos(x) * jnp.cos(2.0 * y)
    radiation = temperature**4 + 0.02 * jnp.exp(-x**2)
    state = model.pack(radiation, temperature)
    old_state = model.pack(0.98 * radiation, 0.99 * temperature)
    source = jnp.zeros(grid.shape)
    residual = lambda value: model.residual(value, old_state, 0.02, source, source, grid, bc)
    return grid, bc, model, state, residual


def test_ad_jvp_matches_centered_fd():
    _grid, _bc, _model, state, residual = problem()
    direction = jnp.linspace(-1.0, 1.0, state.size)
    ad = build_jacobian_action(residual, state, JVPOptions("ad", jit=False))(direction)
    fd = build_jacobian_action(
        residual,
        state,
        JVPOptions("fd", fd_scheme="central", fd_relative_step=2.0e-6, jit=False),
    )(direction)
    np.testing.assert_allclose(ad, fd, rtol=2e-6, atol=2e-8)


def test_residual_is_genuinely_nonlinear():
    _grid, _bc, _model, state, residual = problem()
    perturbation = 0.02 * jnp.sin(jnp.arange(state.size))
    midpoint_defect = residual(state + perturbation) + residual(state - perturbation) - 2.0 * residual(state)
    assert float(jnp.linalg.norm(midpoint_defect)) > 1.0e-6


def test_exchange_conserves_radiation_plus_material_energy_rhs():
    grid, bc, model, state, _residual = problem()
    radiation, temperature = model.unpack(state, grid)
    zero = jnp.zeros(grid.shape)
    radiation_rhs, material_rhs = model.rhs(radiation, temperature, zero, grid, bc)
    # Periodic diffusion sums to zero, and exchange cancels cell by cell.
    assert abs(float(jnp.sum(radiation_rhs + material_rhs))) < 1.0e-10
