import jax.numpy as jnp
import numpy as np

from trt_jfnk.discretization.boundary import BoundaryCondition2D, PERIODIC
from trt_jfnk.discretization.grid import Grid2D
from trt_jfnk.models.gray_trt import GrayTRTModel
from trt_jfnk.models.material import PolynomialMaterial
from trt_jfnk.models.opacity import PowerLawOpacity
from trt_jfnk.solvers.block_preconditioners import gray_local_block_factory
from trt_jfnk.solvers.jvp import JVPOptions
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy


def test_scaled_and_unscaled_converge_to_same_nonlinear_root():
    grid = Grid2D(7, 3, periodic_x=True, periodic_y=True)
    bc = BoundaryCondition2D(PERIODIC, PERIODIC)
    model = GrayTRTModel(
        opacity=PowerLawOpacity(sigma_a0=0.5, exponent=2.0, temperature_floor=0.1),
        material=PolynomialMaterial(cv0=0.5, cv1=0.2, exponent=3),
    )
    temperature = jnp.full(grid.shape, 0.5)
    radiation = temperature**4
    old_state = model.pack(radiation, temperature)
    x, _y = grid.mesh
    source = 0.2 * jnp.exp(-x**2)
    residual = lambda candidate: model.residual(
        candidate, old_state, 0.01, source, source, grid, bc
    )
    options = NewtonOptions(
        atol=1e-11,
        rtol=1e-9,
        max_iterations=12,
        jvp=JVPOptions("ad"),
        krylov=KrylovOptions(backend="scipy", rtol=1e-10, max_iterations=200),
    )
    preconditioner = gray_local_block_factory(model, 0.01, grid, bc)
    admissible = lambda state: model.is_admissible(state, grid, radiation_floor=-1e-12)
    unscaled = newton_krylov(
        residual, old_state, grid.size, ScalePolicy("none"), options, admissible, preconditioner
    )
    scaled = newton_krylov(
        residual,
        old_state,
        grid.size,
        ScalePolicy("fixed", (0.05, 0.5)),
        options,
        admissible,
        preconditioner,
    )
    assert unscaled.converged, unscaled.reason
    assert scaled.converged, scaled.reason
    np.testing.assert_allclose(scaled.state, unscaled.state, rtol=2e-9, atol=2e-11)
