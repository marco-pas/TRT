import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
try:
    import pytest
except ImportError:
    pytest = None

from trt_jfnk.benchmarks.mms import MMSConfig, MMSProblem1D, SymbolicTRTMMS
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy, build_scale_context



def test_symbolic_mms_exactness():
    """Verify that SymPy symbolic source terms Q_E and Q_T are strictly periodic."""
    config = MMSConfig(p=3, m=3, length=1.0)
    symbolic = SymbolicTRTMMS(config)

    t = 0.05
    x_left = 0.0
    x_right = 1.0

    # Periodicity of fields and manufactured sources
    assert np.isclose(symbolic.E_exact(x_left, t), symbolic.E_exact(x_right, t), atol=1e-15)
    assert np.isclose(symbolic.T_exact(x_left, t), symbolic.T_exact(x_right, t), atol=1e-15)
    assert np.isclose(symbolic.QE(x_left, t), symbolic.QE(x_right, t), atol=1e-14)
    assert np.isclose(symbolic.QT(x_left, t), symbolic.QT(x_right, t), atol=1e-14)


def test_mms_negative_diffusion_matrix_matches_jvp():
    """Verify that the sparse periodic negative diffusion matrix matches JAX JVP."""
    problem = MMSProblem1D(nx=16, config=MMSConfig(p=3, m=3))
    rng = np.random.RandomState(42)

    E_test = jnp.array(rng.uniform(0.5, 1.5, size=problem.nx))
    T_test = jnp.array(rng.uniform(0.8, 1.2, size=problem.nx))
    v_test = jnp.array(rng.uniform(-0.1, 0.1, size=problem.nx))
    u_test = problem.pack(E_test, T_test)

    # JAX JVP of diffusion with respect to E
    action = jax.jvp(lambda e: problem.diffusion(e, T_test), (E_test,), (v_test,))[1]

    # Sparse matrix multiplication
    A = problem.negative_diffusion_matrix(u_test)
    expected = -A @ np.asarray(v_test)

    np.testing.assert_allclose(action, expected, atol=1e-13, rtol=1e-13)


def test_mms_schur_preconditioner_inversion():
    """Verify that Schur preconditioner solves the physical block Jacobian system."""
    problem = MMSProblem1D(nx=12, config=MMSConfig(p=3, m=3))
    u = problem.initial(0.0)
    dt = 0.002
    context = build_scale_context(u, problem.size, ScalePolicy("none"))

    a, b, c, d = map(np.asarray, problem.local_blocks(u, dt))
    A = np.diag(a) + dt * problem.negative_diffusion_matrix(u).toarray()
    M = np.block([[A, np.diag(b)], [np.diag(c), np.diag(d)]])

    rhs = jnp.linspace(-1.0, 1.0, problem.total_size)
    prec_fn = problem.preconditioner(dt, kind="schur")(u, context)
    sol = prec_fn(rhs)

    expected = np.linalg.solve(M, np.asarray(rhs))
    np.testing.assert_allclose(sol, expected, atol=1e-12, rtol=1e-11)


def test_mms_spatial_second_order_convergence(p: int = 3, m: int = 3):
    """Verify O(dx^2) spatial convergence across refinement levels for given (p, m)."""
    config = MMSConfig(p=p, m=m)
    dt = 0.0002
    t_final = 0.01
    options = NewtonOptions(atol=1e-12, rtol=1e-10, krylov=KrylovOptions(rtol=1e-10, atol=1e-12))

    errors_E = []
    errors_T = []
    resolutions = [16, 32, 64]

    for nx in resolutions:
        problem = MMSProblem1D(nx=nx, config=config)
        state = problem.initial(0.0)
        n_steps = int(round(t_final / dt))
        prec = problem.preconditioner(dt, kind="schur")

        for s in range(n_steps):
            time = (s + 1) * dt
            old_state = state
            res = newton_krylov(
                lambda u: problem.residual(u, old_state, dt, time),
                old_state,
                problem.size,
                ScalePolicy("none"),
                options,
                problem.admissible,
                prec,
            )
            assert res.converged
            state = res.state

        l2_E, _, l2_T, _ = problem.compute_errors(state, t_final)
        errors_E.append(l2_E)
        errors_T.append(l2_T)

    # Check convergence order between 32 and 64
    rate_E = np.log2(errors_E[1] / errors_E[2])
    rate_T = np.log2(errors_T[1] / errors_T[2])

    # Second-order asymptotic spatial convergence O(dx^2): rate should be ~2.0
    assert rate_E > 1.90, f"Expected order ~2.0 for E, got {rate_E:.3f}"
    assert rate_T > 1.90, f"Expected order ~2.0 for T, got {rate_T:.3f}"

