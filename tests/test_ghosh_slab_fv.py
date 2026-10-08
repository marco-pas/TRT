import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from trt_jfnk.benchmarks.ghosh_slab_analytic import GhoshSlabAnalyticBenchmark
from trt_jfnk.benchmarks.ghosh_slab_fv import GhoshSlabFVProblem
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy


def test_steady_state_discrete_exact():
    """Verify that cell-centered FV steady-state satisfies analytical steady state to machine precision."""
    problem = GhoshSlabFVProblem(b=1.0, nx=20, ny=1)
    A = problem.negative_diffusion_matrix()

    # The affine drive on cell 0 is w0 / dx
    rhs = np.zeros(problem.size)
    rhs[0] = (problem.w0 / problem.dx) * 1.0

    u_num = np.linalg.solve(A.toarray(), rhs)
    u_exact, _ = problem.steady_state()

    np.testing.assert_allclose(u_num, u_exact.ravel(), atol=1e-14, rtol=1e-14)


def test_energy_conservation_discrete():
    """Verify discrete global energy conservation identity to machine precision."""
    problem = GhoshSlabFVProblem(b=1.0, epsilon=0.1, nx=15, ny=2)
    dt = 0.005

    u_test = jnp.array(np.random.RandomState(42).rand(problem.nx, problem.ny))
    v_test = jnp.array(np.random.RandomState(43).rand(problem.nx, problem.ny))
    state = problem.pack(u_test, v_test)
    initial = problem.initial()

    res = problem.residual(state, initial, dt=dt)
    defect = (
        problem.energy(state)
        - problem.energy(initial)
        - dt * problem.input_power(state)
    )

    # In discrete conservation: defect == sum(res) * dx * dy
    np.testing.assert_allclose(defect, jnp.sum(res) * problem.dx * problem.dy, atol=1e-14)


def test_negative_diffusion_matrix_matches_jvp():
    """Verify that the sparse negative diffusion matrix matches JAX JVP for both 1D and 2D."""
    for ny in (1, 3):
        problem = GhoshSlabFVProblem(b=1.0, nx=12, ny=ny)
        u_test = jnp.array(np.random.RandomState(101).rand(problem.nx, problem.ny))
        v_test = jnp.array(np.random.RandomState(102).rand(problem.nx, problem.ny))

        action = jax.jvp(problem.diffusion, (u_test,), (v_test,))[1]
        A = problem.negative_diffusion_matrix()
        expected = -A @ np.asarray(v_test).ravel()

        np.testing.assert_allclose(action.ravel(), expected, atol=1e-13)


def test_newton_krylov_schur_preconditioner():
    """Verify that Newton-Krylov with exact Schur complement converges in 1 Newton and 1 Krylov iteration."""
    problem = GhoshSlabFVProblem(b=1.0, epsilon=0.1, nx=25, ny=1)
    dt = 0.01
    old = problem.initial()

    prec = problem.preconditioner(dt, kind="schur")
    options = NewtonOptions(
        atol=1e-10,
        rtol=1e-8,
        krylov=KrylovOptions(rtol=1e-8, backend="scipy"),
    )

    residual = lambda v: problem.residual(v, old, dt=dt)
    result = newton_krylov(
        residual,
        old,
        problem.size,
        ScalePolicy("none"),
        options,
        problem.admissible,
        prec,
    )

    assert result.converged
    assert result.iterations == 1
    assert result.total_krylov_iterations == 1
    assert result.physical_residual_norm < 1e-12


def test_transient_convergence_vs_analytical_reference():
    """Verify that transient time integration converges to Ghosh (2014) analytical benchmark."""
    b = 1.0
    epsilon = 0.1
    nx = 60
    problem = GhoshSlabFVProblem(b=b, epsilon=epsilon, nx=nx, ny=1)
    benchmark = GhoshSlabAnalyticBenchmark(b=b, epsilon=epsilon, n_roots=40)

    dt = 0.001
    steps = 10  # reaches tau = 0.01
    u = problem.initial()

    prec = problem.preconditioner(dt, kind="schur")
    options = NewtonOptions(
        atol=1e-10,
        rtol=1e-8,
        krylov=KrylovOptions(rtol=1e-8, backend="scipy"),
    )

    for step in range(steps):
        old = u
        residual = lambda v, o=old: problem.residual(v, o, dt=dt)
        result = newton_krylov(
            residual,
            old,
            problem.size,
            ScalePolicy("none"),
            options,
            problem.admissible,
            prec,
        )
        assert result.converged
        u = result.state

    u_num, v_num = problem.unpack(u)
    u_ref, v_ref = benchmark.evaluate(problem.x, tau=0.01)

    # Discretization error (L_inf and L_2) at tau = 0.01
    linf_u = float(np.max(np.abs(np.asarray(u_num[:, 0]) - u_ref)))
    linf_v = float(np.max(np.abs(np.asarray(v_num[:, 0]) - v_ref)))
    l2_u = float(np.linalg.norm(np.asarray(u_num[:, 0]) - u_ref) / np.sqrt(nx))
    l2_v = float(np.linalg.norm(np.asarray(v_num[:, 0]) - v_ref) / np.sqrt(nx))

    assert linf_u < 5.0e-3
    assert l2_u < 2.5e-3
    assert linf_v < 1.0e-3
    assert l2_v < 5.0e-4


def test_error_linf_and_l2_less_than_1e3():
    """Verify that both L_inf and L_2 errors for u and v are strictly less than 1e-3."""
    b = 1.0
    epsilon = 0.1
    nx = 40
    problem = GhoshSlabFVProblem(b=b, epsilon=epsilon, nx=nx, ny=1)
    benchmark = GhoshSlabAnalyticBenchmark(b=b, epsilon=epsilon, n_roots=40)

    dt = 0.005
    t_final = 1.0
    steps = int(round(t_final / dt))

    u = problem.initial()
    prec = problem.preconditioner(dt, kind="schur")
    options = NewtonOptions(
        atol=1e-10,
        rtol=1e-8,
        krylov=KrylovOptions(rtol=1e-8, backend="scipy"),
    )

    for step in range(steps):
        old = u
        t = (step + 1) * dt
        res = newton_krylov(
            lambda v, o=old, cur_t=t: problem.residual(v, o, dt=dt, time=cur_t),
            old,
            problem.size,
            ScalePolicy("none"),
            options,
            problem.admissible,
            prec,
        )
        assert res.converged
        u = res.state

    u_field, v_field = problem.unpack(u)
    u_num = np.asarray(u_field[:, 0])
    v_num = np.asarray(v_field[:, 0])
    u_ref, v_ref = benchmark.evaluate(problem.x, tau=t_final)

    linf_u = float(np.max(np.abs(u_num - u_ref)))
    l2_u = float(np.linalg.norm(u_num - u_ref) / np.sqrt(nx))
    linf_v = float(np.max(np.abs(v_num - v_ref)))
    l2_v = float(np.linalg.norm(v_num - v_ref) / np.sqrt(nx))

    assert linf_u < 1.0e-3, f"Radiation L_inf error {linf_u:.4e} must be < 1e-3"
    assert l2_u < 1.0e-3, f"Radiation L_2 error {l2_u:.4e} must be < 1e-3"
    assert linf_v < 1.0e-3, f"Material L_inf error {linf_v:.4e} must be < 1e-3"
    assert l2_v < 1.0e-3, f"Material L_2 error {l2_v:.4e} must be < 1e-3"
