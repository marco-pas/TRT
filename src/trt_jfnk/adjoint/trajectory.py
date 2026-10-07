"""Terminal-objective discrete adjoint; store-all reference, unscaled physical residual.

Differentiates converged implicit roots, not Newton iterations. Parameter dependence
is confined to the initial condition in the supplied inverse example.
"""
import jax
import jax.numpy as jnp
from trt_jfnk.adjoint.implicit_step import build_transpose_action
from trt_jfnk.solvers.krylov import solve_krylov


def terminal_adjoint(states, residual_at_step, objective, options):
    """residual_at_step(n,new,old); states contains u_0 through u_N.

    Returns cotangent at u_0 and linear-solver diagnostics in reverse time order.
    There are no running costs or direct parameter derivatives in this interface.
    """
    p=jax.grad(objective)(states[-1])
    diagnostics=[]
    for n in reversed(range(len(states)-1)):
        new,old=states[n+1],states[n]
        Atranspose=build_transpose_action(lambda z:residual_at_step(n,z,old),new)
        result=solve_krylov(Atranspose,p,options)
        if not result.converged:
            raise RuntimeError(f'adjoint failed at step {n}: residual {result.residual_norm}')
        _,pullback=jax.vjp(lambda z:residual_at_step(n,new,z),old)
        p=-pullback(result.solution)[0]
        diagnostics.append(result)
    return p,diagnostics
