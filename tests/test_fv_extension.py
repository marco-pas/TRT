import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from trt_jfnk.benchmarks.fv_problems import FVProblem
from trt_jfnk.adjoint.trajectory import terminal_adjoint
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions,newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy,build_scale_context


@pytest.mark.parametrize('case',['marshak','hotspot','inclusion'])
def test_energy_identity_and_frozen_diffusion(case):
    p=FVProblem(case,nx=5,ny=4)
    T=.3+.1*p.X+.03*p.Y; E=.02+.01*p.X**2+.005*p.Y
    u=p.pack(E,T); dt=.001; time=.002
    r=p.residual(u,p.initial(),dt,time)
    defect=p.energy(u)-p.energy(p.initial())-dt*p.input_power(u,time)
    np.testing.assert_allclose(defect,jnp.sum(r)*p.dx*p.dy,atol=1e-14)
    v=jnp.cos(p.X+2*p.Y)
    action=jax.jvp(lambda e:p.diffusion(e,T),(E,),(v,))[1]
    np.testing.assert_allclose(action.ravel(),-p.negative_diffusion_matrix(u)@np.asarray(v).ravel(),atol=1e-13)
    if case!='marshak':
        assert float(jnp.max(jnp.abs(p.source(time)[:,1:]-p.source(time)[:,:-1])))>1e-4


def test_schur_inverse_and_jvp_vjp():
    p=FVProblem(nx=4,ny=3); u=p.initial(); dt=.003
    context=build_scale_context(u,p.size,ScalePolicy('fixed',(.03,.4),(.2,.7)))
    a,b,c,d=map(np.asarray,p.local_blocks(u,dt))
    A=np.diag(a)+dt*p.negative_diffusion_matrix(u).toarray()
    M=np.block([[A,np.diag(b)],[np.diag(c),np.diag(d)]])
    rhs=jnp.linspace(-1,1,u.size)
    answer=p.preconditioner(dt,'schur')(u,context)(rhs)
    expected=np.linalg.solve(M,np.asarray(context.residual_scale*rhs))/np.asarray(context.state_scale)
    np.testing.assert_allclose(answer,expected,rtol=1e-11,atol=1e-12)
    f=lambda z:p.residual(z,u,dt,dt)
    w=jnp.cos(rhs); Jv=jax.jvp(f,(u,),(rhs,))[1]
    _,pb=jax.vjp(f,u)
    np.testing.assert_allclose(w@Jv,pb(w)[0]@rhs,rtol=1e-12,atol=1e-12)


def test_multistep_adjoint_taylor_and_dense_oracle():
    p=FVProblem(nx=3,ny=2,exponent=1.); dt=.002
    opts=NewtonOptions(atol=1e-14,rtol=1e-12,krylov=KrylovOptions(rtol=1e-12,atol=1e-14))
    residual=lambda n,new,old:p.residual(new,old,dt,(n+1)*dt)
    objective=lambda u:jnp.mean(u**2)
    def forward(u):
        states=[u]
        for n in range(2):
            old=states[-1]
            result=newton_krylov(lambda z:residual(n,z,old),old,p.size,ScalePolicy('none'),opts,p.admissible)
            assert result.converged
            states.append(result.state)
        return states
    initial=p.initial(); states=forward(initial)
    g,_=terminal_adjoint(states,residual,objective,opts.krylov)
    dense=jax.grad(objective)(states[-1])
    for n in [1,0]:
        A=jax.jacfwd(lambda z:residual(n,z,states[n]))(states[n+1])
        B=jax.jacfwd(lambda z:residual(n,states[n+1],z))(states[n])
        dense=-B.T@jnp.linalg.solve(A.T,dense)
    np.testing.assert_allclose(g,dense,rtol=1e-9,atol=1e-11)
    direction=jnp.linspace(.01,.03,initial.size)
    errors=[]
    for h in [1e-2,5e-3,2.5e-3]:
        errors.append(abs(float(objective(forward(initial+h*direction)[-1])-objective(states[-1])-h*g@direction)))
    assert errors[0]/errors[1]>3.5
    assert errors[1]/errors[2]>3.5
