#!/usr/bin/env python3
"""Small initial-equilibrium reconstruction"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import csv
import json
import jax
import jax.numpy as jnp
import numpy as np
from trt_jfnk.benchmarks.fv_problems import FVProblem
from trt_jfnk.adjoint.trajectory import terminal_adjoint
from trt_jfnk.solvers.newton import NewtonOptions,newton_krylov
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.scaling import ScalePolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--metric',choices=['identity','diagonal'],default='identity')
    parser.add_argument('--output-dir',type=Path,default=Path('results/inverse'))
    args=parser.parse_args()
    if args.iterations<1: parser.error('iterations must be positive')
    jax.config.update('jax_enable_x64',True)
    problem=FVProblem(nx=12,ny=2,cold=.3,exponent=1.)
    dt=.005; steps=4; beta=1e-6
    basis=jnp.stack([jnp.cos(k*jnp.pi*problem.X) for k in range(4)],axis=-1)
    # Unequal coefficient scales give a controlled optimization-conditioning example.
    scales=jnp.array([1.,.5,.2,.1])
    def initial(m):
        T=problem.cold*jnp.exp(basis@(scales*m))
        return problem.pack(T**4,T)
    options=NewtonOptions(atol=1e-13,rtol=1e-11,max_iterations=30,
                          krylov=KrylovOptions(rtol=1e-12,atol=1e-14,max_iterations=1000))
    def residual(n,new,old):
        return problem.residual(new,old,dt,(n+1)*dt)
    def forward(m):
        states=[initial(m)]
        for n in range(steps):
            old=states[-1]
            result=newton_krylov(lambda z:residual(n,z,old),old,problem.size,
                ScalePolicy('none'),options,problem.admissible,problem.preconditioner(dt,'local-block'))
            if not result.converged: raise RuntimeError('forward step failed: '+result.reason)
            states.append(result.state)
        return states
    truth=jnp.array([.1,-.2,.15,.1]); target=forward(truth)[-1]
    weights=jnp.concatenate((jnp.full(problem.size,1/problem.cold**4),jnp.full(problem.size,1/problem.cold)))
    def objective(u): return .5*jnp.mean((weights*(u-target))**2)
    def evaluate(m,gradient=False):
        states=forward(m)
        value=float(objective(states[-1])+.5*beta*jnp.dot(scales*m,scales*m))
        if not gradient: return value
        p,_=terminal_adjoint(states,residual,objective,options.krylov)
        _,pullback=jax.vjp(initial,m)
        return value,pullback(p)[0]+beta*scales**2*m
    args.output_dir.mkdir(parents=True,exist_ok=True)
    m=jnp.zeros(4); value,g=evaluate(m,True)
    direction=jnp.array([.3,-.4,.2,.1]); taylor=[]
    for h in [1e-2,5e-3,2.5e-3,1.25e-3]:
        error=abs(evaluate(m+h*direction)-value-h*float(g@direction))
        taylor.append(dict(h=h,remainder=error))
    with (args.output_dir/'taylor.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=taylor[0]); w.writeheader(); w.writerows(taylor)
    records=[]
    for k in range(args.iterations):
        value,g=evaluate(m,True)
        d=-g if args.metric=='identity' else -g/scales**2
        slope=float(g@d)
        if float(jnp.linalg.norm(g))<1e-10: break
        alpha=1.; accepted=False
        for _ in range(20):
            trial=m+alpha*d
            try: new_value=evaluate(trial)
            except RuntimeError: new_value=float('inf')
            if np.isfinite(new_value) and new_value<=value+1e-4*alpha*slope:
                accepted=True; break
            alpha*=.5
        if not accepted: raise RuntimeError('line search exhausted')
        records.append(dict(iteration=k,cost_before=value,cost_after=new_value,
                            gradient_norm=float(jnp.linalg.norm(g)),alpha=alpha))
        m=trial
        print(records[-1])
    with (args.output_dir/'optimization.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['iteration','cost_before','cost_after','gradient_norm','alpha'])
        w.writeheader(); w.writerows(records)
    np.savez(args.output_dir/'inverse.npz',parameters=np.asarray(m),truth=np.asarray(truth),
             recovered=np.asarray(forward(m)[-1]),target=np.asarray(target))
    (args.output_dir/'config.json').write_text(json.dumps(dict(metric=args.metric,nx=12,ny=2,
        dt=dt,steps=steps,beta=beta,coefficient_scales=np.asarray(scales).tolist(),
        note='Fixed objective; coefficient-metric illustration, not Tran duality reproduction'),indent=2))


if __name__=='__main__': main()
