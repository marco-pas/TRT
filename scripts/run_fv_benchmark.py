#!/usr/bin/env python3
"""Forward benchmarks. All physics is nondimensional with c=a=rho=1."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import argparse
import csv
import json
import time
from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
from trt_jfnk.benchmarks.fv_problems import FVProblem
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.jvp import JVPOptions
from trt_jfnk.solvers.scaling import ScalePolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=['marshak','hotspot','inclusion'],default='marshak')
    for name,default in [('nx',32),('ny',4),('steps',20)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('dt',.002),('cold',.2),('bath',1.),('sigma',1.),
                         ('exponent',3.),('contrast',100.),('floor',.01),('amplitude',5.),('pulse-end',.05)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--jvp',choices=['ad','fd'],default='ad')
    p.add_argument('--scaling',choices=['none','state','fixed'],default='state')
    p.add_argument('--preconditioner',choices=['none','local-block','schur'],default='local-block')
    p.add_argument('--output-dir',type=Path,default=Path('results/fv'))
    args=p.parse_args()
    if args.dt<=0 or args.steps<1:
        p.error('dt and steps must be positive')
    jax.config.update('jax_enable_x64',True)
    problem=FVProblem(**{k:getattr(args,k) for k in FVProblem.__dataclass_fields__})
    options=NewtonOptions(atol=1e-11,rtol=1e-9,max_iterations=30,jvp=JVPOptions(mode=args.jvp),
                          krylov=KrylovOptions(rtol=1e-9,max_iterations=1000))
    policy=ScalePolicy(args.scaling,(args.cold**4,args.cold))
    prec=problem.preconditioner(args.dt,args.preconditioner)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy(); config['output_dir']=str(args.output_dir)
    config.update(jax_version=jax.__version__,devices=str(jax.devices()),normalization='c=a=rho=1')
    (args.output_dir/'config.json').write_text(json.dumps(config,indent=2))
    u=problem.initial(); records=[]
    for n in range(args.steps):
        old=u; t=(n+1)*args.dt
        residual=lambda v:problem.residual(v,old,args.dt,t)
        start=time.perf_counter()
        result=newton_krylov(residual,old,problem.size,policy,options,problem.admissible,prec)
        jax.block_until_ready(result.state)
        elapsed=time.perf_counter()-start
        candidate=result.state
        defect=float(problem.energy(candidate)-problem.energy(old)-args.dt*problem.input_power(candidate,t))
        E,T=problem.unpack(candidate); sigma,D=problem.coefficients(T)
        physical=np.asarray(residual(candidate))
        row=dict(step=n+1,time=t,converged=result.converged,newton=result.iterations,
                 krylov=result.total_krylov_iterations,seconds=elapsed,reason=result.reason,
                 physical_residual=float(np.linalg.norm(physical)),scaled_residual=result.scaled_residual_norm,
                 energy_defect=defect,min_E=float(E.min()),min_T=float(T.min()),
                 exchange_stiffness=float(args.dt*jnp.max(sigma*(1+4*T**3/problem.material.heat_capacity(T)))),
                 diffusion_stiffness=float(args.dt*jnp.max(D)*(problem.dx**-2+problem.dy**-2)))
        records.append(row)
        with (args.output_dir/'metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=row.keys()); writer.writeheader(); writer.writerows(records)
        print(f'{n+1}: converged={result.converged}, Newton={result.iterations}, Krylov={result.total_krylov_iterations}')
        if not result.converged:
            np.savez(args.output_dir/'failed_state.npz',state=np.asarray(candidate),last_accepted=np.asarray(old))
            raise SystemExit('Step failed; trajectory stopped. See metrics.csv.')
        u=candidate
    E,T=problem.unpack(u)
    np.savez(args.output_dir/'state.npz',x=np.asarray(problem.x),y=np.asarray(problem.y),
             E=np.asarray(E),T=np.asarray(T),time=args.steps*args.dt)


if __name__=='__main__':
    main()
