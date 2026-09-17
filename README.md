# Scale-preconditioned TRT JFNK

This is an additive implementation for the upstream
[`JFNK-ADvsFD`](https://github.com/marco-pas/JFNK-ADvsFD) repository. It leaves
`raddiffSolver.py` intact and imports its Crank--Nicolson residual, Laplacian,
source, initial-condition, and boundary-condition functions.

## Nonlinear equations

The new physical state is radiation energy `E` and material temperature `T`:

$$\frac{\partial E}{\partial t}
=\nabla\!\cdot\left(D(T)\nabla E\right)
-\sigma_a(T)\left(E-aT^4\right)+Q,$$

$$
\frac{\partial e(T)}{\partial t}
=\sigma_a(T)\left(E-aT^4\right).
$$

The constitutive laws are

$$T_{\mathrm{eff}}=\sqrt{T^2+T_f^2},\qquad
\sigma_a(T)=\sigma_{a0}\left(\frac{T_{\mathrm{ref}}}{T_{\mathrm{eff}}}\right)^p,
\qquad
D(T)=\frac{c}{3\sigma_t(T)},
$$

$$
c_v(T)=c_{v0}+c_{v1}T^m,\qquad
e(T)=c_{v0}T+\frac{c_{v1}}{m+1}T^{m+1}.
$$

Thus the JVP differentiates through four nonlinear mechanisms: Planck
emission, absorption opacity, variable-coefficient diffusion, and material
internal energy.  The default theta value is 1 (backward Euler).

For one time step, the residual is

$$
F_E=E^{n+1}-E^n-\Delta t\left[\theta f_E^{n+1}+(1-\theta)f_E^n\right],
$$

$$
F_T=e(T^{n+1})-e(T^n)-\Delta t\left[\theta g^{n+1}+(1-\theta)g^n\right],
\quad g=\sigma_a(T)(E-aT^4).
$$

## Runs

Run all cases with the same physical parameters, grid, time step, tolerances,
and local preconditioner.

```bash
python scripts/run_gray_trt.py \
  --jvp ad --scaling none --preconditioner local-block \
  --output-dir results/ad_unscaled

python scripts/run_gray_trt.py \
  --jvp ad --scaling state --preconditioner local-block \
  --output-dir results/ad_scaled

python scripts/run_gray_trt.py \
  --jvp fd --fd-scheme forward --scaling state --preconditioner local-block \
  --output-dir results/fd_scaled
```

## Scale-aware local block preconditioner

Ignoring diffusion derivatives, the retained local physical Jacobian is

$$
M_i=\begin{bmatrix}
1+\theta\Delta t\,\sigma_a &
\theta\Delta t\,\partial_T g\\
-\theta\Delta t\,\sigma_a &
c_v(T)-\theta\Delta t\,\partial_T g
\end{bmatrix}_i,
$$

$$
\partial_T g=\sigma_a'(T)(E-aT^4)-4a\sigma_a(T)T^3.
$$

The scaled linear system uses

$$
\widehat M=R^{-1}MS,
$$

and `gray_local_block_factory` inverts its 2-by-2 cell blocks exactly.  This
keeps preconditioning consistent when state and residual scales differ.

## Other runs
```bash
# Stronger opacity nonlinearity and larger step
python scripts/run_gray_trt.py --opacity-exponent 4 --sigma-a0 3 --dt 0.01

# Pure device-resident AD/GMRES path
python scripts/run_gray_trt.py --jvp ad --krylov-backend jax --platform gpu

# Independent residual equilibration instead of R=S
python scripts/run_gray_trt.py --scaling fixed \
  --radiation-scale 0.01 --temperature-scale 0.2 \
  --residual-radiation-scale 0.1 --residual-temperature-scale 0.02
```
