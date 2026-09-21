# Scale-preconditioned TRT JFNK

[![TRT-JFNK CI](https://github.com/marco-pas/TRT/actions/workflows/ci.yml/badge.svg?branch=feature%2Fhlim_adjoint_opt)](https://github.com/marco-pas/TRT/actions/workflows/ci.yml)


This repository provides a modular, scale-preconditioned Jacobian-Free
Newton--Krylov (JFNK) framework for thermal radiative transfer (TRT),
supporting both CPU (SciPy) and GPU (CuPy via DLPack) solvers.

## Nonlinear equations

The new physical state is radiation energy `E` and material temperature `T`:

$$\frac{\partial E}{\partial t}
=\nabla\cdot\left(D(T)\nabla E\right)
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
1+\theta\Delta t\sigma_a &
\theta\Delta t\partial_T g\\
-\theta\Delta t\sigma_a &
c_v(T)-\theta\Delta t\partial_T g
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

# Device-resident GPU AD/Krylov path via CuPy
python scripts/run_gray_trt.py --jvp ad --krylov-backend cupy --platform gpu

# Independent residual equilibration instead of R=S
python scripts/run_gray_trt.py --scaling fixed \
  --radiation-scale 0.01 --temperature-scale 0.2 \
  --residual-radiation-scale 0.1 --residual-temperature-scale 0.02
```

### More runs with physical benchmarks
```
# Nonlinear Marshak wave
python scripts/run_fv_benchmark.py \
  --case marshak --nx 32 --ny 4 \
  --steps 20 --dt 0.002 \
  --preconditioner local-block \
  --output-dir results/marshak

# Genuine two-dimensional heating
python scripts/run_fv_benchmark.py \
  --case hotspot --nx 32 --ny 32 \
  --steps 20 --dt 0.002 \
  --preconditioner schur \
  --output-dir results/hotspot

# Two-dimensional heterogeneous material
python scripts/run_fv_benchmark.py \
  --case inclusion --nx 32 --ny 32 \
  --contrast 100 --steps 20 --dt 0.002 \
  --preconditioner schur \
  --output-dir results/inclusion

# More demanding nonlinear/heterogeneous case
python scripts/run_fv_benchmark.py \
  --case inclusion --nx 64 --ny 64 \
  --cold 0.05 --floor 0.005 \
  --exponent 3 --contrast 10000 \
  --dt 0.01 --steps 10 \
  --preconditioner schur \
  --output-dir results/inclusion_stiff
```

### Test the adjoint with simple inverse problem
```
python scripts/run_initial_inverse.py \
  --metric identity --iterations 8 \
  --output-dir results/inverse_identity

python scripts/run_initial_inverse.py \
  --metric diagonal --iterations 8 \
  --output-dir results/inverse_diagonal
```


## Hypothesis Benchmarks (H1 & H4)

Dedicated benchmark drivers are provided to systematically investigate the core hypotheses:
- **Hypothesis H1 (Tangent Accuracy)**: AD directional derivatives remain exact up to machine precision, whereas finite-difference (FD) approximations suffer truncation error ($\epsilon > \sqrt{\epsilon_{\text{mach}}}$) and cancellation error ($\epsilon < \sqrt{\epsilon_{\text{mach}}}$), especially pronounced in single precision (FP32).
- **Hypothesis H4 (Reduced Precision)**: Combining AD JVPs, dimensionless state scaling, and local-block preconditioning enables robust, convergent FP32 solutions even under stiff TRT coupling ($\epsilon = 10^{-4}$).

### Running via Command Line

Execute benchmarks using `scripts/run_benchmarks_su_olson.py`:

```bash
# Run both H1 and H4 on CPU with SciPy
python scripts/run_benchmarks_su_olson.py --target all --platform cpu --krylov-backend scipy

# Run only H1 (tangent accuracy across FD step sizes)
python scripts/run_benchmarks_su_olson.py --target h1

# Run H4 on GPU with CuPy (tracks exact Krylov iteration counts on device)
python scripts/run_benchmarks_su_olson.py --target h4 --platform gpu --krylov-backend cupy --steps 20
```

#### CLI Options

| Argument | Choices / Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--target` | `all`, `h1`, `h4` | `all` | Benchmark suite to execute |
| `--platform` | `auto`, `cpu`, `gpu` | `auto` | Hardware platform for JAX |
| `--krylov-backend` | `scipy`, `cupy`, `jax` | `scipy` | Linear solver backend (`scipy` for CPU, `cupy` for GPU) |
| `--steps` | `int` | `20` | Number of time steps for H4 multi-step solve |
| `--nx` | `int` | `65` | Grid points along $x$ |
| `--ny` | `int` | `8` | Grid points along $y$ |
| `--output-dir` | `path` | `results/benchmarks_su_olson` | Directory where benchmark CSVs are written |

### Running via Python API

Both benchmarks can also be driven and analyzed directly from Python:

#### Testing H1 in Python (Tangent Accuracy)
```python
from trt_jfnk.benchmarks.benchmark_h1_tangent import run_tangent_accuracy_benchmark
from trt_jfnk.benchmarks.configurations import SuOlsonBenchmarkConfig

# Configure grid and perturbation range
config = SuOlsonBenchmarkConfig(nx=65, ny=8)
epsilons = [10.0**p for p in range(-12, 1)]

# Evaluate AD, FD-forward, and FD-central in FP64 and FP32
records = run_tangent_accuracy_benchmark(config=config, epsilons=epsilons)

for r in records:
    print(f"{r.scheme:12s} | prec={r.precision} | eps={r.epsilon:.1e} | rel_err={r.rel_error:.2e}")
```

#### Testing H4 in Python (Reduced Precision & Ablation)
```python
from trt_jfnk.benchmarks.benchmark_h4_precision import run_reduced_precision_benchmark

# Run multi-step solver across coupling stiffnesses (eps = 1e-2, 1e-4)
# Evaluates: FP64 Ref, FP32 Baseline, FP32+AD, FP32+AD+Scaling, FP32 Full H4, FP32 FD Ablation
results = run_reduced_precision_benchmark(
    coupling_epsilons=[1.0e-2, 1.0e-4],
    steps=20,
    nx=65,
    ny=8,
    krylov_backend="cupy",  # or "scipy" on CPU
)

for r in results:
    status = "CONVERGED" if r.converged else "FAILED"
    print(
        f"{r.name:38s} | eps={r.coupling_epsilon:.0e} | {status:9s} | "
        f"Krylov={r.total_krylov:<5d} | RelErr={r.rel_error_to_fp64_ref:.2e}"
    )
```

