# Thermal Radiative Transfer (TRT) Non-Linear Benchmarks

The gray TRT system couples radiation energy density $E(x, t)$ and material temperature $T(x, t)$:
$$\frac{\partial E}{\partial t} - \frac{\partial}{\partial x}\left(D(T)\frac{\partial E}{\partial x}\right) = -c \sigma_a(T)\left(E - a T^4\right) + Q_E$$
$$\frac{\partial e(T)}{\partial t} = c \sigma_a(T)\left(E - a T^4\right) + Q_T$$
with diffusion coefficient $D(T) = \frac{c}{3\sigma_a(T)}$, power-law opacity $\sigma_a(T) = \sigma_0 (T_{\text{ref}}/T_{\text{eff}})^p$, heat capacity $c_v(T) = c_{v0} + c_{v1} T^m$, and internal energy $e(T) = \int_0^T c_v(T') dT'$.

Four complementary benchmarks evaluate the Jacobian-Free Newton–Krylov (JFNK) solver across 7 configurations ($N_x = 128$, 100 time steps). **Solver Time** measures cumulative time inside `newton_krylov`.

---

## Summary Overview of the 4 Benchmarks

| Benchmark | Primary Role | Active Non-Linearities | Reference Ground Truth | Key Solver Challenge |
| :--- | :--- | :--- | :--- | :--- |
| **1. Perturbed Su-Olson** | Physical Benchmarking | Non-linear EOS ($m = 3 + \delta$), cubic emission $T^4$ | Base linear state ($\delta = 0$) | FD difference quotient breakdown on $T^{4+\delta}$ |
| **2. Perturbed Marshak** | Physical Benchmarking | $E \ne a T^4$ decoupling, Kramers opacity ($p=3$), Robin BC | Converged Schur reference | Multiscale non-equilibrium precursor wave |
| **3. MMS Sine** | Code Verification | All 4 mechanisms ($D(T), \sigma(T), aT^4, c_v(T)$) | Exact analytic ($< 10^{-16}$) | Asymptotic $\mathcal{O}(\Delta x^2)$ order verification |
| **4. MMS Stiff** | Extreme Stress Test | 9 orders in $\sigma_a$, 12 orders in $E$, sinc peak | Exact analytic ($< 10^{-16}$) | Ill-conditioning ($\kappa > 10^{12}$); unpreconditioned stall |

---

## 1. Perturbed Su-Olson Benchmark ($m = 3 + \delta, \delta = 0.01$)

### Theory
The standard Su-Olson problem assumes a cubic Debye heat capacity ($c_v = \alpha T^3, m=3$), which linearizes the material equation under $v = a T^4$. Real high-energy-density media deviate from Debye scaling due to ionization. We introduce an exponent perturbation:
$$m = 3 + \delta, \quad \delta = 0.01 \implies c_v(T) = c_{v0} + \alpha T^{3+\delta}, \quad e(T) = c_{v0} T + \frac{\alpha}{4+\delta} T^{4+\delta}$$
Taylor expansion around $\delta = 0$ reveals the non-linear logarithmic sensitivity:
$$e_\delta(T) = e_0(T) + \delta \left[ \frac{\alpha}{4} T^4 \left( \ln T - \frac{1}{4} \right) \right] + \mathcal{O}(\delta^2)$$

- **Setup**: Periodic domain $x \in [-5, 5]$, $N_x = 128$, $\Delta t = 5 \times 10^{-4}$ (100 steps to $t=0.05$), $D = 1/3, \sigma_0 = 1.0, c_{v0} = 0.05, \alpha = 4.0$, Gaussian source pulse active for $t \in [0, 0.02]$.

### Results (128 points, 100 steps)
| Label | JVP | Scale policy | Physics preconditioner | Status | Steps | Solver Time (s) | Avg GMRES/Step | $\Delta T$ ($L_2$) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **FD** | finite difference | none | none | Stalled (step 14) | 14/100 | 6.71 | 29.8 | $2.639 \times 10^{-2}$ |
| **AD** | automatic diff | none | none | **Converged** | 100/100 | 26.76 | 26.7 | $1.316 \times 10^{-4}$ |
| **S-FD** | finite difference | global | none | Stalled (step 14) | 14/100 | 6.78 | 29.8 | $2.639 \times 10^{-2}$ |
| **S-AD** | automatic diff | global/local | none | **Converged** | 100/100 | 30.55 | 26.8 | $1.316 \times 10^{-4}$ |
| **AD-BD** | automatic diff | none | block diagonal | **Converged** | 100/100 | 31.39 | 13.1 | $1.316 \times 10^{-4}$ |
| **S-AD-BD** | automatic diff | global/local | block diagonal | **Converged** | 100/100 | 31.43 | 13.1 | $1.316 \times 10^{-4}$ |
| **S-AD-Schur** | automatic diff | global/local | Schur-AMG | **Converged** | 100/100 | **29.11** | **2.0** | $1.316 \times 10^{-4}$ |

- **Key Takeaways**:
  1. Finite difference JVP fails at step 14 because difference quotient errors on $T^{4+\delta}$ corrupt GMRES orthogonalization. Exact AD JVP converges smoothly.
  2. Block-diagonal preconditioning halves GMRES iterations (26.8 $\to$ 13.1). Schur preconditioning achieves **2.0 GMRES iterations/step**.
- **Plots**: `results/perturbed_su_olson/su_olson_perturbed_comparison_<LABEL>.png`
- **Run Command**:
  ```bash
  PYTHONPATH=src /scratch/marcopas/.venv/bin/python3 scripts/compare_perturbed_su_olson.py --nx 128 --steps 100 --dt 0.0005 --delta 0.01
  ```

---

## 2. Perturbed (Non-Equilibrium) Marshak Wave Benchmark

### Theory
Classical Marshak diffusion assumes equilibrium ($E = a T^4$). When **equilibrium is not reached yet** ($E \ne a T^4$), radiation and matter decouple into two separate equations coupled by finite exchange rate $c \sigma_a(T)(E - a T^4)$:
$$\frac{\partial E}{\partial t} - \frac{\partial}{\partial x}\left(D(T)\frac{\partial E}{\partial x}\right) = -c \sigma_a(T)\left(E - a T^4\right), \qquad c_v \frac{\partial T}{\partial t} = c \sigma_a(T)\left(E - a T^4\right)$$
with Kramers opacity $\sigma_a(T) = \sigma_0 T^{-3} \implies D(T) \propto T^3$ and non-linear Marshak Robin boundary condition at $x = 0$:
$$E(0, t) - 2 D(T(0, t)) \left.\frac{\partial E}{\partial x}\right|_{x=0} = a T_{\text{bath}}^4$$
Radiation diffuses ahead of the thermal wave, creating a non-equilibrium precursor where $E \gg a T^4$ (up to $40\times$).

- **Setup**: Domain $x \in [0, 1.0]$, $N_x = 128$, $\Delta t = 0.001$ (100 steps to $t=0.10$), $T_{\text{cold}} = 0.2, T_{\text{bath}} = 1.0, c_v = 1.0, \sigma_0 = 1.0, p = 3.0$.

### Results (128 points, 100 steps)
| Label | JVP | Scale policy | Physics preconditioner | Status | Steps | Solver Time (s) | Avg GMRES/Step | Rel $L_2(T)$ |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **FD** | finite difference | none | none | **Converged** | 100/100 | 86.23 | 16.2 | $2.264 \times 10^{-10}$ |
| **AD** | automatic diff | none | none | **Converged** | 100/100 | **54.90** | 16.2 | $2.259 \times 10^{-10}$ |
| **S-FD** | finite difference | global | none | **Converged** | 100/100 | 69.80 | 17.2 | $1.068 \times 10^{-10}$ |
| **S-AD** | automatic diff | global/local | none | **Converged** | 100/100 | 59.73 | 16.9 | $1.065 \times 10^{-10}$ |
| **AD-BD** | automatic diff | none | block diagonal | **Converged** | 100/100 | 101.64 | 15.2 | $1.814 \times 10^{-10}$ |
| **S-AD-BD** | automatic diff | global/local | block diagonal | **Converged** | 100/100 | 108.18 | 15.8 | $9.424 \times 10^{-11}$ |
| **S-AD-Schur** | automatic diff | global/local | Schur-AMG | **Converged** | 100/100 | 103.84 | **7.7** | $0.000 \times 10^{00}$ |

- **Key Takeaways**:
  1. AD evaluates exact JVPs without extra residual evaluations, achieving a **$1.57\times$ speedup** over FD (54.90s vs 86.23s).
  2. Schur preconditioning reduces GMRES iterations from 16.9 to **7.7 iterations/step** ($2.2\times$ reduction).
- **Plots**: `results/marshak/marshak_comparison_<LABEL>.png`
- **Run Command**:
  ```bash
  PYTHONPATH=src /scratch/marcopas/.venv/bin/python3 scripts/compare_nonequilibrium_marshak.py --nx 128 --steps 100 --dt 0.001
  ```

---

## 3. MMS Sine Benchmark (Smooth Manufactured Solution)

### Theory
The Method of Manufactured Solutions (MMS) prescribes smooth trigonometric fields:
$$T_{\text{exact}}(x, t) = T_0 + T_1 \sin\left(\frac{2\pi x}{L}\right) e^{-\lambda_T t}, \qquad E_{\text{exact}}(x, t) = E_0 + E_1 \cos\left(\frac{2\pi x}{L}\right) e^{-\lambda_E t}$$
with $T_0 = 1.0, T_1 = 0.2, E_0 = 1.0, E_1 = 0.3$. Exact source terms $Q_E, Q_T$ derived via SymPy provide machine-precision ground truth ($< 10^{-16}$) to verify spatial second-order accuracy $\mathcal{O}(\Delta x^2)$.

- **Setup**: Periodic domain $x \in [0, 1.0]$, $N_x = 128$, $\Delta t = 2 \times 10^{-4}$ (100 steps to $t=0.02$), $p = 3, m = 3$.

### Results (128 points, 100 steps)
| Label | JVP | Scale policy | Physics preconditioner | Status | Steps | Solver Time (s) | Avg GMRES/Step | Rel $L_2(T)$ |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **FD** | finite difference | none | none | **Converged** | 100/100 | 83.42 | 23.0 | $2.018 \times 10^{-7}$ |
| **AD** | automatic diff | none | none | **Converged** | 100/100 | **51.02** | 23.0 | $2.018 \times 10^{-7}$ |
| **S-FD** | finite difference | global | none | **Converged** | 100/100 | 86.40 | 22.9 | $2.018 \times 10^{-7}$ |
| **S-AD** | automatic diff | global/local | none | **Converged** | 100/100 | 56.10 | 22.8 | $2.018 \times 10^{-7}$ |
| **AD-BD** | automatic diff | none | block diagonal | **Converged** | 100/100 | 43.81 | 7.6 | $2.018 \times 10^{-7}$ |
| **S-AD-BD** | automatic diff | global/local | block diagonal | **Converged** | 100/100 | 48.08 | 7.7 | $2.018 \times 10^{-7}$ |
| **S-AD-Schur** | automatic diff | global/local | Schur-AMG | **Converged** | 100/100 | 52.13 | **3.0** | $2.018 \times 10^{-7}$ |

- **Key Takeaways**:
  1. In the smooth regime, all 7 methods converge with identical discretization error ($L_2(E) = 2.025 \times 10^{-5}, L_2(T) = 2.018 \times 10^{-7}$).
  2. AD is **$1.64\times$ faster** than FD (51.02s vs 83.42s).
  3. Preconditioning cuts GMRES iterations from 23.0 down to **3.0 iterations/step** (Schur).
- **Plots**: `results/mms/mms_comparison_<LABEL>.png` and `results/mms/mms_convergence_rates.png`
- **Run Command**:
  ```bash
  PYTHONPATH=src /scratch/marcopas/.venv/bin/python3 scripts/compare_normal_mms_methods.py --nx 128 --steps 100 --dt 0.0002
  ```

---

## 4. MMS Stiff Benchmark (Multiscale Sinc-Exponential Solution)

### Theory
Models extreme High-Energy-Density Physics (HEDP) contrasts ($T \in [10, 10^4]$, $E \in [10^4, 10^{16}]$, $\sigma_a \in [1, 10^9]$) using a squared sinc function modulated by a Gaussian envelope:
$$\phi(x) = \left[ \frac{\sin(z)}{z} \exp\left(-\frac{(x - x_0)^2}{2 w^2}\right) \right]^2, \quad z = \frac{x - x_0}{s}$$
$$T_{\text{exact}}(x, t) = T_{\text{cold}} + (T_{\text{hot}} - T_{\text{cold}}) \phi(x) e^{-\lambda_T t}, \qquad E_{\text{exact}}(x, t) = E_{\text{cold}} + (E_{\text{hot}} - E_{\text{cold}}) \phi(x) e^{-\lambda_E t}$$
with $x_0 = 0.5, s = 0.06, w = 0.10$. Guarantees strict positivity, $C^\infty$ smoothness, and machine-zero periodic boundaries. The 9-order variation in opacity produces Jacobian condition numbers $\kappa > 10^{12}$.

- **Setup**: Periodic domain $x \in [0, 1.0]$, $N_x = 128$, $\Delta t = 10^{-5}$ (100 steps to $t=10^{-3}$), $p = 3, m = 3$.

### Results (128 points, 100 steps)
| Label | JVP | Scale policy | Physics preconditioner | Status | Steps | Solver Time (s) | Avg GMRES/Step | Rel $L_2(T)$ |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **FD** | finite difference | none | none | Stalled (step 1) | 1/100 | 1.27 | 60.0 | $4.993 \times 10^{-7}$ |
| **AD** | automatic diff | none | none | Stalled (step 1) | 1/100 | 0.36 | 60.0 | $4.993 \times 10^{-7}$ |
| **S-FD** | finite difference | global | none | Stalled (step 1) | 1/100 | 0.44 | 60.0 | $4.993 \times 10^{-7}$ |
| **S-AD** | automatic diff | global/local | none | Stalled (step 1) | 1/100 | 0.38 | 60.0 | $4.993 \times 10^{-7}$ |
| **AD-BD** | automatic diff | none | block diagonal | **Converged** | 100/100 | **55.68** | 10.5 | $4.262 \times 10^{-4}$ |
| **S-AD-BD** | automatic diff | global/local | block diagonal | **Converged** | 100/100 | 56.24 | 9.6 | $4.262 \times 10^{-4}$ |
| **S-AD-Schur** | automatic diff | global/local | Schur-AMG | **Converged** | 100/100 | 56.35 | **6.0** | $4.262 \times 10^{-4}$ |

- **Key Takeaways**:
  1. Unpreconditioned solvers fail at step 1 due to condition number $> 10^{12}$. Automatic differentiation alone cannot overcome ill-conditioning; physics preconditioning is indispensable.
  2. Block-diagonal preconditioning eliminates stiff point coupling, enabling complete 100-step convergence.
  3. Schur preconditioning reduces GMRES iterations to **6.0 iterations/step** (a 43% reduction over Block-Diagonal).
- **Plots**: `results/mms_stiff/mms_stiff_comparison_<LABEL>.png` and `results/mms_stiff/mms_stiff_convergence.png`
- **Run Command**:
  ```bash
  PYTHONPATH=src /scratch/marcopas/.venv/bin/python3 scripts/compare_stiff_methods.py --nx 128 --steps 100 --dt 1e-5
  ```
