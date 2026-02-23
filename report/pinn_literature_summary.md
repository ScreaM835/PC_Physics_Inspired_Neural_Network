# PINN Literature Summary — Practical Methodological Details

**Purpose:** Actionable implementation guidance for solving the Zerilli equation (black hole QNMs) with PINNs, extracted from four key papers.

---

## Paper 1: Wang et al. 2023 — "An Expert's Guide to Training PINNs" (arXiv:2308.08468)

### Architecture

| Parameter | Recommendation |
|---|---|
| **Width** | 128–512 neurons per hidden layer |
| **Depth** | 3–6 hidden layers |
| **Activation** | `tanh` (strongly recommended). Also acceptable: GeLU, sinusoidal. **ReLU is unsuitable** (zero second derivative breaks PDE residual computation) |
| **Initialization** | Glorot (Xavier) normal scheme |
| **Architecture** | Modified MLP (see below) generally outperforms standard MLP for nonlinear PDEs |

### Modified MLP (from Wang, Teng & Perdikaris 2021, refined here)

Two encoder layers project input $\mathbf{x}$ into feature space:

$$U = \sigma(W_1 \mathbf{x} + b_1), \quad V = \sigma(W_2 \mathbf{x} + b_2)$$

Each hidden layer $l$ computes:

$$z^{(l)} = \sigma(H^{(l)} W_z^{(l)} + b_z^{(l)})$$
$$H^{(l+1)} = (1 - z^{(l)}) \odot U + z^{(l)} \odot V$$

The final output is $f = H^{(L)} W + b$.

**Key insight:** The two encoders $U, V$ account for multiplicative interactions between input dimensions and create residual-like connections that improve gradient flow.

### Random Fourier Features (RFF) — Input Embedding

$$\gamma(\mathbf{x}) = [\cos(B\mathbf{x}),\ \sin(B\mathbf{x})]$$

where $B \in \mathbb{R}^{m \times d}$ is sampled from $\mathcal{N}(0, \sigma^2)$.

| Parameter | Recommendation |
|---|---|
| $\sigma$ (scale) | **Moderately large**, $\sigma \in [1, 10]$. Controls frequency bias of NTK eigenspace |
| Typical values in paper | $\sigma = 1.0$ (Allen-Cahn, KS, NS), $\sigma = 2.0$ (Allen-Cahn alternate), $\sigma = 10.0$ (lid-driven cavity) |

### Random Weight Factorization (RWF)

Each weight matrix is factored as:

$$W^{(l)} = \text{diag}(\exp(s^{(l)})) \cdot V^{(l)}$$

where $s \sim \mathcal{N}(\mu, \sigma_{\text{rwf}} I)$.

| Parameter | Recommendation |
|---|---|
| $\mu$ | 0.5 or 1.0 |
| $\sigma_{\text{rwf}}$ | 0.1 |

This effectively assigns a self-adaptive per-neuron learning rate. Applied **after** Glorot initialization.

### Optimizer & Learning Rate

| Parameter | Recommendation |
|---|---|
| **Optimizer** | Adam (consistently good). **No weight decay for forward problems** |
| **Initial LR** | 0.001 |
| **Schedule** | Exponential decay, rate 0.9, decay steps 2000–5000 |
| **L-BFGS** | Not used in this paper; Adam alone is sufficient with proper techniques |

### Sampling

| Parameter | Recommendation |
|---|---|
| **Strategy** | **Random resampling** at each iteration (strongly recommended over fixed full-batch — provides regularization effect) |
| **Batch size** | 4096–8192 collocation points |

### Causal Training (Section 5.1, detailed in Paper 2)

Weighted residual loss with temporal causality:

$$\mathcal{L}_r(\theta) = \frac{1}{N_t} \sum_{i=1}^{N_t} w_i \, \mathcal{L}_r(t_i, \theta)$$

where $w_i = \exp\left(-\epsilon \sum_{k=1}^{i-1} \mathcal{L}_r(t_k, \theta)\right)$.

- $\epsilon$: should be **moderately large** to ensure all weights converge to 1
- Weights computed via `stop_gradient` (no backprop through them)
- **Not a replacement for time-marching** but a crucial enhancement to it

### Time-Marching / Curriculum Training (Section 5.3)

- Divide temporal domain into windows, train sequentially
- Each window's IC from previous window's last prediction
- Recommended: 5–10 time windows depending on problem complexity

### Hyper-parameter Settings from Benchmarks

| Problem | Arch | Layers | Neurons | Act | FF $\sigma$ | RWF $\mu/\sigma$ | LR decay steps | Steps | Batch | Weighting | $\epsilon$ | Windows |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Allen-Cahn | Mod MLP | 4 | 256 | tanh | 2.0 | 0.5/0.1 | 5000 | 300K | 8192 | NTK | 1.0 | 32 chunks |
| KS equation | Mod MLP | 5 | 256 | tanh | 1.0 | — | — | 200K/win | — | Grad Norm | 1.0 | 10 |
| NS torus | Mod MLP | 4 | 256 | tanh | 1.0 | — | — | 150K/win | — | Grad Norm | 1.0 | 5 |
| NS cylinder | Mod MLP | 5 | 256 | tanh | 1.0 | 1.0/0.1 | — | 200K/win | — | — | — | 10 |
| Lid-driven | Mod MLP | 5 | 256 | tanh | 10.0 | 1.0/0.1 | — | — | — | — | — | curriculum Re |
| Stokes | — | 4 | 128 | GeLU | — | — | — | 100K | — | — | — | — |

### Overall Recommended Algorithm (Algorithm 1 from the paper)

1. Non-dimensionalize the PDE
2. Use Modified MLP + Fourier Features + RWF, tanh activation, Glorot init
3. Weighted loss: $\mathcal{L} = \lambda_{ic} \mathcal{L}_{ic} + \lambda_{bc} \mathcal{L}_{bc} + \lambda_r \mathcal{L}_r$
4. Apply causal weighting to residual loss
5. Use gradient-based loss balancing (preferred) or NTK-based scheme
6. Train with Adam, LR 0.001, exponential decay
7. Use time-marching for long temporal domains

---

## Paper 2: Wang, Sankaran & Perdikaris 2022 — "Respecting Causality" (arXiv:2203.07404)

### Core Insight

PINNs trained via gradient descent are **implicitly biased towards minimizing PDE residuals at later times first**, before resolving the initial conditions. This violates temporal causality and leads to convergence to erroneous solutions — especially for problems with strong temporal correlations (like wave propagation, exactly the Zerilli equation scenario).

### Causal Training Algorithm (Algorithm 1 — exact specification)

**Loss function:**

$$\mathcal{L}(\theta) = \frac{1}{N_t} \sum_{i=0}^{N_t} w_i \, \mathcal{L}(t_i, \theta)$$

where:
- $\mathcal{L}(t_0, \theta) = \lambda_{ic} \mathcal{L}_{ic}(\theta)$ (initial condition treated as temporal residual at $t=0$)
- $\mathcal{L}(t_i, \theta) = \frac{1}{N_x} \sum_j |\mathcal{R}_\theta(t_i, x_j)|^2$ for $1 \le i \le N_t$

**Temporal weights:**

$$w_i = \exp\left(-\epsilon \sum_{k=1}^{i-1} \mathcal{L}(t_k, \theta)\right), \quad i = 2, 3, \ldots, N_t$$

$w_1 = 1$ always. All $w_i$ initialized to 1.

**Critical implementation details:**
1. **`stop_gradient` on $w_i$**: Do NOT backpropagate through the weight computation. Use `lax.stop_gradient` (JAX) or `tensor.detach()` (PyTorch).
2. **$\epsilon$ annealing**: Use an increasing sequence, recommended: $\{\epsilon_i\} = [10^{-2}, 10^{-1}, 10^0, 10^1, 10^2]$
3. **Stopping criterion**: Terminate training when $\min_i w_i > \delta$, with recommended $\delta = 0.99$. This means all temporal weights have converged to ~1, indicating the residual is properly minimized across all time steps.
4. **$\lambda_{ic}$**: Recommended $= 10^3$ (some examples use $10^4$)

### Interaction with Time-Marching

**Causal training should NOT replace time-marching; it should enhance it.** Causality violations can still occur within each time window of a time-marching scheme.

### Interaction with Collocation Point Resampling

The algorithm **works with random resampling** — collocation points can be randomly sampled at each iteration. The only requirement is that the sampled temporal points $\{t_i\}$ form a **non-decreasing sequence** so temporal causality can be respected.

### Optimizer

- **Adam with exponential LR decay**: rate 0.9, every 5000 iterations
- **No L-BFGS used** in this paper — all results achieved with Adam alone

### Architecture Choices (from Appendix B)

| Problem | Arch | Depth | Width | $N_t$ | $N_x$ | Time Windows | Max Iter/window |
|---|---|---|---|---|---|---|---|
| Allen-Cahn (MLP) | MLP | 6 | 128 | 100 | 256 | 1 | ~300K |
| Allen-Cahn (Mod) | Mod MLP | 6 | 128 | 100 | 256 | 1 | ~300K |
| Lorenz | MLP | 5 | 512 | 256 | — | 40 (Δt=0.5) | variable |
| KS (regular) | Mod MLP | 5 | 256 | 32 | 64 | 10 (Δt=0.1) | variable |
| KS (chaotic) | Mod MLP | 10 | 128 | 32 | 256 | 5 (Δt=0.1) | variable |
| Navier-Stokes | Mod MLP | 6 | 128 | 64 | 512 | 10 (Δt=0.1) | variable |

### Key Practical Notes

- **Hard BC enforcement** recommended where possible (e.g., Fourier embedding for periodic BCs)
- **Larger batch sizes** lead to higher accuracy (parallel GPU scaling helps)
- **Stopping criterion** ($\min_i w_i > 0.99$) not only speeds training but actually **improves accuracy** — training past this point can cause overfitting
- For the Lorenz system with initial condition enforcement:  $\hat{x}_\theta(t) = x_\theta(t) \cdot t + x(0)$ — multiplying by $t$ ensures exact IC

---

## Paper 3: Wang, Teng & Perdikaris 2021 — "Gradient Flow Pathologies" (arXiv:2001.04536)

### Core Problem Identified

In PINNs, the PDE residual loss $\mathcal{L}_r$ generates much larger gradients than the boundary/IC loss terms $\mathcal{L}_{bc}, \mathcal{L}_{ic}$. This **gradient imbalance** causes the network to learn solutions that satisfy the PDE but violate boundary/initial conditions → erroneous solutions.

**Root cause:** The gradients scale as $\|\nabla_\theta \mathcal{L}_r\| \leq O(C^4) \cdot \epsilon \cdot \|\nabla_\theta \epsilon_\theta\|$ where $C$ relates to the solution's characteristic frequency. Higher-frequency or stiffer PDEs → worse imbalance.

### Learning Rate Annealing Algorithm (Algorithm 1 — exact specification)

**Given:** Loss $\mathcal{L}(\theta) = \mathcal{L}_r(\theta) + \sum_{i=1}^{M} \lambda_i \mathcal{L}_i(\theta)$

Initialize $\lambda_i = 1$ for all $i$.

At each update step (or every $k$ steps):

**(a) Compute instantaneous weights:**

$$\hat{\lambda}_i = \frac{\max_\theta |\nabla_\theta \mathcal{L}_r(\theta_n)|}{\overline{|\nabla_\theta \mathcal{L}_i(\theta_n)|}}, \quad i = 1, \ldots, M$$

where:
- **Numerator**: maximum absolute value of the gradient of $\mathcal{L}_r$ over all parameters $\theta$
- **Denominator**: mean of the absolute values of the gradient of $\mathcal{L}_i$ over all parameters $\theta$

**(b) Update via exponential moving average:**

$$\lambda_i = (1 - \alpha) \lambda_i + \alpha \hat{\lambda}_i$$

**(c) Gradient descent update:**

$$\theta_{n+1} = \theta_n - \eta \nabla_\theta \left[\mathcal{L}_r(\theta_n) + \sum_i \lambda_i \mathcal{L}_i(\theta_n)\right]$$

### Practical Details

| Parameter | Recommendation |
|---|---|
| **$\alpha$ (EMA decay)** | $\alpha \in [0.5, 0.9]$ — low sensitivity within this range. Paper uses $\alpha = 0.9$ |
| **Update frequency** | Every 10 iterations of gradient descent (not every step, to reduce overhead) |
| **Computational overhead** | Small — gradient statistics are already computed during backprop |

### The Modified MLP Architecture (Original Proposal)

This paper **introduces** the Modified MLP (called "improved fully-connected architecture"). The forward pass:

$$U = \phi(X W_1 + b_1), \quad V = \phi(X W_2 + b_2)$$
$$H^{(1)} = \phi(X W_{z,1} + b_{z,1})$$
$$Z^{(k)} = \phi(H^{(k)} W_{z,k} + b_{z,k}), \quad k = 1, \ldots, L$$
$$H^{(k+1)} = (1 - Z^{(k)}) \odot U + Z^{(k)} \odot V$$
$$f_\theta = H^{(L)} W + b$$

**Key benefits:**
1. Explicitly accounts for multiplicative interactions between different input dimensions
2. Residual connections enhance gradient flow
3. Reduces stiffness of the gradient flow dynamics (~3× reduction in leading Hessian eigenvalue)
4. Consistently outperforms standard MLP by **50–100×** across all benchmarks

### Model Comparison (from the paper)

| Model | Description | Helmholtz Error (100u/7L) |
|---|---|---|
| M1 | Original PINN (Raissi et al.) | 8.14e-02 |
| M2 | M1 + gradient balancing (Algorithm 1) | 4.52e-03 |
| M3 | M1 + Modified MLP architecture | 9.36e-03 |
| **M4** | **Modified MLP + gradient balancing** | **1.49e-03** |

### L-BFGS Compatibility

- The paper **uses only Adam** for training (initial LR $10^{-3}$, exponential decay rate 0.9, steps 1000)
- The gradient balancing algorithm is designed for gradient descent variants (Adam)
- L-BFGS compatibility: The algorithm computes per-term gradient norms, which **can** be computed alongside L-BFGS, but the paper does not test this. The EMA update of $\lambda_i$ is naturally compatible with any optimizer since it only requires the gradient values, not the optimizer state.

### Training Setup Used Throughout

| Parameter | Value |
|---|---|
| Activation | tanh |
| Mini-batch size | 128 data-points |
| Precision | Single (float32) |
| Optimizer | Adam with default settings |
| Initialization | Glorot scheme |
| Regularization | None (no dropout, no L1/L2) |
| Weight updates | Every 10 gradient descent steps |

---

## Paper 4: Raissi, Perdikaris & Karniadakis 2019 — "Original PINN Paper" (arXiv:1711.10561)

### Original PINN Formulation

**Loss function:**

$$MSE = MSE_u + MSE_f$$

$$MSE_u = \frac{1}{N_u} \sum_{i=1}^{N_u} |u(t_u^i, x_u^i) - u^i|^2$$

$$MSE_f = \frac{1}{N_f} \sum_{i=1}^{N_f} |f(t_f^i, x_f^i)|^2$$

where $MSE_u$ enforces IC/BC data fit and $MSE_f$ enforces PDE residual.

### Architecture & Optimizer

| Parameter | Value |
|---|---|
| **Optimizer** | **L-BFGS** (quasi-Newton, full-batch). For larger datasets: Adam (SGD) |
| **Typical architecture** | 9 layers, 20 neurons per layer (Burgers); 5 layers, 100 neurons (Schrödinger) |
| **Activation** | Hyperbolic tangent (tanh) |
| **Collocation points** | 10,000–20,000, generated via **Latin Hypercube Sampling** |
| **Training data** | Very small: 50–200 initial/boundary points |
| **Training time** | ~60 seconds on NVIDIA Titan X GPU (Burgers) |

### Key Findings

1. **L-BFGS is preferred** for small datasets (full-batch); Adam for larger
2. The PDE residual acts as **regularization** — prevents overfitting even with large networks
3. Deeper/wider networks → better accuracy (no overfitting thanks to physics constraint)
4. Network capacity must be sufficient for solution complexity
5. Collocation points ($N_f$) matter more than training data ($N_u$) — more collocation = better physics enforcement

### Architecture Sensitivity (Burgers equation, $N_u=100$, $N_f=10000$)

| Layers | 20 neurons | 40 neurons | 60 neurons |
|---|---|---|---|
| 2 | 7.4e-02 | 5.3e-02 | 1.0e-01 |
| 4 | 3.0e-03 | 9.4e-04 | 6.4e-04 |
| 6 | 9.6e-03 | 1.3e-03 | 6.1e-04 |
| 8 | 2.5e-03 | 9.6e-04 | 5.6e-04 |

**Takeaway:** 4+ layers with 40+ neurons is the sweet spot for this problem class. Deeper networks (8 layers) don't always help — 4–6 layers is often optimal.

---

## Consolidated Recommendations for Zerilli Equation PINN

### Why These Papers Matter for the Zerilli Equation

The Zerilli equation is a **hyperbolic PDE** (wave-like) describing gravitational perturbations of a Schwarzschild black hole. It has:
- **Temporal causality** — information propagates at finite speed → causal training is critical
- **Multi-scale behavior** — QNM ringdown has both oscillatory and exponential decay components → RFF and Modified MLP help
- **Stiffness** — the effective potential $V_l(r_*)$ introduces spatial stiffness → gradient balancing is essential
- **Long-time integration** — need to capture the QNM ringdown over multiple oscillation periods → time-marching is recommended

### Recommended Configuration

```yaml
# Architecture
network:
  type: modified_mlp           # Modified MLP with skip connections
  hidden_layers: 4-5           # Sweet spot for wave equations
  neurons_per_layer: 256       # 128-256 range
  activation: tanh             # MUST be tanh (need 2nd derivatives for wave eq)
  initialization: glorot       # Xavier/Glorot normal

# Input embedding
fourier_features:
  enabled: true
  scale_sigma: 1.0-5.0        # Start with 1.0, increase if solution has
                                # multiple frequency components
  num_features: 128-256        # Dimensionality of embedding

# Random Weight Factorization
rwf:
  enabled: true
  mu: 0.5                     # or 1.0
  sigma: 0.1

# Optimizer
optimizer:
  type: adam
  learning_rate: 0.001
  weight_decay: 0.0            # NO weight decay for forward problems
  lr_schedule:
    type: exponential_decay
    decay_rate: 0.9
    decay_steps: 2000-5000

# Loss balancing (gradient-based, from Paper 3)
gradient_balancing:
  enabled: true
  alpha: 0.9                   # EMA decay parameter
  update_every: 100            # iterations between weight updates
  formula: max_over_mean       # max|∇L_r| / mean|∇L_i|

# Causal training (from Paper 2)
causal_training:
  enabled: true
  epsilon_schedule: [0.01, 0.1, 1.0, 10.0, 100.0]  # Annealing
  lambda_ic: 1000              # Strong IC enforcement
  stopping_delta: 0.99         # Stop when min(w_i) > 0.99
  stop_gradient_weights: true  # CRITICAL: detach weights from graph

# Time-marching
time_marching:
  enabled: true
  num_windows: 5-10            # Depends on total time span
  ic_from_previous: true       # Each window starts from prev prediction

# Sampling
sampling:
  strategy: random             # Resample each iteration
  batch_size: 4096-8192
  collocation_points: 4096-8192
  ic_points: 256-512
  bc_points: 256-512
```

### Training Pipeline

1. **Non-dimensionalize** the Zerilli equation (scale $r_*$ and $t$ to $[0,1]$ or similar)
2. **Set up Modified MLP** with RFF ($\sigma=1.0$), RWF ($\mu=0.5, \sigma=0.1$), tanh, Glorot
3. **For each time window:**
   a. Set IC from previous window (or true IC for first window)
   b. For each $\epsilon$ in $[10^{-2}, 10^{-1}, 10^0, 10^1, 10^2]$:
      - Train with Adam (LR=0.001, decay 0.9/2000 steps)
      - Apply causal weights with `stop_gradient`
      - Apply gradient balancing every 100 iterations
      - Check stopping criterion: $\min_i w_i > 0.99$ → break
   c. Save checkpoint, extract prediction at final time as IC for next window
4. **Evaluate:** Compute relative $L^2$ error against FD reference solution

### Common Pitfalls to Avoid

1. **Using ReLU activation** — zero second derivative kills wave equation residual
2. **Using weight decay** — acts as unwanted regularization for forward problems
3. **Fixed collocation points** — random resampling provides better regularization
4. **Ignoring causality** — without causal weights, PINN will try to resolve the late-time ringdown before the initial pulse, producing garbage
5. **Training past convergence** — when all causal weights ≈ 1, stop! Further training degrades accuracy (overfitting observation from Paper 2)
6. **Backpropagating through causal weights** — must use `detach()`/`stop_gradient` on $w_i$
7. **Too small $\epsilon$** — network won't activate later temporal weights; too large → optimization becomes too hard
8. **Not non-dimensionalizing** — the gradient imbalance problem (Paper 3) is directly worsened by large characteristic scales in the PDE

---

## Gap Analysis: Current Implementation vs. Recommendations

### Already Implemented (Good)

| Feature | Status | Notes |
|---|---|---|
| Modified MLP | ✅ | Wang-style skip connections with U/V encoders |
| RFF embedding | ✅ | Trainable, σ=1.0, num_features=64 |
| tanh activation | ✅ | Correct for 2nd-order PDEs |
| Glorot init | ✅ | Xavier uniform (paper uses normal, but both fine) |
| Gradient balancing | ✅ | period=100, α=0.9 — matches Paper 3 recommendations |
| Adam → L-BFGS training | ✅ | Supported with frozen gradient balancing weights during L-BFGS |
| Collocation resampling | ✅ | via `PDEPointResampler` every 100 steps |
| Curriculum/Time-marching | ✅ | Expanding windows [10, 20, 30, 40, 50] |
| Gradient-enhanced residuals | ✅ | r, r_x, r_t in PDE output |
| Output transform | ✅ | `A * tanh(y)` bounds output |

### Missing or Suboptimal

| Feature | Issue | Recommendation |
|---|---|---|
| **Network width** | Currently 64 neurons per layer | Increase to **128–256** (all benchmarks in Papers 1-3 use 128–256) |
| **Random Weight Factorization** | Not implemented | Add RWF with μ=0.5, σ=0.1 — provides self-adaptive per-neuron learning rates (Paper 1, Algorithm 1 step 2) |
| **Causal training** | Disabled (`causal.enabled: false`) | **Re-enable** — critical for wave equations. Use ε annealing [0.01, 0.1, 1, 10, 100], stop_gradient on weights, stopping criterion min(w_i)>0.99 (Paper 2) |
| **LR schedule** | No exponential decay configured | Add exponential decay: rate=0.9, every 2000–5000 steps (Papers 1-3 all use this) |
| **λ_ic** | Currently λ_ic=1.0 (zerilli_l2.yaml) or mixed in fd_refined | Set λ_ic = 10³ to 10⁴ for strong IC enforcement (Paper 2 recommendation) |
| **Batch size** | Nr=32000 (full domain), chunk_size=4096 | Papers recommend random resampling with 4096–8192 batch size per iteration — already partially aligned via resampling |
| **Gradient balancing formula** | Uses `mean(max_grad) / max_grad_i` (mean-normalized **inverse**) | Paper 3 Algorithm 1 uses `max|∇L_r| / mean|∇L_i|` — the reference term should specifically be the **PDE residual** gradient, not the average of all terms. Current implementation normalizes symmetrically across all terms rather than using the PDE residual gradient as the reference. See note below. |
| **Training iterations** | Adam: 2,000 per window (fd_refined), 10,000 (zerilli_l2) | Paper benchmarks use 100K–300K Adam iterations per window. Current settings may be too low for convergence. |
| **No weight decay** | Not explicitly set | Verify Adam is configured without weight_decay (Papers 1, 3 explicitly state no regularization for forward problems) |

### Gradient Balancing Formula — Detailed Note

**Current implementation** (from `GradientBalancing.on_epoch_end()`):
```python
mean_mg = sum(max_grads) / n           # average max-gradient across ALL terms
raw = [mean_mg / mg for mg in max_grads]  # normalize each term inversely
```

**Paper 3, Algorithm 1** specifies:
```
λ̂_i = max_θ|∇L_r(θ)| / mean(|∇L_i(θ)|)
```

The reference gradient should be **only from the PDE residual term (L_r)**, not the average across all terms. The denominator should be the **mean** of absolute parameter gradients for each non-PDE loss term (not the max). This difference means the current implementation balances all terms equally relative to each other, while the paper specifically keeps the PDE residual as the anchor and adjusts BC/IC weights relative to it. For the Zerilli equation with 7 loss terms (r, r_x, r_t, ic_disp, ic_vel, bc_left, bc_right), this distinction matters.

### Priority Improvements (Ordered)

1. **Increase network width** to 128–256 (trivial config change, large impact)
2. **Re-enable causal training** with ε annealing — essential for wave equations
3. **Add LR schedule** (exponential decay)
4. **Increase λ_ic** to 10³–10⁴
5. **Increase training iterations** to ~50K–100K Adam per window
6. **Consider RWF** implementation for additional adaptivity
7. **Fix gradient balancing formula** to use PDE residual as reference
