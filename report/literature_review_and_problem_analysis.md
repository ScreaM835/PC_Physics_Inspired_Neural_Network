# Deep Literature Review & Problem Analysis

**Project:** PINN for Zerilli Equation — Schwarzschild Black Hole QNMs  
**Target Paper:** Patel, Aykutalp & Laguna, arXiv:2401.01440 (2024)  
**Date:** Literature review conducted against 4 seminal PINN papers + 20+ gravitational physics references

---

## Executive Summary

After cross-referencing the full implementation against the target paper and the four most influential PINN methodology papers (Wang et al. 2021, 2022, 2023; Raissi et al. 2019), I have identified **12 problems** ranging from critical architectural mismatch to subtle training-pipeline bugs. The problems are categorised as:

- **Critical** (3): Will prevent or severely limit convergence
- **Major** (5): Significant accuracy/efficiency degradation
- **Minor** (4): Suboptimal choices that limit final accuracy

---

## Part I — Paper-vs-Implementation Discrepancy Analysis

### What the Target Paper Specifies (Patel et al. 2024)

| Aspect | Paper Specification |
|--------|-------------------|
| Architecture | Standard fully-connected MLP: [2 → 80 → 40 → 20 → 10 → 1] |
| Parameters | 4,521 trainable parameters |
| Activation | tanh throughout |
| Initialiser | Glorot uniform |
| Output transform | Φ_θ = A·tanh(Φ_θ) |
| Optimizer | Adam (10,000 iter) → L-BFGS (15,000 iter) |
| Loss weights λ | [100, 100, 100, 1, 100, 1, 1] for [L_r, L_rx, L_rt, L_ic, L_iv, L_bl, L_br] |
| Collocation pts | N_r = 32,000; N_i = 800; N_b = 400 (total 33,600) |
| Resampling | Every 100 iterations |
| Domain | x/M ∈ [-50, 150], t/M ∈ [0, 50] |
| IC | Gaussian: A=1, x₀=4M, σ=5M |
| IC velocity | Eq.23: Φ_t = 2(x-x₀)²/σ² · Φ (quadratic — nonstandard) |
| BCs | Sommerfeld: (∂_t - ∂_x)Φ=0 at x_min, (∂_t + ∂_x)Φ=0 at x_max |
| FD reference | dx=0.2M, dt=0.1M, N_x=1000, RK4, 2nd-order spatial |
| Gradient enhancement | Yes — residual returns [r, r_x, r_t] |
| Curriculum/causal | Not used |
| Gradient balancing | Not used |
| RFF | Not used |

### What the Implementation Actually Does

| Aspect | Implementation |
|--------|---------------|
| Architecture | **ModifiedMLP** with Wang skip connections + Trainable RFF (128-dim embedding → [64, 64, 64, 64] → 1) |
| Parameters | ~25,000+ (much larger than paper's 4,521) |
| Activation | tanh ✓ |
| Initialiser | Glorot uniform ✓ |
| Output transform | A·tanh(y) ✓ |
| Optimizer (zerilli_l2.yaml) | Adam (10,000) → L-BFGS (15,000) ✓ |
| Optimizer (fd_refined.yaml) | Adam (2,000) → L-BFGS (3,000) per curriculum window |
| Loss weights (zerilli_l2.yaml) | [1, 1, 1, 1, 1, 1, 1] + gradient balancing |
| Loss weights (fd_refined.yaml) | [100, 100, 100, 1, 100, 1, 1] (paper's weights, no gradient balancing) |
| Collocation pts | N_r=32,000; N_i=800; N_b=400 ✓ |
| Resampling | Every 100 iterations ✓ |
| Domain | [-50, 150] × [0, 50] ✓ |
| IC velocity | **"outgoing"** profile: Φ_t = 2(x-x₀)/σ² · Φ (linear, not the paper's quadratic) |
| Gradient enhanced | Yes ✓ |
| Curriculum | Yes — 5 windows [10, 20, 30, 40, 50] (not in paper) |
| Gradient balancing | Yes (zerilli_l2.yaml only, Wang et al. 2021) |
| RFF | Yes — trainable, σ=1.0, 64 features |
| Causal training | Disabled |

---

## Part II — Identified Problems

### CRITICAL PROBLEMS

#### Problem 1: Architecture Mismatch — Tapering vs. Uniform Width

**Paper architecture:** [80, 40, 20, 10] — a tapering (narrowing) network with 4,521 parameters.  
**Implementation:** [64, 64, 64, 64] — a uniform-width network (required by the Wang ModifiedMLP skip connections).

The Wang ModifiedMLP requires uniform width across all hidden layers because the encoder outputs U and V have a fixed width that must match every hidden layer for the element-wise gating H_k = (1-Z_k)⊙U + Z_k⊙V. This forces a uniform architecture.

**Why this matters:**
- The paper's tapering architecture is a deliberate design choice: wide early layers capture spatial features, narrow later layers compress to the single output. This is standard for small-parameter-count networks.
- The uniform [64, 64, 64, 64] architecture is **smaller than the paper's** in effective capacity (the paper's first layer alone has 80×2+80=240 parameters from the input, while yours has 128×64+64 ≈ 8,256 from RFF, but the subsequent layers are narrower).
- **More importantly:** the Wang et al. 2023 Expert's Guide recommends **128–256 neurons per layer** for wave equations. The current 64 is too narrow.

**Impact:** The 64-neuron width is the single most limiting factor for representational capacity. All PINN benchmark problems of comparable complexity (Allen-Cahn, Kuramoto-Sivashinsky, Navier-Stokes) use 128–256 neurons per layer.

**Fix:** Change `hidden_layers: [64, 64, 64, 64]` → `[128, 128, 128, 128]` or `[256, 256, 256, 256]` in both configs.

---

#### Problem 2: Causal Training Disabled

The implementation has causal training infrastructure (`causal.enabled: false` in the config) but it is **disabled** and never implemented in the DeepXDE migration.

**Why this matters:**
- The Zerilli equation is a **hyperbolic PDE** (wave equation with potential). Information propagates causally — the solution at time t depends only on data at times < t.
- Wang et al. 2022 proved that without causal weighting, PINNs are "implicitly biased towards minimizing PDE residuals at later times first" — exactly the failure mode for wave propagation.
- The paper's "future directions" section explicitly lists "incorporating causality" as a key improvement (Section 7, bullet 4).
- The curriculum learning in `zerilli_l2_fd_refined.yaml` partially addresses this (by restricting training to progressively larger time windows), but **causal violations can still occur within each window**.

**Impact:** Without causal training, the PINN will try to learn the late-time QNM ringdown before correctly resolving the initial pulse propagation. This produces phase errors — exactly the "phase problem" the paper discusses.

**Fix:** Implement causal loss weighting within the DeepXDE training loop. Use ε-annealing schedule [0.01, 0.1, 1, 10, 100] with `stop_gradient` on weights. Even within curriculum windows, causal weighting is essential.

---

#### Problem 3: Gradient Balancing Formula Deviates from Paper

The current `GradientBalancing` callback computes:

```python
mean_mg = sum(max_grads) / n           # average of max-grads across ALL terms
raw = [mean_mg / mg for mg in max_grads]  # weight = average / term_i
```

Wang et al. 2021 Algorithm 1 specifies:

```
λ̂_i = max_θ|∇_θ L_r| / mean_θ|∇_θ L_i|    for each i ∈ {bc, ic, ...}
```

**Three differences:**
1. **Reference term:** The paper uses the PDE residual gradient (max|∇L_r|) as the fixed numerator for all weight computations. The implementation uses the average across all terms — this means the PDE residual is also reweighted, which was not intended.
2. **Numerator statistic:** The paper uses max|grad| of the PDE residual specifically. The implementation averages the max|grad| across all 7 terms.
3. **Denominator statistic:** The paper uses mean|grad| for each non-PDE term. The implementation uses max|grad| for each term.

**Impact:** The current formula balances all terms equally relative to each other (symmetric normalisation), whereas the paper anchors everything to the PDE residual. This may over-weight or under-weight the PDE residual relative to IC/BC terms. For the Zerilli equation with its known gradient stiffness, this distinction matters.

**Fix:** Modify `GradientBalancing.on_epoch_end()` to compute per-term mean|grad| (not max) and use the PDE residual's max|grad| as the fixed numerator:
```python
max_grad_pde = max_grads[0]  # L_r is the first loss term
raw = [max_grad_pde / mean_grads[i] for i in range(1, n)]  # only for non-PDE terms
# PDE residual weight stays at 1.0
```

---

### MAJOR PROBLEMS

#### Problem 4: Initial Velocity Profile Diverges from Paper (But Deliberately)

The configs use `velocity_profile: outgoing`, which computes:

$$\Phi_t = \frac{2(x-x_0)}{\sigma^2} \Phi \quad \text{(linear factor)}$$

The paper's Eq. 23 specifies:

$$\Phi_t = \frac{2(x-x_0)^2}{\sigma^2} \Phi \quad \text{(quadratic factor)}$$

The existing literature review (§6.2) correctly identifies the paper's formula as **nonstandard** and possibly erroneous. The "outgoing" profile is more physically motivated (it produces a purely right-moving pulse). However:

**Impact:** If the goal is to **reproduce the paper's results**, neither profile matches the paper. If the goal is better physics, the "outgoing" profile is superior. This is a deliberate design choice, but should be documented clearly. The QNM frequencies are independent of initial data, but the **transient solution shape** differs, meaning error metrics (RMSD, MAD, RL2) computed against a FD solution with a different velocity profile will be inflated.

**Critical point:** The FD solver must use the same velocity profile as the PINN. Check that `fd_solver.py` respects the `velocity_profile` config parameter.

---

#### Problem 5: No Learning Rate Decay

Neither config uses learning rate scheduling. Adam runs at constant lr=0.001 throughout.

**What the literature recommends:**
- Wang et al. 2021: exponential decay, rate=0.9, every 1000 steps
- Wang et al. 2022: exponential decay, rate=0.9, every 5000 steps
- Wang et al. 2023: exponential decay, rate=0.9, every 2000–5000 steps

**Impact:** Without LR decay, the optimizer oscillates around the minimum during later training iterations instead of converging. This is especially harmful during long Adam phases (10,000 iterations in `zerilli_l2.yaml`). The oscillation manifests as noisy loss curves and prevents the network from reaching the accuracy floor.

**Fix:** Add exponential LR scheduling to DeepXDE's Adam via a callback or by using `torch.optim.lr_scheduler.ExponentialLR`. Decay factor 0.9 every 2000 steps is a good starting point.

---

#### Problem 6: Insufficient Training Iterations (fd_refined config)

The `zerilli_l2_fd_refined.yaml` uses only **2,000 Adam + 3,000 L-BFGS per window** (5,000 total per window, 25,000 total across 5 windows).

**Benchmarks from the literature:**
- Patel et al. 2024: 10,000 Adam + 15,000 L-BFGS = 25,000 total (no windows)
- Wang et al. 2022: 100,000–300,000 Adam per time window
- Wang et al. 2023: 150,000–300,000 Adam per time window

The current training budget is **5–60× less** than what the literature uses for comparable PDEs.

**Impact:** The network likely has not converged, especially in later curriculum windows where the time domain is larger and the solution is more complex. The L-BFGS phase (3,000 iterations) is also too short — the paper uses 15,000.

**Fix:** For a faithful reproduction, use at minimum the paper's 10,000 Adam + 15,000 L-BFGS. For superior results with the ModifiedMLP architecture, consider 20,000–50,000 Adam iterations per window.

---

#### Problem 7: Loss Weight Configuration Inconsistency

The two configs use dramatically different loss weight strategies:

| Config | Weights [L_r, L_rx, L_rt, L_ic, L_iv, L_bl, L_br] | Gradient Balancing |
|--------|------------------------------------------------------|-------------------|
| `zerilli_l2.yaml` | [1, 1, 1, 1, 1, 1, 1] | Enabled |
| `zerilli_l2_fd_refined.yaml` | [100, 100, 100, 1, 100, 1, 1] | **Not configured** |

The paper uses [100, 100, 100, 1, 100, 1, 1] **without** gradient balancing.

**Problem with fd_refined config:** It uses the paper's manual weights but lacks gradient balancing. The paper explicitly explains why λ_iv = 100: "The weight λ_iv = 100 is because of the phase problem." But the paper's weights were calibrated for the paper's architecture (4,521 params, standard MLP) — they may not be optimal for the ModifiedMLP with RFF.

**Problem with zerilli_l2 config:** Equal weights + gradient balancing is more principled but the gradient balancing formula has the bug described in Problem 3.

**Impact:** Neither config has a fully correct loss weighting strategy. The fd_refined config is particularly suspect because it combines the paper's weights (tuned for a different architecture) with curriculum learning (not in the paper) and a different velocity profile.

---

#### Problem 8: RFF Feature Count Too Low

The implementation uses `num_features: 64` (output dimension = 128). Wang et al. 2023 benchmarks use 128–256 Fourier features.

**Why this matters for the Zerilli equation:**
- The QNM oscillation frequency is ω ≈ 0.374/M with damping time τ ≈ 11.2M
- The initial Gaussian pulse has spatial frequency content determined by σ=5M
- The RFF embedding should capture both the spatial pulse structure and the temporal oscillation
- With only 64 features, the frequency resolution may be insufficient

**Fix:** Increase `num_features` to 128 or 256.

---

### MINOR PROBLEMS

#### Problem 9: ModifiedMLP Skip Connections May Mask Failure

In the Wang ModifiedMLP, if all Z_k ≈ 0 or ≈ 1, the hidden layers degenerate: the output becomes approximately U or V everywhere, bypassing the hidden layer transformations entirely. This is a failure mode that's hard to detect.

**Diagnostic:** Add logging/monitoring of the mean Z_k values during training. If they collapse to 0 or 1, the skip connections are not functioning as intended.

---

#### Problem 10: No Input Normalisation / Non-dimensionalisation

The input domain is x ∈ [-50, 150] and t ∈ [0, 50]. This is not normalised — x spans 200 units while t spans 50 units. Wang et al. 2023 explicitly recommend non-dimensionalising the PDE so that inputs are O(1).

**Impact:** The 4:1 aspect ratio between spatial and temporal domains creates an asymmetry in the RFF embedding (the B matrix must learn different scales for x and t) and can contribute to the gradient pathologies described in Wang et al. 2021. The gradient stiffness scales with the characteristic frequency, which is amplified by the large domain extent.

**Fix:** Add input normalisation: x_norm = (x - x_mid) / (x_max - x_min), t_norm = t / t_max, so both inputs are in [0, 1] or [-1, 1]. Apply this as the RFF input or as a DeepXDE input transform.

---

#### Problem 11: Missing Random Weight Factorisation (RWF)

Wang et al. 2023 (Section 3.2) introduce Random Weight Factorisation: each weight matrix W is factored as W = diag(exp(s)) · V where s ~ N(μ, σ_rwf). This provides per-neuron adaptive learning rates without extra optimiser complexity.

**Impact:** Moderate. The ModifiedMLP with RFF already addresses spectral bias, but RWF provides additional adaptivity that can improve convergence speed by ~2×.

---

#### Problem 12: QNM Extraction Limited to Two Methods

The implementation uses FFT + peak fitting (Method 1) and nonlinear damped cosine fit (Method 2). Both are from the target paper. Missing: the **Prony method** (matrix pencil), which is the standard in the gravitational physics community and can extract multiple overtones simultaneously.

**Impact:** Low for fundamental mode extraction, but the Prony method would enable overtone extraction and provide a more robust frequency estimate, especially when the PINN solution has residual phase errors.

---

## Part III — Physics Implementation Review

### Potential Computation: Correct ✓

The Zerilli potential formula in `potentials.py` matches the paper's Eq. 19:

$$V(r) = f(r) \frac{2n^2(n+1)r^3 + 6n^2 M r^2 + 18n M^2 r + 18 M^3}{r^3(nr + 3M)^2}$$

with n = (l-1)(l+2)/2 and f(r) = 1-2M/r. The Lambert-W inversion r(x*) is mathematically correct and the Halley iteration converges to machine precision.

### Boundary Conditions: Correct ✓

Sommerfeld BCs are correctly implemented:
- Left (x → -∞): (∂_t - ∂_x)Φ = 0 (ingoing)
- Right (x → +∞): (∂_t + ∂_x)Φ = 0 (outgoing)

### Gradient-Enhanced Residual: Correct ✓

The PDE residual returns [r, r_x, r_t] where r = Φ_tt - Φ_xx + V·Φ, and r_x, r_t are computed via autograd (not finite differences). Because V(x*) is computed via differentiable Lambert-W, the gradients dV/dx are exact — no manual correction needed.

### FD Solver: Correct ✓

The method-of-lines RK4 with 2nd-order spatial derivatives matches the paper's description. Boundary conditions use 2nd-order one-sided formulas.

### Output Transform: Correct ✓

A·tanh(y) correctly bounds the output in [-A, A], matching Eq. 5 in the paper.

---

## Part IV — Prioritised Recommendations

### Tier 1: Essential for Convergence (Do First)

1. **Increase hidden layer width** to [128, 128, 128, 128]. Config-only change.
2. **Re-enable causal training** with ε-annealing within curriculum windows.
3. **Fix gradient balancing formula** to use PDE residual as the reference anchor.
4. **Add learning rate decay** (exponential, rate=0.9, every 2000 steps).

### Tier 2: Significant Accuracy Improvement (Do Second)

5. **Increase training iterations** to ≥10K Adam + 15K L-BFGS per window (or 25K total without curriculum).
6. **Increase RFF features** from 64 to 128.
7. **Add input normalisation** to map x, t to O(1) ranges.
8. **Sort out loss weight strategy**: either use gradient balancing (fixed formula) with equal initial weights, or use the paper's manual weights without gradient balancing.

### Tier 3: Refinements (Do Third)

9. **Implement RWF** (Random Weight Factorisation) for per-neuron adaptive LR.
10. **Add Prony method** to QNM extraction for multi-overtone analysis.
11. **Monitor Z_k statistics** in ModifiedMLP for degenerate skip connections.
12. **Decide on velocity profile** and document the rationale.

---

## Part V — Key Papers Referenced

| # | Paper | Relevance |
|---|-------|-----------|
| 1 | Patel, Aykutalp & Laguna (2024), arXiv:2401.01440 | Target paper being reproduced |
| 2 | Wang, Teng & Perdikaris (2021), SIAM Rev. | Gradient balancing algorithm, Modified MLP architecture |
| 3 | Wang, Sankaran & Perdikaris (2022), arXiv:2203.07404 | Causal training for hyperbolic PDEs |
| 4 | Wang et al. (2023), arXiv:2308.08468 | Expert's guide — RFF, RWF, training best practices |
| 5 | Raissi, Perdikaris & Karniadakis (2019), JCP | Original PINN formulation |
| 6 | Yu, Lu, Meng & Karniadakis (2022), CMAME | Gradient-enhanced PINNs |
| 7 | Luna, Maselli & Pani (2024), arXiv:2404.11583 | Frequency-domain PINN for QNMs |
| 8 | McClenny & Braga-Neto (2023), JCP | Self-adaptive per-point weights |
| 9 | Bulut (2025), arXiv:2512.23396 | Time-marching PINNs for wave equations |
| 10 | Wu, Zhu, Tan et al. (2023), CMAME | Residual-based adaptive sampling |

---

*This report was generated by systematic cross-referencing of the full implementation codebase against the target paper and four key PINN methodology papers.*
