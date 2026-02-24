"""
PINN for the 1+1D Zerilli/Regge-Wheeler equation using DeepXDE.

PDE:  phi_tt - phi_xx + V(x*) phi = 0

Migrated from custom PyTorch to the DeepXDE framework.
Key improvements over the previous implementation:
  - Proper L-BFGS via PyTorch LBFGS with correct iteration management
    (DeepXDE handles the optimizer loop, no more max_iter=1 hack)
  - Exact autograd through V(x*) via pure-torch Lambert-W potential
    (no manual dV/dx correction needed for gradient-enhanced residuals)
  - Framework-standard IC/BC handling and collocation-point resampling
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import deepxde as dde
import numpy as np
import torch

from .potentials import V_of_x_torch
from .initial_data import gaussian_phi, gaussian_phi_t
from .modified_mlp import PlainRFFNet


# ---------------------------------------------------------------------------
# Causal training (Wang, Perdikaris & Sifakis 2022, arXiv:2203.07404)
# ---------------------------------------------------------------------------

class CausalWeighter:
    """Apply causal weighting to PDE residuals (Wang et al. 2022).

    Divides the time domain into slices and weights residuals so that
    later times are penalised unless earlier times are already well-resolved.

        w_k = exp(-epsilon * sum_{j<k} L̂_j)

    where L̂_j = L_j / max_k(L_k) is the *rescaled* mean squared PDE
    residual in time slice j.  Normalising by the max slice loss ensures
    that L̂ ∈ [0, 1] regardless of absolute loss magnitude, so epsilon
    controls causal strength as Wang et al. intended (their derivation
    assumes O(1) losses).  Without rescaling, gradient-enhanced training
    drives per-slice losses to O(10^-4), making epsilon=10 a no-op.

    Light-cone masking (Patel et al. 2024): when computing L_j, only
    points inside the past light cone of the initial data are included.
    This prevents causally-empty regions (where psi=0 trivially) from
    diluting the per-slice loss estimate.  c=1 in tortoise coordinates,
    so the light cone is |x* - x0| <= t + 3*sigma.

    Residuals are multiplied by sqrt(w_k) so that MSE(sqrt(w)*r) = w*r**2,
    giving the causally weighted loss without modifying DeepXDE's loss
    pipeline.  The weights are detached so gradients flow only through the
    residuals, not through the weights themselves.
    """

    def __init__(
        self,
        tmin: float,
        tmax: float,
        epsilon: float = 10.0,
        num_slices: int = 20,
        x0: float = 0.0,
        sigma: float = 5.0,
    ):
        self.tmin = tmin
        self.tmax = tmax
        self.num_slices = num_slices
        self.epsilon = epsilon
        self.active = True   # disabled before L-BFGS (Adam-only mechanism)
        self.w_min = 1.0  # min causal weight (approaches 1 as training converges)

        # Light-cone parameters: signal from Gaussian at x0 with width sigma
        # reaches x* when |x* - x0| <= t + 3*sigma  (c=1 in tortoise coords)
        self.x0 = x0
        self.sigma = sigma

    def apply(self, x: torch.Tensor, *residuals: torch.Tensor) -> list:
        """Weight residuals by causal factor.

        Parameters
        ----------
        x : Tensor (N, 2)
            Input coordinates [x*, t].
        *residuals : Tensor (N, 1) each
            PDE residual components [r, r_x, r_t].

        Returns
        -------
        list of Tensor
            Weighted residuals, same shapes as inputs.
        """
        # Passthrough when disabled (L-BFGS phase)
        if not self.active:
            self.w_min = 1.0
            return list(residuals)

        xs = x[:, 0:1].detach()  # (N, 1)
        t = x[:, 1:2].detach()   # (N, 1)

        # Assign each point to a time slice
        dt = (self.tmax - self.tmin) / self.num_slices
        slice_idx = ((t - self.tmin) / dt).long().clamp(
            0, self.num_slices - 1
        ).squeeze(-1)  # (N,)

        # Light-cone mask: point is causally active if |x* - x0| <= t + 3*sigma
        in_light_cone = (torch.abs(xs - self.x0) <= t + 3.0 * self.sigma).squeeze(-1)  # (N,)

        # Per-slice mean squared PDE residual (primary residual only)
        # Only include causally-active points to avoid dilution
        r_det = residuals[0].detach()  # (N, 1)
        per_slice_loss = torch.zeros(
            self.num_slices, device=r_det.device, dtype=r_det.dtype
        )
        for k in range(self.num_slices):
            mask = (slice_idx == k) & in_light_cone
            if mask.any():
                vals = r_det[mask] ** 2
                vals = torch.nan_to_num(vals, nan=0.0)
                per_slice_loss[k] = vals.mean()

        # Rescale per-slice losses to O(1) so that epsilon controls the
        # causal strength independent of absolute loss magnitude.
        # Without this, losses of O(10^-4) make epsilon=10 a no-op.
        max_loss = per_slice_loss.max().clamp(min=1e-30)
        per_slice_loss = per_slice_loss / max_loss

        # Causal weights: w_k = exp(-epsilon * sum_{j<k} L_j)
        # w_0 = 1 (no prior losses), w_1 = exp(-eps*L_0), ...
        cumulative = torch.cumsum(per_slice_loss, dim=0)
        shifted = torch.cat(
            [torch.zeros(1, device=r_det.device, dtype=r_det.dtype),
             cumulative[:-1]]
        )
        w = torch.exp(-self.epsilon * shifted)

        # Track minimum weight for monitoring convergence
        self.w_min = w.min().item()

        # Map slice weights back to individual points
        # (weights apply to ALL points, including outside light cone)
        sqrt_w = torch.sqrt(w[slice_idx]).unsqueeze(1)  # (N, 1), detached

        return [res * sqrt_w for res in residuals]


class CausalTrainingMonitor(dde.callbacks.Callback):
    """Log the minimum causal weight for monitoring convergence."""

    def __init__(self, causal_weighter: CausalWeighter, period: int = 100):
        super().__init__()
        self.cw = causal_weighter
        self.period = period

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.period != 0:
            return

        print(f"  [Causal] step {step}: epsilon={self.cw.epsilon}, w_min={self.cw.w_min:.6f}")


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------

def _make_pde_func(cfg: Dict, tmax_override: Optional[float] = None):
    """Build the PDE residual function, closed over physics parameters.

    If causal training is enabled in the config, residuals are weighted by
    time-slice-dependent causal factors (Wang et al. 2022).
    """
    M = float(cfg["physics"]["M"])
    l = int(cfg["physics"]["l"])
    potential = cfg["physics"]["potential"]

    # --- optional causal weighting ---
    causal_cfg = cfg["pinn"].get("causal", {})
    causal_weighter = None
    if causal_cfg.get("enabled", False):
        tmin = float(cfg["domain"]["tmin"])
        tmax = tmax_override if tmax_override is not None else float(cfg["domain"]["tmax"])

        # Light-cone parameters from initial data
        x0_ic = float(cfg["initial_data"]["x0"])
        sigma_ic = float(cfg["initial_data"]["sigma"])

        causal_weighter = CausalWeighter(
            tmin=tmin,
            tmax=tmax,
            epsilon=float(causal_cfg.get("epsilon", 10.0)),
            num_slices=int(causal_cfg.get("n_slices", 20)),
            x0=x0_ic,
            sigma=sigma_ic,
        )
        print(f"[PINN] Causal training enabled: epsilon={causal_weighter.epsilon}, "
              f"n_slices={causal_weighter.num_slices}, t=[{tmin},{tmax}], "
              f"light-cone: x0={x0_ic}, sigma={sigma_ic}")

    def pde(x, y):
        """
        PDE residual for  phi_tt - phi_xx + V(x*) phi = 0.

        Returns [r, r_x, r_t] for gradient-enhanced training.

        Parameters
        ----------
        x : Tensor (N, 2)  -- columns are [x*, t]
        y : Tensor (N, 1)  -- network output phi
        """
        # Second derivatives via DeepXDE's cached Hessian
        phi_xx = dde.grad.hessian(y, x, i=0, j=0)
        phi_tt = dde.grad.hessian(y, x, i=1, j=1)

        # Potential -- fully inside the autograd graph (pure-torch Lambert-W)
        V = V_of_x_torch(x[:, 0:1], M, l, potential)

        # Standard PDE residual
        r = phi_tt - phi_xx + V * y

        # Gradient-enhanced residuals via autograd.
        # Because V(x) is differentiable, dr/dx automatically includes dV/dx * phi.
        dr = torch.autograd.grad(
            r, x,
            grad_outputs=torch.ones_like(r),
            create_graph=True,
            retain_graph=True,
        )[0]
        r_x = dr[:, 0:1]
        r_t = dr[:, 1:2]

        # Apply causal weighting if enabled
        if causal_weighter is not None:
            r, r_x, r_t = causal_weighter.apply(x, r, r_x, r_t)

        return [r, r_x, r_t]

    # Attach the causal weighter so callers can access it (e.g. for monitoring)
    pde._causal_weighter = causal_weighter
    return pde


# ---------------------------------------------------------------------------
# Initial / boundary conditions
# ---------------------------------------------------------------------------

def _make_ic_bcs(cfg: Dict, geomtime):
    """Create the four IC/BC objects for the Zerilli equation."""
    A0 = float(cfg["initial_data"]["A"])
    x0_ic = float(cfg["initial_data"]["x0"])
    sigma = float(cfg["initial_data"]["sigma"])
    profile = cfg["initial_data"]["velocity_profile"]
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])

    # ---- IC: phi(x, 0) = Gaussian ----
    def phi0_func(x):
        """x is numpy (N, 2). Return (N, 1)."""
        return gaussian_phi(x[:, 0:1], A=A0, x0=x0_ic, sigma=sigma)

    ic_disp = dde.icbc.IC(
        geomtime, phi0_func, lambda _, on_initial: on_initial
    )

    # ---- IC: d_phi/dt(x, 0) = v0(x) ----
    def vel_func(inputs, outputs, X):
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        v0 = gaussian_phi_t(
            X[:, 0:1], A=A0, x0=x0_ic, sigma=sigma, profile=profile
        )
        v0_t = torch.as_tensor(v0, dtype=outputs.dtype, device=outputs.device)
        return phi_t - v0_t

    # Selects points at t = tmin from train_x_all (includes num_initial points)
    ic_vel = dde.icbc.OperatorBC(
        geomtime, vel_func,
        lambda x, on_boundary: np.isclose(x[1], tmin),
    )

    # ---- BC left: (d_t - d_x) phi = 0  at x = xmin  (Sommerfeld, ingoing) ----
    def bc_left_func(inputs, outputs, X):
        phi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        return phi_t - phi_x

    bc_left = dde.icbc.OperatorBC(
        geomtime, bc_left_func,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], xmin),
    )

    # ---- BC right: (d_t + d_x) phi = 0  at x = xmax  (Sommerfeld, outgoing) ----
    def bc_right_func(inputs, outputs, X):
        phi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        return phi_t + phi_x

    bc_right = dde.icbc.OperatorBC(
        geomtime, bc_right_func,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], xmax),
    )

    return [ic_disp, ic_vel, bc_left, bc_right]


# ---------------------------------------------------------------------------
# Gradient balancing (Wang, Teng & Perdikaris 2021, Algorithm 1)
# ---------------------------------------------------------------------------

class GradientBalancing(dde.callbacks.Callback):
    """Dynamically adjust loss weights via Wang, Teng & Perdikaris 2021, Alg. 1.

    At every *period* Adam steps the callback:
      1. computes each unweighted loss L_i,
      2. backpropagates each independently,
      3. sets  λ̂_i = max_θ|∇L_r| / mean_θ|∇L_i|  (Alg. 1),
         where L_r (index 0) is the primary PDE residual,
      4. applies an exponential moving average with decay *alpha*.

    The PDE residual (index 0) always keeps weight 1.0 — it is the
    reference anchor.  All other terms (gradient-enhanced PDE residuals,
    IC, BC) are reweighted relative to it.

    The weights are frozen during L-BFGS (callback only fires in SGD loops).
    """

    def __init__(self, period: int = 100, alpha: float = 0.9):
        super().__init__()
        self.period = period
        self.alpha = alpha
        self._ema_weights: Optional[List[float]] = None

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step == 0 or step % self.period != 0:
            return

        net = self.model.net
        n = len(self.model.loss_weights)

        # --- unweighted forward pass ---
        self.model.loss_weights = [1.0] * n

        _, losses = self.model.outputs_losses_train(
            self.model.train_state.X_train,
            self.model.train_state.y_train,
            self.model.train_state.train_aux_vars,
        )

        # --- per-term gradient statistics ---
        # max|grad|  : needed for the reference term (L_r, index 0)
        # mean|grad| : needed for the denominator of all other terms
        max_grads: List[float] = []
        mean_grads: List[float] = []
        for i, loss_i in enumerate(losses):
            net.zero_grad()
            loss_i.backward(retain_graph=(i < n - 1))
            mg = 0.0
            total_abs = 0.0
            count = 0
            for p in net.parameters():
                if p.grad is not None:
                    mg = max(mg, p.grad.abs().max().item())
                    total_abs += p.grad.abs().sum().item()
                    count += p.grad.numel()
            max_grads.append(mg + 1e-16)
            mean_grads.append(total_abs / max(count, 1) + 1e-16)

        net.zero_grad()

        # --- compute balanced weights (Algorithm 1) ---
        # Reference: max|∇_θ L_r| where L_r is the primary PDE residual (index 0).
        # Weight_i = max|∇L_r| / mean|∇L_i|  for i >= 1.
        # Weight_0 = 1.0 (PDE residual is the anchor).
        max_grad_r = max_grads[0]
        raw = [1.0]  # L_r weight
        for i in range(1, n):
            raw.append(max_grad_r / mean_grads[i])

        if self._ema_weights is None:
            self._ema_weights = raw
        else:
            # Standard EMA: alpha controls retention of history.
            # alpha=0.9 -> 90% old + 10% new (slow, smooth adaptation).
            self._ema_weights = [
                self.alpha * ew + (1.0 - self.alpha) * rw
                for ew, rw in zip(self._ema_weights, raw)
            ]

        self.model.loss_weights = list(self._ema_weights)
        # restore not needed — we just overwrote with the new balanced weights

        wstr = ", ".join(f"{w:.2f}" for w in self._ema_weights)
        print(f"  [GradBal] step {step}: weights=[{wstr}]")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(cfg: Dict, tmax_override: Optional[float] = None, net_override: Optional[torch.nn.Module] = None) -> Tuple[dde.Model, dde.data.TimePDE]:
    """Construct the DeepXDE Model from the experiment config.

    Uses PlainRFFNet (plain MLP + Trainable RFF input embedding)
    instead of a standard FNN, to overcome spectral bias.
    """
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = tmax_override if tmax_override is not None else float(cfg["domain"]["tmax"])

    geom = dde.geometry.Interval(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(tmin, tmax)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    tmax_actual = tmax_override if tmax_override is not None else float(cfg["domain"]["tmax"])
    pde_func = _make_pde_func(cfg, tmax_override=tmax_actual)
    ic_bcs = _make_ic_bcs(cfg, geomtime)

    # Scale the number of points by the time domain fraction
    t_fraction = (tmax - tmin) / (float(cfg["domain"]["tmax"]) - tmin)
    Nr = int(int(cfg["pinn"]["Nr"]) * t_fraction)
    Ni = int(cfg["pinn"]["Ni"])
    Nb = int(int(cfg["pinn"]["Nb"]) * t_fraction)

    data = dde.data.TimePDE(
        geomtime,
        pde_func,
        ic_bcs,
        num_domain=Nr,
        num_boundary=Nb,
        num_initial=Ni,
        train_distribution="uniform",
    )

    if net_override is not None:
        net = net_override
        # Update input normalization for the new time window (curriculum)
        x_lo, x_hi = xmin, xmax
        t_lo, t_hi = tmin, tmax

        def _input_normalize_update(x, _xlo=x_lo, _xhi=x_hi, _tlo=t_lo, _thi=t_hi):
            x_n = 2.0 * (x[:, 0:1] - _xlo) / (_xhi - _xlo) - 1.0
            t_n = 2.0 * (x[:, 1:2] - _tlo) / (_thi - _tlo) - 1.0
            return torch.cat([x_n, t_n], dim=1)

        net.apply_feature_transform(_input_normalize_update)
    else:
        # --- Modified MLP with Trainable RFF ---
        hidden = [int(w) for w in cfg["pinn"]["hidden_layers"]]
        rff_cfg = cfg["pinn"].get("rff", {})
        num_rff = int(rff_cfg.get("num_features", 64))
        rff_sigma = float(rff_cfg.get("sigma", 1.0))
        rff_trainable = bool(rff_cfg.get("trainable", True))
        activation = cfg["pinn"].get("activation", "tanh")

        net = PlainRFFNet(
            hidden_layers=hidden,
            num_rff=num_rff,
            rff_sigma=rff_sigma,
            rff_trainable=rff_trainable,
            activation=activation,
        )

        print(f"[PINN] PlainRFFNet: hidden={hidden}, "
              f"RFF(num={num_rff}, sigma={rff_sigma}, trainable={rff_trainable})")
        print(f"[PINN] Trainable parameters: {net.num_trainable_parameters()}")

        # Input normalization: map physical domain → [-1, 1]²
        # Ensures the RFF embedding sees O(1) inputs so that rff.sigma
        # controls frequency resolution as intended (Wang et al. 2023).
        # Autograd traces through this, so PDE derivatives remain in
        # physical coordinates — no manual chain-rule correction needed.
        x_lo, x_hi = xmin, xmax
        t_lo, t_hi = tmin, tmax   # uses tmax_override for curriculum windows

        def _input_normalize(x, _xlo=x_lo, _xhi=x_hi, _tlo=t_lo, _thi=t_hi):
            x_n = 2.0 * (x[:, 0:1] - _xlo) / (_xhi - _xlo) - 1.0
            t_n = 2.0 * (x[:, 1:2] - _tlo) / (_thi - _tlo) - 1.0
            return torch.cat([x_n, t_n], dim=1)

        net.apply_feature_transform(_input_normalize)
        print(f"[PINN] Input normalization: x*∈[{x_lo},{x_hi}]→[-1,1], "
              f"t∈[{t_lo},{t_hi}]→[-1,1]")

        # Output transform: A * tanh(y)
        # Bounding the output enforces the physical constraint of energy conservation
        # and prevents the network from adapting a blowing-up solution (Patel et al. 2024).
        A_bound = float(cfg["initial_data"]["A"])
        net.apply_output_transform(lambda x, y: A_bound * torch.tanh(y))

    model = dde.Model(data, net)

    # Expose causal weighter on model for callback/monitoring access
    model._causal_weighter = getattr(pde_func, '_causal_weighter', None)

    return model, data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_pinn_curriculum(
    cfg: Dict,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> Tuple[dde.Model, Dict[str, List[float]]]:
    """
    Train the PINN using Curriculum Learning (Expanding Time Windows).
    This is mathematically equivalent to Time-Marching but avoids error accumulation
    at window boundaries and bypasses PyTorch derivative extraction bugs.
    """
    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    loss_weights = [float(w) for w in cfg["pinn"]["lambda"]]
    
    curriculum_cfg = cfg["pinn"]["curriculum"]
    windows = curriculum_cfg["windows"]  # e.g., [10.0, 20.0, 30.0, 40.0, 50.0]
    
    net = None
    history_all = None
    
    for i, tmax in enumerate(windows):
        print(f"\n{'='*50}")
        print(f"[PINN] Curriculum Window {i+1}/{len(windows)}: t in [0, {tmax}]")
        print(f"{'='*50}\n")
        
        # Build model for this window
        model, data = build_model(cfg, tmax_override=tmax, net_override=net)
        net = model.net  # Keep the network for the next window
        
        # Set up checkpointing for this window
        window_ckpt_dir = None
        model_save_path = None
        if checkpoint_dir:
            window_ckpt_dir = os.path.join(checkpoint_dir, f"window_{i+1}")
            os.makedirs(window_ckpt_dir, exist_ok=True)
            model_save_path = os.path.join(window_ckpt_dir, "model")
            
        # Check if this window is already fully trained
        # DeepXDE appends the step number: model-final-{step}.pt
        if resume and window_ckpt_dir:
            final_ckpt = _find_final_checkpoint(window_ckpt_dir)
            if final_ckpt is not None:
                print(f"[CKPT] Window {i+1} already completed. Restoring and skipping.")
                model.compile("adam", lr=1e-3)  # dummy compile
                model.train(iterations=0, display_every=1)  # init train state
                # Only restore model weights (skip optimizer state to avoid mismatch)
                checkpoint = torch.load(final_ckpt, weights_only=True)
                model.net.load_state_dict(checkpoint["model_state_dict"])
                print(f"  Restored weights from {final_ckpt}")
                continue
            
        # ---- Adam phase ----
        adam_cfg = cfg["pinn"]["adam"]
        adam_iters = int(adam_cfg["iters"])
        lr = float(adam_cfg["lr"])
        resample_period = int(adam_cfg["resample_period"])

        callbacks_adam: List = []
        callbacks_adam.append(
            dde.callbacks.PDEPointResampler(period=resample_period)
        )

        # Gradient balancing (Wang et al. 2021)
        grad_bal_cfg = cfg["pinn"].get("gradient_balancing", {})
        if grad_bal_cfg.get("enabled", False):
            gb_period = int(grad_bal_cfg.get("period", 100))
            gb_alpha = float(grad_bal_cfg.get("alpha", 0.9))
            callbacks_adam.append(
                GradientBalancing(period=gb_period, alpha=gb_alpha)
            )

        # Causal training monitor
        if model._causal_weighter is not None:
            callbacks_adam.append(
                CausalTrainingMonitor(model._causal_weighter, period=100)
            )

        model_restore_path = None
        if resume and window_ckpt_dir:
            ckpt = _find_latest_checkpoint(window_ckpt_dir)
            if ckpt is not None:
                model_restore_path = ckpt
                print(f"[CKPT] Restoring from {ckpt}")

        if model_save_path:
            callbacks_adam.append(
                dde.callbacks.ModelCheckpoint(
                    model_save_path,
                    save_better_only=False,
                    period=checkpoint_every,
                )
            )

        model.compile("adam", lr=lr, loss_weights=loss_weights)

        print(f"[PINN] Adam: {adam_iters} iters, lr={lr}, resample every {resample_period}")
        losshistory_adam, _ = model.train(
            iterations=adam_iters,
            callbacks=callbacks_adam,
            display_every=100,
            model_save_path=model_save_path,
            model_restore_path=model_restore_path,
        )

        if model_save_path:
            model.save(model_save_path + "-adam_done")
            import json
            weights_file = os.path.join(window_ckpt_dir, "loss_weights_adam.json")
            with open(weights_file, "w") as f:
                json.dump(list(model.loss_weights), f)

        # ---- L-BFGS phase ----
        lbfgs_cfg = cfg["pinn"]["lbfgs"]
        lbfgs_iters = int(lbfgs_cfg["iters"])

        lbfgs_loss_weights = list(model.loss_weights)

        # Disable causal weighting for L-BFGS (Adam-only mechanism)
        if model._causal_weighter is not None:
            model._causal_weighter.active = False
            print("[PINN] Causal weighting disabled for L-BFGS phase")

        dde.optimizers.set_LBFGS_options(
            maxcor=100,
            maxiter=lbfgs_iters,
            ftol=0,
            gtol=1e-8,
            maxls=50,
        )
        
        lbfgs_resample_period = int(lbfgs_cfg.get("resample_period", 0))
        if lbfgs_resample_period > 0:
            step_size = min(checkpoint_every, lbfgs_resample_period, lbfgs_iters)
        else:
            step_size = min(checkpoint_every, lbfgs_iters)

        from deepxde.optimizers.config import LBFGS_options as _lbfgs_opts
        _lbfgs_opts["iter_per_step"] = step_size
        _lbfgs_opts["fun_per_step"] = int(_lbfgs_opts["iter_per_step"] * 1.25)

        model.compile("L-BFGS", loss_weights=lbfgs_loss_weights)

        callbacks_lbfgs = []
        if model_save_path:
            callbacks_lbfgs.append(
                dde.callbacks.ModelCheckpoint(
                    model_save_path,
                    save_better_only=False,
                    period=checkpoint_every,
                )
            )
        
        if lbfgs_resample_period > 0:
            callbacks_lbfgs.append(
                dde.callbacks.PDEPointResampler(period=lbfgs_resample_period)
            )

        # Causal training monitor for L-BFGS (curriculum)
        if model._causal_weighter is not None:
            callbacks_lbfgs.append(
                CausalTrainingMonitor(model._causal_weighter, period=100)
            )
            
        losshistory_lbfgs, _ = model.train(
            iterations=lbfgs_iters,
            callbacks=callbacks_lbfgs,
            display_every=100
        )

        if model_save_path:
            model.save(model_save_path + "-final")

        # Combine history
        history = _combine_loss_histories(losshistory_adam, losshistory_lbfgs)
        if history_all is None:
            history_all = history
        else:
            for key in history_all:
                history_all[key].extend(history[key])

    return model, history_all


def train_pinn(
    cfg: Dict,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> Tuple[dde.Model, Dict[str, List[float]]]:
    """
    Train the PINN with DeepXDE: Adam -> L-BFGS.

    Parameters
    ----------
    cfg : dict
        Full experiment config.
    checkpoint_dir : str or None
        Directory for checkpoints (None disables).
    checkpoint_every : int
        Checkpoint interval during Adam phase.
    resume : bool
        If True, restore from the latest checkpoint.

    Returns
    -------
    model : dde.Model
    history : dict  -- per-step loss components
        Keys: L_total, Lr, Lrx, Lrt, Lic, Liv, Lbl, Lbr, w_min
    """
    # Check if curriculum learning is enabled
    curriculum_cfg = cfg["pinn"].get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        return _train_pinn_curriculum(cfg, checkpoint_dir, checkpoint_every, resume)

    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    model, data = build_model(cfg)
    loss_weights = [float(w) for w in cfg["pinn"]["lambda"]]

    print(f"[PINN] Loss weights: {loss_weights}")

    model_save_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_save_path = os.path.join(checkpoint_dir, "model")

    # ---- Check if Adam already completed (for resume) ----
    # DeepXDE appends the step number: model-adam_done-{step}.pt
    adam_done_ckpt = _find_adam_done_checkpoint(checkpoint_dir) if checkpoint_dir else None
    adam_already_done = resume and adam_done_ckpt is not None

    losshistory_adam = None
    lbfgs_resume_ckpt = None  # deferred to after L-BFGS compile

    # Total Adam iterations (needed for global step numbering in L-BFGS)
    adam_iters_total = int(cfg["pinn"]["adam"]["iters"])

    if adam_already_done:
        # ---- Skip Adam: restore weights from adam_done checkpoint ----
        print("[CKPT] Adam already completed — skipping to L-BFGS")
        restore_path = adam_done_ckpt
        model.compile("adam", lr=1e-3, loss_weights=loss_weights)
        # Need one dummy call to initialise train_state before restore
        model.train(iterations=0, display_every=1)
        model.restore(restore_path, verbose=1)

        # Restore gradient-balanced weights saved after Adam
        weights_file = os.path.join(checkpoint_dir, "loss_weights_adam.json")
        if os.path.isfile(weights_file):
            with open(weights_file) as f:
                saved_weights = json.load(f)
            model.loss_weights = saved_weights
            wstr = ", ".join(f"{w:.4f}" for w in saved_weights)
            print(f"[CKPT] Restored gradient-balanced weights: [{wstr}]")

        # Check for a newer L-BFGS checkpoint to continue from.
        # Defer the actual restore to after L-BFGS compile so the
        # optimizer state types match.
        lbfgs_resume_ckpt = _find_latest_checkpoint(
            checkpoint_dir, exclude_prefix="model-adam_done"
        )
        if lbfgs_resume_ckpt is not None:
            print(f"[CKPT] Will restore L-BFGS checkpoint: {lbfgs_resume_ckpt}")

    else:
        # ---- Adam phase ----
        adam_cfg = cfg["pinn"]["adam"]
        adam_iters = int(adam_cfg["iters"])
        lr = float(adam_cfg["lr"])
        resample_period = int(adam_cfg["resample_period"])

        callbacks_adam: List = []
        callbacks_adam.append(
            dde.callbacks.PDEPointResampler(period=resample_period)
        )

        # Gradient balancing (Wang et al. 2021)
        grad_bal_cfg = cfg["pinn"].get("gradient_balancing", {})
        if grad_bal_cfg.get("enabled", False):
            gb_period = int(grad_bal_cfg.get("period", 100))
            gb_alpha = float(grad_bal_cfg.get("alpha", 0.9))
            callbacks_adam.append(
                GradientBalancing(period=gb_period, alpha=gb_alpha)
            )
            print(f"[PINN] Gradient balancing enabled "
                  f"(period={gb_period}, alpha={gb_alpha})")

        # Causal training monitor
        if model._causal_weighter is not None:
            callbacks_adam.append(
                CausalTrainingMonitor(model._causal_weighter, period=100)
            )

        if model_save_path:
            callbacks_adam.append(
                dde.callbacks.ModelCheckpoint(
                    model_save_path,
                    save_better_only=False,
                    period=checkpoint_every,
                )
            )

        # Resume from mid-Adam checkpoint if available
        model_restore_path = None
        if resume and checkpoint_dir:
            ckpt = _find_latest_checkpoint(checkpoint_dir)
            if ckpt is not None:
                model_restore_path = ckpt
                print(f"[CKPT] Restoring from {ckpt}")

        model.compile("adam", lr=lr, loss_weights=loss_weights)

        print(f"[PINN] Adam: {adam_iters} iters, lr={lr}, "
              f"resample every {resample_period}")
        losshistory_adam, _ = model.train(
            iterations=adam_iters,
            callbacks=callbacks_adam,
            display_every=100,
            model_save_path=model_save_path,
            model_restore_path=model_restore_path,
        )

        if model_save_path:
            model.save(model_save_path + "-adam_done")
            print("[CKPT] Adam complete -- checkpoint saved")
            # Persist gradient-balanced weights so L-BFGS resume can use them
            weights_file = os.path.join(checkpoint_dir, "loss_weights_adam.json")
            with open(weights_file, "w") as f:
                json.dump(list(model.loss_weights), f)
            print(f"[CKPT] Gradient-balanced weights saved to {weights_file}")

    # ---- L-BFGS phase ----
    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])

    # Disable causal weighting for L-BFGS — Wang et al. designed it for
    # Adam only.  L-BFGS needs a stationary objective; data-dependent
    # causal weights that change every forward pass violate this.
    if model._causal_weighter is not None:
        model._causal_weighter.active = False
        print("[PINN] Causal weighting disabled for L-BFGS phase")

    # Freeze the gradient-balanced weights from Adam for the L-BFGS phase.
    lbfgs_loss_weights = list(model.loss_weights)
    wstr = ", ".join(f"{w:.4f}" for w in lbfgs_loss_weights)
    print(f"[PINN] L-BFGS loss weights (frozen from Adam): [{wstr}]")

    # Determine how many L-BFGS iterations have already been completed
    lbfgs_iters_done = 0
    if lbfgs_resume_ckpt is not None:
        m = re.search(r"model-(\d+)\.pt$", lbfgs_resume_ckpt)
        if m:
            ckpt_step = int(m.group(1))
            lbfgs_iters_done = max(0, ckpt_step - adam_iters_total)
            if lbfgs_iters_done > 0:
                print(f"[CKPT] L-BFGS iterations already done: {lbfgs_iters_done}")
            else:
                print(f"[CKPT] Checkpoint {lbfgs_resume_ckpt} is from Adam phase — skipping")
                lbfgs_resume_ckpt = None

    iters_remaining = lbfgs_iters - lbfgs_iters_done
    if iters_remaining <= 0:
        print("[PINN] L-BFGS already completed.")
        return model, _convert_loss_history(losshistory_adam) if losshistory_adam else {}

    print(f"[PINN] L-BFGS: {iters_remaining} iterations remaining")

    # Set L-BFGS options
    dde.optimizers.set_LBFGS_options(
        maxcor=100,
        maxiter=iters_remaining,
        ftol=0,
        gtol=1e-8,
        maxls=50,
    )
    
    # DeepXDE executes L-BFGS within a single closure. To enable periodic callbacks
    # (such as ModelCheckpoint and PDEPointResampler) without resetting the optimizer
    # state and losing the Hessian approximation history, we configure iter_per_step.
    # We must set iter_per_step to the greatest common divisor of our callback periods
    # (or simply the minimum period) so that the closure yields control back to the
    # callback loop frequently enough.
    lbfgs_resample_period = int(lbfgs_cfg.get("resample_period", 0))
    if lbfgs_resample_period > 0:
        step_size = min(checkpoint_every, lbfgs_resample_period, iters_remaining)
    else:
        step_size = min(checkpoint_every, iters_remaining)

    from deepxde.optimizers.config import LBFGS_options as _lbfgs_opts
    _lbfgs_opts["iter_per_step"] = step_size
    _lbfgs_opts["fun_per_step"] = int(_lbfgs_opts["iter_per_step"] * 1.25)

    model.compile("L-BFGS", loss_weights=lbfgs_loss_weights)

    if lbfgs_resume_ckpt is not None:
        print(f"[CKPT] Restoring model weights from {lbfgs_resume_ckpt}")
        checkpoint = torch.load(lbfgs_resume_ckpt, weights_only=True)
        model.net.load_state_dict(checkpoint["model_state_dict"])

    callbacks_lbfgs = []
    if model_save_path:
        callbacks_lbfgs.append(
            dde.callbacks.ModelCheckpoint(
                model_save_path,
                save_better_only=False,
                period=checkpoint_every,
            )
        )
    
    # Add PDEPointResampler for L-BFGS if configured
    if lbfgs_resample_period > 0:
        callbacks_lbfgs.append(
            dde.callbacks.PDEPointResampler(period=lbfgs_resample_period)
        )
        print(f"[PINN] L-BFGS: resampling collocation points every {lbfgs_resample_period} iterations")

    # Causal training monitor for L-BFGS
    if model._causal_weighter is not None:
        callbacks_lbfgs.append(
            CausalTrainingMonitor(model._causal_weighter, period=100)
        )

    losshistory_lbfgs, _ = model.train(
        iterations=iters_remaining,
        callbacks=callbacks_lbfgs,
        display_every=100
    )

    if model_save_path:
        model.save(model_save_path + "-final")
        print("[CKPT] Training complete -- final checkpoint saved")

    # ---- Combine loss histories ----
    if losshistory_adam is not None:
        history = _combine_loss_histories(losshistory_adam, losshistory_lbfgs)
    else:
        history = _convert_loss_history(losshistory_lbfgs)

    return model, history


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(
    checkpoint_dir: str, exclude_prefix: Optional[str] = None
) -> Optional[str]:
    """Find the latest DeepXDE checkpoint in the directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to search.
    exclude_prefix : str or None
        If given, skip checkpoints whose filename starts with this prefix.
        Useful for skipping the adam_done marker when looking for L-BFGS
        checkpoints.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(".pt"):
            continue
        if exclude_prefix and f.startswith(exclude_prefix):
            continue
        candidates.append(os.path.join(checkpoint_dir, f))
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return latest


def _find_adam_done_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the model-adam_done-*.pt checkpoint if it exists.

    Returns the full path (with .pt extension) or None.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    for f in os.listdir(checkpoint_dir):
        if f.startswith("model-adam_done") and f.endswith(".pt"):
            return os.path.join(checkpoint_dir, f)
    return None


def _find_final_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the model-final-*.pt checkpoint if it exists.

    DeepXDE appends the step number, so the file is model-final-{step}.pt.
    Returns the full path or None.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    for f in os.listdir(checkpoint_dir):
        if f.startswith("model-final") and f.endswith(".pt"):
            return os.path.join(checkpoint_dir, f)
    return None


# ---------------------------------------------------------------------------
# Loss-history conversion
# ---------------------------------------------------------------------------

def _convert_loss_history(losshistory) -> Dict[str, List[float]]:
    """Convert a single DeepXDE LossHistory to our dict format.

    DeepXDE records per-component MSEs (unweighted).
    Order: [PDE outputs ..., IC/BCs ...] = [r, r_x, r_t, ic, iv, bl, br].
    """
    loss_names = ["Lr", "Lrx", "Lrt", "Lic", "Liv", "Lbl", "Lbr"]
    history: Dict[str, List[float]] = {name: [] for name in loss_names}
    history["L_total"] = []
    history["w_min"] = []  # placeholder (causal training not yet ported)

    steps = losshistory.steps
    losses = np.array(losshistory.loss_train)

    for i in range(len(steps)):
        total = 0.0
        for j, name in enumerate(loss_names):
            val = float(losses[i, j]) if j < losses.shape[1] else 0.0
            history[name].append(val)
            total += val
        history["L_total"].append(total)
        history["w_min"].append(1.0)

    return history


def _combine_loss_histories(lh_adam, lh_lbfgs) -> Dict[str, List[float]]:
    """Concatenate Adam and L-BFGS loss histories."""
    h1 = _convert_loss_history(lh_adam)
    h2 = _convert_loss_history(lh_lbfgs)
    combined: Dict[str, List[float]] = {}
    for key in h1:
        combined[key] = h1[key] + h2[key]
    return combined


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_on_grid(
    model, x: np.ndarray, t: np.ndarray, dtype: str = "float64"
) -> np.ndarray:
    """
    Evaluate the model on a full space-time grid.

    Returns phi[t_index, x_index].
    """
    X_list = []
    for ti in t:
        X_list.append(np.stack([x, np.full_like(x, ti)], axis=1))
    X = np.concatenate(X_list, axis=0)

    if isinstance(model, dde.Model):
        y = model.predict(X)
        return y.reshape(len(t), len(x))

    # Backward compatibility for raw PyTorch model
    device = next(model.parameters()).device
    tdtype = torch.float64 if dtype == "float64" else torch.float32
    X_t = torch.tensor(X, dtype=tdtype, device=device)
    with torch.no_grad():
        y = model(X_t).cpu().numpy()
    return y.reshape(len(t), len(x))
