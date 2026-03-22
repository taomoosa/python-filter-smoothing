"""Utilities for Real-Time Chunking (RTC) inference and training.

This module provides standalone helper functions for:

* **ΠGDM guidance** – Computing the Projected Gradient Descent Matching
  guidance term for inference-time RTC (arXiv:2506.07339, Eq. 4).
* **Training-time RTC** – Preparing batches with simulated delay and
  prefix conditioning (arXiv:2512.05964), and sampling action chunks
  from a trained prefix-conditioned model.

These functions are **model-agnostic**: they accept a model callable
(any function mapping ``(x_t, obs, tau) → velocity``) and operate on
NumPy arrays.  For best performance with deep-learning frameworks
(PyTorch / JAX), pass a ``vjp_fn`` that leverages native autodiff
instead of the default finite-difference approximation.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Union

import numpy as np


# =====================================================================
# ΠGDM Guidance (Inference-Time RTC)
# =====================================================================


def rtc_soft_mask(
    horizon: int,
    delay: int,
    execution_horizon: int,
) -> np.ndarray:
    """Compute the RTC exponential-decay soft mask (Eq. 5).

    Parameters
    ----------
    horizon : int
        Prediction horizon *H*.
    delay : int
        Estimated inference delay *d* (in controller timesteps).
    execution_horizon : int
        Execution horizon *s* (``max(d, s_min)``).

    Returns
    -------
    np.ndarray
        Shape ``(H,)`` with values in ``[0, 1]``.
        ``W[i] = 1`` for the frozen prefix, decaying exponentially in
        the overlap region, and ``0`` in the free region.
    """
    H = int(horizon)
    d = max(0, min(int(delay), H - 1))
    s = max(1, min(int(execution_horizon), H))
    W = np.zeros(H)
    denom = H - s - d + 1
    for i in range(H):
        if i < d:
            W[i] = 1.0
        elif i < H - s and denom > 0:
            c = (H - s - i) / denom
            W[i] = c * (np.exp(c) - 1.0) / (np.e - 1.0)
    return W


def _numerical_vjp(
    model_fn: Callable,
    x_t: np.ndarray,
    obs: Any,
    tau: Union[float, np.ndarray],
    vec: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Approximate the VJP  vec @ ∂Â¹/∂A^τ  via central finite differences.

    Â¹ = A^τ + (1 − τ) · v(A^τ, o, τ)

    The full Jacobian is ``(H*D, H*D)``; we compute the VJP directly
    (one pair of forward passes per element of A^τ) using column-wise
    finite differences.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(x_t, obs, tau) -> velocity``  with shapes matching *x_t*.
    x_t : np.ndarray, shape (H, D)
    obs : any
    tau : float or array broadcastable to (H,) or (H, D)
    vec : np.ndarray, shape (H, D)
        The vector being left-multiplied against the Jacobian.
    eps : float
        Finite-difference step size.

    Returns
    -------
    np.ndarray, shape (H, D)
    """
    H, D = x_t.shape
    tau_arr = np.broadcast_to(np.asarray(tau, dtype=float), x_t.shape)
    one_minus_tau = 1.0 - tau_arr

    result = np.zeros_like(x_t)
    for i in range(H):
        for j in range(D):
            x_plus = x_t.copy()
            x_plus[i, j] += eps
            x_minus = x_t.copy()
            x_minus[i, j] -= eps

            v_plus = np.asarray(model_fn(x_plus, obs, tau), dtype=float)
            v_minus = np.asarray(model_fn(x_minus, obs, tau), dtype=float)

            # Â¹ = x + (1-τ)*v  → ∂Â¹/∂x[i,j] = δ_{i,j} + (1-τ) * ∂v/∂x[i,j]
            dv = (v_plus - v_minus) / (2.0 * eps)
            dA1 = np.zeros_like(x_t)
            dA1[i, j] = 1.0
            dA1 += one_minus_tau * dv
            result[i, j] = np.sum(vec * dA1)
    return result


def rtc_pigdm_guidance(
    model_fn: Callable,
    x_t: np.ndarray,
    obs: Any,
    tau: Union[float, np.ndarray],
    prefix: np.ndarray,
    mask: np.ndarray,
    *,
    beta: float = 1.0,
    vjp_fn: Optional[Callable] = None,
    eps: float = 1e-4,
) -> np.ndarray:
    r"""Compute the ΠGDM guidance term (Eq. 4 of arXiv:2506.07339).

    .. math::

        \mathbf{g} = \min\!\Bigl(\beta,\;
            \frac{1-\tau}{\tau \cdot r_\tau^2}\Bigr)
        \;(\mathbf{Y} - \hat{\mathbf{A}}^1)^\top
        \operatorname{diag}(\mathbf{W})
        \;\frac{\partial \hat{\mathbf{A}}^1}{\partial \mathbf{A}^\tau}

    where :math:`\hat{\mathbf{A}}^1 = \mathbf{A}^\tau + (1-\tau)\mathbf{v}`.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(x_t, obs, tau) -> velocity``.  Input and output
        shapes must be ``(H, D)``.
    x_t : np.ndarray, shape (H, D)
        Current noisy action trajectory at flow-matching time *τ*.
    obs : any
        Observation token(s), passed through to *model_fn*.
    tau : float or array
        Flow-matching timestep(s) in ``(0, 1]``.
    prefix : np.ndarray, shape (H, D)
        Target (frozen) actions for the prefix region — typically the
        ground-truth actions from the previous chunk.
    mask : np.ndarray, shape (H,)
        Soft mask ``W`` (e.g. from :func:`rtc_soft_mask`).
    beta : float, optional
        Upper bound on the guidance weight (default ``1.0``).
    vjp_fn : callable, optional
        If provided, used instead of finite-difference VJP::

            vjp_fn(model_fn, x_t, obs, tau, vec) -> np.ndarray

        where *vec* is the ``(H, D)`` left-vector.  Implement this with
        framework-native autodiff for efficiency.
    eps : float, optional
        Finite-difference step (used only when *vjp_fn* is ``None``).

    Returns
    -------
    np.ndarray, shape (H, D)
        Guidance correction to add to the base velocity.
    """
    x_t = np.asarray(x_t, dtype=float)
    prefix = np.asarray(prefix, dtype=float)
    mask = np.asarray(mask, dtype=float)
    H, D = x_t.shape

    tau_scalar = float(np.mean(tau)) if np.ndim(tau) > 0 else float(tau)
    tau_arr = np.broadcast_to(np.asarray(tau, dtype=float), x_t.shape)

    # Base velocity and one-step prediction
    v = np.asarray(model_fn(x_t, obs, tau), dtype=float)
    A_hat_1 = x_t + (1.0 - tau_arr) * v  # predicted clean actions

    # Residual weighted by soft mask: (Y - Â¹) * diag(W)
    residual = prefix - A_hat_1  # (H, D)
    W_diag = mask.reshape(H, 1)  # broadcast to (H, D)
    weighted_residual = residual * W_diag  # (H, D)

    # VJP: weighted_residual @ ∂Â¹/∂A^τ
    if vjp_fn is not None:
        vjp_result = np.asarray(
            vjp_fn(model_fn, x_t, obs, tau, weighted_residual), dtype=float
        )
    else:
        vjp_result = _numerical_vjp(
            model_fn, x_t, obs, tau, weighted_residual, eps
        )

    # Guidance weight: min(β, (1-τ) / (τ · r²_τ))
    # r²_τ = (1-τ)² / (τ² + (1-τ)²)
    tau_c = np.clip(tau_scalar, 1e-8, 1.0 - 1e-8)
    r_sq = (1.0 - tau_c) ** 2 / (tau_c**2 + (1.0 - tau_c) ** 2)
    weight = min(beta, (1.0 - tau_c) / (tau_c * r_sq + 1e-12))

    return weight * vjp_result


def rtc_pigdm_denoise_step(
    model_fn: Callable,
    x_t: np.ndarray,
    obs: Any,
    tau: float,
    dt_flow: float,
    prefix: np.ndarray,
    mask: np.ndarray,
    *,
    beta: float = 1.0,
    vjp_fn: Optional[Callable] = None,
    eps: float = 1e-4,
) -> tuple[np.ndarray, float]:
    r"""Perform one Euler denoising step with ΠGDM guidance.

    .. math::

        \mathbf{A}^{\tau + \Delta\tau}
          = \mathbf{A}^\tau + \Delta\tau \bigl(
              \mathbf{v} + \mathbf{g}_{\Pi\text{GDM}}
            \bigr)

    Parameters
    ----------
    model_fn : callable
        ``model_fn(x_t, obs, tau) -> velocity``.
    x_t : np.ndarray, shape (H, D)
        Current noisy trajectory.
    obs : any
        Observation.
    tau : float
        Current flow-matching time.
    dt_flow : float
        Integration step size (e.g. ``1 / num_denoise_steps``).
    prefix, mask, beta, vjp_fn, eps
        Same as :func:`rtc_pigdm_guidance`.

    Returns
    -------
    x_next : np.ndarray, shape (H, D)
        Updated trajectory at ``τ + dt_flow``.
    tau_next : float
        ``τ + dt_flow``.
    """
    x_t = np.asarray(x_t, dtype=float)
    tau_arr = np.full(x_t.shape, tau)

    v = np.asarray(model_fn(x_t, obs, tau), dtype=float)
    g = rtc_pigdm_guidance(
        model_fn, x_t, obs, tau, prefix, mask,
        beta=beta, vjp_fn=vjp_fn, eps=eps,
    )
    x_next = x_t + dt_flow * (v + g)
    return x_next, tau + dt_flow


# =====================================================================
# Training-Time RTC (arXiv:2512.05964)
# =====================================================================


def rtc_training_prepare_batch(
    action_chunks: np.ndarray,
    max_delay: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, np.ndarray]:
    r"""Prepare a training batch with simulated delay and prefix conditioning.

    Implements the training-time RTC procedure from arXiv:2512.05964:

    1. Sample a random delay ``d ∈ [0, max_delay)`` per example.
    2. Assign flow-matching time ``τ = 1`` to prefix positions
       (``i < d``), random ``τ`` to postfix positions.
    3. Construct noised trajectories:
       ``x_t = τ · action + (1 − τ) · noise``.
    4. Compute the flow-matching target: ``v_target = action − noise``.
    5. Build a loss mask that excludes prefix positions.

    Parameters
    ----------
    action_chunks : np.ndarray, shape (B, H, D)
        Ground-truth action chunks.
    max_delay : int
        Upper bound (exclusive) for sampled delays.
    rng : np.random.Generator, optional
        Random number generator (default: ``np.random.default_rng()``).

    Returns
    -------
    dict with keys:
        ``'noise'``       – (B, H, D)  sampled noise.
        ``'delay'``       – (B,)       sampled delays.
        ``'time'``        – (B, H)     per-token flow-matching τ.
        ``'prefix_mask'`` – (B, H)     bool, ``True`` for prefix positions.
        ``'x_t'``         – (B, H, D)  noised actions with prefix at τ=1.
        ``'target_v'``    – (B, H, D)  flow-matching velocity target.
        ``'loss_mask'``   – (B, H)     bool, ``True`` for postfix (compute loss here).
    """
    action_chunks = np.asarray(action_chunks, dtype=float)
    if action_chunks.ndim != 3:
        raise ValueError("action_chunks must be 3-D (B, H, D).")
    B, H, D = action_chunks.shape
    if max_delay < 1:
        raise ValueError("max_delay must be >= 1.")

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.standard_normal((B, H, D))
    delay = rng.integers(0, max_delay, size=(B,))

    # Per-example random τ (scalar per example, broadcast to H)
    base_time = rng.uniform(0.0, 1.0, size=(B,))
    time = np.broadcast_to(base_time[:, None], (B, H)).copy()

    # Prefix mask: True for i < delay[b]
    idx = np.arange(H)[None, :]  # (1, H)
    prefix_mask = idx < delay[:, None]  # (B, H)

    # Override τ = 1.0 for prefix positions
    time[prefix_mask] = 1.0

    # Construct noised trajectory
    time_3d = time[:, :, None]  # (B, H, 1)
    x_t = time_3d * action_chunks + (1.0 - time_3d) * noise

    # Flow-matching velocity target
    target_v = action_chunks - noise

    # Loss mask: postfix only
    loss_mask = ~prefix_mask

    return {
        "noise": noise,
        "delay": delay,
        "time": time,
        "prefix_mask": prefix_mask,
        "x_t": x_t,
        "target_v": target_v,
        "loss_mask": loss_mask,
    }


def rtc_training_sample(
    model_fn: Callable,
    obs: Any,
    action_prefix: np.ndarray,
    delay: int,
    horizon: int,
    action_dim: int,
    *,
    num_steps: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    r"""Sample an action chunk using training-time RTC prefix conditioning.

    Implements the Euler ODE solver with prefix replacement at each step
    (arXiv:2512.05964, Algorithm 2):

    1. Start from random noise ``x_0 ~ N(0, I)``.
    2. At each denoising step, replace prefix positions with
       ``action_prefix`` and set their ``τ = 1``.
    3. Integrate: ``x_{t+Δt} = x_t + Δt · v(x_t, obs, τ_t)``.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(x_t, obs, tau) -> velocity``.  Accepts:

        * ``x_t``:  shape ``(H, D)`` or ``(B, H, D)``
        * ``obs``:  any
        * ``tau``:  float, or array broadcastable to ``x_t``

    obs : any
        Observation token(s).
    action_prefix : np.ndarray, shape (d, D) or (B, d, D)
        Ground-truth actions for the prefix region.
    delay : int
        Number of prefix steps *d*.
    horizon : int
        Prediction horizon *H*.
    action_dim : int
        Action dimensionality *D*.
    num_steps : int, optional
        Number of Euler integration steps (default ``10``).
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray, shape (H, D) or (B, H, D)
        Sampled action chunk.
    """
    if rng is None:
        rng = np.random.default_rng()

    prefix = np.asarray(action_prefix, dtype=float)
    batched = prefix.ndim == 3
    if not batched:
        prefix = prefix[None, ...]  # (1, d, D)
    B = prefix.shape[0]

    x_t = rng.standard_normal((B, horizon, action_dim))
    dt_flow = 1.0 / num_steps
    tau = 0.0

    prefix_mask = np.arange(horizon)[None, :] < delay  # (1, H)

    for _ in range(num_steps):
        # Replace prefix with ground truth
        x_t[:, :delay, :] = prefix[:, :delay, :]

        # Per-token τ: prefix gets 1.0, postfix gets current τ
        tau_arr = np.where(prefix_mask[:, :, None], 1.0, tau)
        tau_arr = np.broadcast_to(tau_arr, x_t.shape)

        v = np.asarray(model_fn(x_t, obs, tau_arr), dtype=float)
        x_t = x_t + dt_flow * v
        tau = tau + dt_flow

    # Final prefix replacement
    x_t[:, :delay, :] = prefix[:, :delay, :]

    if not batched:
        return x_t[0]
    return x_t
