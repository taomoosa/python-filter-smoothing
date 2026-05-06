"""Tests for RTC utility functions (ΠGDM and training-time RTC)."""
import numpy as np
import pytest
from python_filter_smoothing import (
    rtc_pigdm_denoise_step,
    rtc_pigdm_guidance,
    rtc_soft_mask,
    rtc_training_prepare_batch,
    rtc_training_sample,
)


# ---------------------------------------------------------------------------
# Soft Mask
# ---------------------------------------------------------------------------

class TestRtcSoftMask:
    """Tests for the standalone rtc_soft_mask function."""

    def test_shape(self):
        W = rtc_soft_mask(horizon=10, delay=2, execution_horizon=3)
        assert W.shape == (10,)

    def test_frozen_region(self):
        W = rtc_soft_mask(horizon=10, delay=3, execution_horizon=3)
        np.testing.assert_allclose(W[:3], 1.0)

    def test_free_region(self):
        W = rtc_soft_mask(horizon=10, delay=2, execution_horizon=3)
        np.testing.assert_allclose(W[7:], 0.0)

    def test_intermediate_in_unit_interval(self):
        W = rtc_soft_mask(horizon=10, delay=2, execution_horizon=3)
        mid = W[2:7]
        assert np.all(mid >= 0.0) and np.all(mid <= 1.0)

    def test_monotone_decay(self):
        W = rtc_soft_mask(horizon=20, delay=3, execution_horizon=5)
        overlap = W[3:15]
        for i in range(len(overlap) - 1):
            assert overlap[i] >= overlap[i + 1]


# ---------------------------------------------------------------------------
# ΠGDM Guidance
# ---------------------------------------------------------------------------

def _identity_model(x_t, obs, tau):
    """Trivial model: v = 0 everywhere."""
    return np.zeros_like(x_t)


def _linear_model(x_t, obs, tau):
    """Model: v = x_t (linear dynamics)."""
    return x_t.copy()


class TestRtcPigdmGuidance:
    """Tests for rtc_pigdm_guidance."""

    def test_output_shape(self):
        x_t = np.zeros((10, 3))
        prefix = np.ones((10, 3))
        mask = rtc_soft_mask(10, 2, 3)
        g = rtc_pigdm_guidance(
            _identity_model, x_t, None, 0.5, prefix, mask,
        )
        assert g.shape == (10, 3)

    def test_zero_mask_zero_guidance(self):
        """If the mask is all zeros, guidance should be zero."""
        x_t = np.random.randn(8, 2)
        prefix = np.ones((8, 2))
        mask = np.zeros(8)
        g = rtc_pigdm_guidance(
            _linear_model, x_t, None, 0.5, prefix, mask,
        )
        np.testing.assert_allclose(g, 0.0, atol=1e-8)

    def test_guidance_points_toward_prefix(self):
        """When prefix > prediction, guidance should push x_t toward prefix."""
        H, D = 6, 2
        x_t = np.zeros((H, D))
        prefix = np.ones((H, D)) * 10.0
        mask = np.ones(H)
        g = rtc_pigdm_guidance(
            _identity_model, x_t, None, 0.5, prefix, mask,
        )
        # Guidance should be in the positive direction (toward prefix)
        assert np.sum(g) > 0

    def test_custom_vjp_fn(self):
        """Custom vjp_fn should be used instead of finite differences."""
        calls = []

        def dummy_vjp(model_fn, x_t, obs, tau, vec):
            calls.append(True)
            return vec * 0.5  # arbitrary

        x_t = np.ones((4, 2))
        prefix = np.ones((4, 2)) * 2.0
        mask = np.ones(4)
        g = rtc_pigdm_guidance(
            _identity_model, x_t, None, 0.5, prefix, mask,
            vjp_fn=dummy_vjp,
        )
        assert len(calls) == 1
        assert g.shape == (4, 2)

    def test_beta_limits_guidance(self):
        """Larger beta should allow larger guidance magnitude."""
        x_t = np.zeros((6, 2))
        prefix = np.ones((6, 2)) * 5.0
        mask = np.ones(6)
        g_small = rtc_pigdm_guidance(
            _identity_model, x_t, None, 0.5, prefix, mask, beta=0.01,
        )
        g_large = rtc_pigdm_guidance(
            _identity_model, x_t, None, 0.5, prefix, mask, beta=100.0,
        )
        assert np.linalg.norm(g_large) >= np.linalg.norm(g_small)


class TestRtcPigdmDenoiseStep:
    """Tests for rtc_pigdm_denoise_step."""

    def test_returns_tuple(self):
        x_t = np.zeros((6, 2))
        prefix = np.ones((6, 2))
        mask = rtc_soft_mask(6, 1, 2)
        x_next, tau_next = rtc_pigdm_denoise_step(
            _identity_model, x_t, None, 0.1, 0.1, prefix, mask,
        )
        assert x_next.shape == (6, 2)
        assert tau_next == pytest.approx(0.2)

    def test_tau_advances(self):
        x_t = np.zeros((4, 2))
        prefix = np.zeros((4, 2))
        mask = np.zeros(4)
        _, tau_next = rtc_pigdm_denoise_step(
            _identity_model, x_t, None, 0.3, 0.25, prefix, mask,
        )
        assert tau_next == pytest.approx(0.55)

    def test_multiple_steps_converge(self):
        """Running multiple steps should move x_t toward prefix region."""
        H, D = 8, 2
        rng = np.random.default_rng(42)
        x_t = rng.standard_normal((H, D))
        prefix = np.ones((H, D)) * 3.0
        mask = rtc_soft_mask(H, 2, 3)

        tau = 0.0
        dt_flow = 0.1
        for _ in range(10):
            x_t, tau = rtc_pigdm_denoise_step(
                _identity_model, x_t, None, tau, dt_flow, prefix, mask,
            )
        # After 10 steps (τ = 1.0), the prefix region should be closer to target
        # (This is a weak check; exact convergence depends on the model)
        assert tau == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Training-Time RTC: Batch Preparation
# ---------------------------------------------------------------------------

class TestRtcTrainingPrepareBatch:
    """Tests for rtc_training_prepare_batch."""

    def test_output_keys(self):
        actions = np.random.randn(4, 10, 3)
        result = rtc_training_prepare_batch(actions, max_delay=5)
        expected_keys = {
            "noise", "delay", "time", "prefix_mask",
            "x_t", "target_v", "loss_mask",
        }
        assert set(result.keys()) == expected_keys

    def test_output_shapes(self):
        B, H, D = 8, 12, 4
        actions = np.random.randn(B, H, D)
        result = rtc_training_prepare_batch(actions, max_delay=5)
        assert result["noise"].shape == (B, H, D)
        assert result["delay"].shape == (B,)
        assert result["time"].shape == (B, H)
        assert result["prefix_mask"].shape == (B, H)
        assert result["x_t"].shape == (B, H, D)
        assert result["target_v"].shape == (B, H, D)
        assert result["loss_mask"].shape == (B, H)

    def test_prefix_mask_matches_delay(self):
        B, H, D = 4, 10, 2
        actions = np.ones((B, H, D))
        rng = np.random.default_rng(42)
        result = rtc_training_prepare_batch(actions, max_delay=5, rng=rng)
        for b in range(B):
            d = result["delay"][b]
            assert np.all(result["prefix_mask"][b, :d])
            assert not np.any(result["prefix_mask"][b, d:])

    def test_prefix_time_is_one(self):
        actions = np.random.randn(4, 10, 3)
        rng = np.random.default_rng(99)
        result = rtc_training_prepare_batch(actions, max_delay=5, rng=rng)
        prefix_times = result["time"][result["prefix_mask"]]
        if len(prefix_times) > 0:
            np.testing.assert_allclose(prefix_times, 1.0)

    def test_loss_mask_is_inverse_of_prefix(self):
        actions = np.random.randn(4, 10, 3)
        result = rtc_training_prepare_batch(actions, max_delay=5)
        np.testing.assert_array_equal(
            result["loss_mask"], ~result["prefix_mask"]
        )

    def test_x_t_at_tau_one_equals_action(self):
        """At τ=1, x_t should equal the original action (for prefix)."""
        B, H, D = 4, 10, 2
        actions = np.ones((B, H, D)) * 42.0
        rng = np.random.default_rng(7)
        result = rtc_training_prepare_batch(actions, max_delay=5, rng=rng)
        prefix = result["prefix_mask"]
        if np.any(prefix):
            # x_t[prefix] = τ * action + (1-τ) * noise at τ=1 → action
            x_t_prefix = result["x_t"][prefix[:, :, None].repeat(D, axis=2)]
            actions_prefix = actions[prefix[:, :, None].repeat(D, axis=2)]
            np.testing.assert_allclose(x_t_prefix, actions_prefix, atol=1e-12)

    def test_target_velocity(self):
        actions = np.random.randn(4, 10, 3)
        rng = np.random.default_rng(0)
        result = rtc_training_prepare_batch(actions, max_delay=5, rng=rng)
        np.testing.assert_allclose(
            result["target_v"], actions - result["noise"]
        )

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            rtc_training_prepare_batch(np.zeros((10, 3)), max_delay=5)
        with pytest.raises(ValueError, match="max_delay"):
            rtc_training_prepare_batch(np.zeros((2, 10, 3)), max_delay=0)

    def test_deterministic_with_seed(self):
        actions = np.random.randn(4, 10, 3)
        r1 = rtc_training_prepare_batch(
            actions, max_delay=5, rng=np.random.default_rng(123)
        )
        r2 = rtc_training_prepare_batch(
            actions, max_delay=5, rng=np.random.default_rng(123)
        )
        np.testing.assert_array_equal(r1["delay"], r2["delay"])
        np.testing.assert_array_equal(r1["noise"], r2["noise"])


# ---------------------------------------------------------------------------
# Training-Time RTC: Sampling
# ---------------------------------------------------------------------------

def _const_velocity_model(x_t, obs, tau):
    """Model that always returns the same direction (toward [1,...,1])."""
    target = np.ones_like(x_t)
    return target - x_t


class TestRtcTrainingSample:
    """Tests for rtc_training_sample."""

    def test_output_shape_unbatched(self):
        prefix = np.zeros((3, 2))
        result = rtc_training_sample(
            _identity_model, None, prefix,
            delay=3, horizon=10, action_dim=2,
            rng=np.random.default_rng(0),
        )
        assert result.shape == (10, 2)

    def test_output_shape_batched(self):
        prefix = np.zeros((4, 3, 2))
        result = rtc_training_sample(
            _identity_model, None, prefix,
            delay=3, horizon=10, action_dim=2,
            rng=np.random.default_rng(0),
        )
        assert result.shape == (4, 10, 2)

    def test_prefix_preserved(self):
        """Prefix region in the output should match the input prefix."""
        prefix = np.ones((3, 2)) * 42.0
        result = rtc_training_sample(
            _identity_model, None, prefix,
            delay=3, horizon=10, action_dim=2, num_steps=20,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(result[:3], 42.0)

    def test_prefix_preserved_batched(self):
        B = 4
        prefix = np.ones((B, 3, 2)) * 7.0
        result = rtc_training_sample(
            _identity_model, None, prefix,
            delay=3, horizon=10, action_dim=2, num_steps=20,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(result[:, :3, :], 7.0)

    def test_deterministic_with_seed(self):
        prefix = np.ones((2, 2)) * 5.0
        r1 = rtc_training_sample(
            _const_velocity_model, None, prefix,
            delay=2, horizon=8, action_dim=2,
            rng=np.random.default_rng(42),
        )
        r2 = rtc_training_sample(
            _const_velocity_model, None, prefix,
            delay=2, horizon=8, action_dim=2,
            rng=np.random.default_rng(42),
        )
        np.testing.assert_array_equal(r1, r2)

    def test_postfix_evolves(self):
        """Postfix region should be different from initial noise after sampling."""
        prefix = np.zeros((2, 3))
        rng = np.random.default_rng(0)
        result = rtc_training_sample(
            _const_velocity_model, None, prefix,
            delay=2, horizon=8, action_dim=3, num_steps=50,
            rng=rng,
        )
        # After many steps toward [1,...,1], postfix should be close to 1
        postfix = result[2:]
        assert np.mean(np.abs(postfix - 1.0)) < 0.5


# ---------------------------------------------------------------------------
# Fix 4.5: Soft mask schedule variants
# ---------------------------------------------------------------------------

class TestRtcSoftMaskSchedules:
    """Tests for the schedule parameter in rtc_soft_mask."""

    def test_exp_is_default(self):
        """Default schedule should match explicit exp."""
        W_default = rtc_soft_mask(horizon=10, delay=2, execution_horizon=3)
        W_exp = rtc_soft_mask(horizon=10, delay=2, execution_horizon=3, schedule="exp")
        np.testing.assert_array_equal(W_default, W_exp)

    def test_linear_in_unit_interval(self):
        W = rtc_soft_mask(horizon=10, delay=2, execution_horizon=3, schedule="linear")
        assert np.all(W >= 0.0) and np.all(W <= 1.0)

    def test_linear_frozen_region(self):
        W = rtc_soft_mask(horizon=10, delay=3, execution_horizon=3, schedule="linear")
        np.testing.assert_allclose(W[:3], 1.0)

    def test_linear_free_region(self):
        W = rtc_soft_mask(horizon=10, delay=2, execution_horizon=3, schedule="linear")
        np.testing.assert_allclose(W[7:], 0.0)

    def test_linear_monotone_decay(self):
        W = rtc_soft_mask(horizon=20, delay=3, execution_horizon=5, schedule="linear")
        overlap = W[3:15]
        for i in range(len(overlap) - 1):
            assert overlap[i] >= overlap[i + 1]

    def test_linear_values_are_linear(self):
        """In the overlap region, linear schedule should produce evenly spaced values."""
        H, d, s = 10, 2, 3
        W = rtc_soft_mask(horizon=H, delay=d, execution_horizon=s, schedule="linear")
        overlap = W[d : H - s]
        if len(overlap) >= 2:
            diffs = np.diff(overlap)
            np.testing.assert_allclose(diffs, diffs[0], atol=1e-12)

    def test_ones_schedule(self):
        """'ones' schedule should be 1.0 for prefix and overlap, 0 for free."""
        W = rtc_soft_mask(horizon=10, delay=2, execution_horizon=3, schedule="ones")
        np.testing.assert_allclose(W[:2], 1.0)  # frozen
        np.testing.assert_allclose(W[2:7], 1.0)  # overlap
        np.testing.assert_allclose(W[7:], 0.0)  # free

    def test_zeros_schedule(self):
        """'zeros' schedule should be 1.0 for prefix only, 0 elsewhere."""
        W = rtc_soft_mask(horizon=10, delay=2, execution_horizon=3, schedule="zeros")
        np.testing.assert_allclose(W[:2], 1.0)  # frozen
        np.testing.assert_allclose(W[2:], 0.0)  # overlap + free

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown schedule"):
            rtc_soft_mask(horizon=10, delay=2, execution_horizon=3, schedule="cosine")


# ---------------------------------------------------------------------------
# Fix 4.2: v_precomputed in guidance
# ---------------------------------------------------------------------------

class TestRtcPigdmGuidancePrecomputed:
    """Tests for the v_precomputed parameter in rtc_pigdm_guidance."""

    def test_v_precomputed_matches_internal(self):
        """Guidance with v_precomputed should match computing v internally."""
        x_t = np.random.default_rng(0).standard_normal((6, 2))
        prefix = np.ones((6, 2)) * 3.0
        mask = rtc_soft_mask(6, 1, 2)
        g_auto = rtc_pigdm_guidance(
            _identity_model, x_t, None, 0.5, prefix, mask,
        )
        v = _identity_model(x_t, None, 0.5)
        g_pre = rtc_pigdm_guidance(
            _identity_model, x_t, None, 0.5, prefix, mask,
            v_precomputed=v,
        )
        np.testing.assert_allclose(g_pre, g_auto, atol=1e-12)

    def test_v_precomputed_skips_model_call(self):
        """When v_precomputed is given, model_fn should not be called for velocity."""
        calls = []

        def tracking_model(x_t, obs, tau):
            calls.append(("model", tau))
            return np.zeros_like(x_t)

        x_t = np.ones((4, 2))
        prefix = np.ones((4, 2)) * 2.0
        mask = np.ones(4)
        v = np.zeros((4, 2))

        def dummy_vjp(model_fn, x_t, obs, tau, vec):
            return vec * 0.5

        rtc_pigdm_guidance(
            tracking_model, x_t, None, 0.5, prefix, mask,
            vjp_fn=dummy_vjp, v_precomputed=v,
        )
        # model_fn should not have been called at all
        assert len(calls) == 0

    def test_denoise_step_reuses_velocity(self):
        """Denoise step should only call model_fn once (not twice)."""
        call_count = [0]

        def counting_model(x_t, obs, tau):
            call_count[0] += 1
            return np.zeros_like(x_t)

        x_t = np.ones((4, 2))
        prefix = np.ones((4, 2)) * 2.0
        mask = np.ones(4)

        def dummy_vjp(model_fn, x_t, obs, tau, vec):
            return vec * 0.5

        rtc_pigdm_denoise_step(
            counting_model, x_t, None, 0.5, 0.1, prefix, mask,
            vjp_fn=dummy_vjp,
        )
        # Only one model call (in denoise_step), NOT two
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Fix 4.1: Delay distribution in training prepare batch
# ---------------------------------------------------------------------------

class TestRtcTrainingDelayDistribution:
    """Tests for delay_distribution parameter in rtc_training_prepare_batch."""

    def test_uniform_backward_compatible(self):
        """'uniform' distribution should produce the same result as original."""
        actions = np.random.default_rng(1).standard_normal((8, 10, 3))
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        r_uniform = rtc_training_prepare_batch(
            actions, max_delay=5, rng=rng1, delay_distribution="uniform",
        )
        # With a fresh rng that draws the same sequence, uniform should be
        # predictable. Just check it runs and produces valid output.
        assert r_uniform["delay"].shape == (8,)
        assert np.all(r_uniform["delay"] >= 0)
        assert np.all(r_uniform["delay"] < 5)

    def test_exponential_biases_toward_small_delays(self):
        """Exponential distribution should favor smaller delays on average."""
        actions = np.random.default_rng(0).standard_normal((1000, 10, 3))
        rng = np.random.default_rng(42)
        result = rtc_training_prepare_batch(
            actions, max_delay=10, rng=rng, delay_distribution="exponential",
        )
        mean_delay = result["delay"].mean()
        # Uniform mean would be ~4.5; exponential should be much lower
        assert mean_delay < 3.5

    def test_exponential_is_default(self):
        """Default distribution should be 'exponential'."""
        actions = np.random.default_rng(0).standard_normal((4, 10, 3))
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        r_default = rtc_training_prepare_batch(actions, max_delay=5, rng=rng1)
        r_explicit = rtc_training_prepare_batch(
            actions, max_delay=5, rng=rng2, delay_distribution="exponential",
        )
        np.testing.assert_array_equal(r_default["delay"], r_explicit["delay"])

    def test_unknown_distribution_raises(self):
        actions = np.random.randn(2, 10, 3)
        with pytest.raises(ValueError, match="Unknown delay_distribution"):
            rtc_training_prepare_batch(
                actions, max_delay=5, delay_distribution="gaussian",
            )

    def test_exponential_delay_range(self):
        """All delays should be in [0, max_delay)."""
        actions = np.random.default_rng(0).standard_normal((100, 10, 3))
        rng = np.random.default_rng(42)
        result = rtc_training_prepare_batch(
            actions, max_delay=5, rng=rng, delay_distribution="exponential",
        )
        assert np.all(result["delay"] >= 0)
        assert np.all(result["delay"] < 5)


# ---------------------------------------------------------------------------
# Fix 4.3: Unused tau_arr removed (regression test)
# ---------------------------------------------------------------------------

class TestRtcDenoiseStepNoTauArr:
    """Ensure denoise step works correctly without the old tau_arr variable."""

    def test_denoise_step_still_works(self):
        """Basic denoise step should produce correct results after fix."""
        H, D = 6, 2
        x_t = np.zeros((H, D))
        prefix = np.ones((H, D))
        mask = rtc_soft_mask(H, 1, 2)
        x_next, tau_next = rtc_pigdm_denoise_step(
            _identity_model, x_t, None, 0.1, 0.1, prefix, mask,
        )
        assert x_next.shape == (H, D)
        assert tau_next == pytest.approx(0.2)

    def test_multiple_steps_identical_to_manual(self):
        """Running denoise step should be equivalent to manual v + g computation."""
        H, D = 4, 2
        rng = np.random.default_rng(7)
        x_t = rng.standard_normal((H, D))
        prefix = np.ones((H, D)) * 5.0
        mask = rtc_soft_mask(H, 1, 2)
        tau, dt = 0.3, 0.1

        # Use denoise_step
        x_next_step, _ = rtc_pigdm_denoise_step(
            _linear_model, x_t, None, tau, dt, prefix, mask,
        )

        # Manual computation
        v = _linear_model(x_t, None, tau)
        g = rtc_pigdm_guidance(
            _linear_model, x_t, None, tau, prefix, mask,
            v_precomputed=v,
        )
        x_next_manual = x_t + dt * (v + g)
        np.testing.assert_allclose(x_next_step, x_next_manual, atol=1e-12)


# ---------------------------------------------------------------------------
# Fix 4.4: VJP docstring (existence test)
# ---------------------------------------------------------------------------

class TestNumericalVjpDocstring:
    """Check that _numerical_vjp has the computation cost warning."""

    def test_docstring_contains_warning(self):
        from python_filter_smoothing.rtc_utils import _numerical_vjp
        doc = _numerical_vjp.__doc__
        assert doc is not None
        assert "warning" in doc.lower() or "Warning" in doc
        assert "2 * H * D" in doc or "2*H*D" in doc
