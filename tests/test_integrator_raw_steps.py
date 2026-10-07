import numpy as np
import pytest

from src.pd853_integrate import pd853_integrate
from src.symplectic_integrate import symplectic_integrate
from src.gravity_ode import gravity_ode


def test_pd853_raw_steps():
    """Test that PD853 returns raw steps and respects step size limits."""
    # Short horizon: 600 seconds
    t_final = 600.0
    t_eval = np.array([0.0, t_final])  # endpoints-only
    
    # Initial state: circular orbit at 500 km altitude
    # Use simple initial condition for testing
    y0 = np.array([6371000.0 + 500000.0, 0.0, 0.0, 0.0, 7600.0, 0.0])  # [x, y, z, vx, vy, vz]
    
    options = {
        "RelTol": 1e-12,
        "AbsTol": 1e-14,
        "MaxStep": 60.0,
        "InternalStep": 1e-3,
    }
    
    t_raw, y_raw, stats = pd853_integrate(gravity_ode, t_eval, y0, options)
    
    # Check that t_raw is strictly increasing
    assert np.all(np.diff(t_raw) > 0), "Time steps must be strictly increasing"
    
    # Check that it starts at t_eval[0] and ends at t_eval[-1] (within tolerance)
    assert abs(t_raw[0] - t_eval[0]) < 1e-12, "Initial time mismatch"
    assert abs(t_raw[-1] - t_eval[-1]) < 1e-6, "Final time mismatch"  # Allow small integration error
    
    # Check that accepted_steps matches the number of steps
    assert stats.accepted_steps == len(t_raw) - 1, "accepted_steps should equal number of steps"
    
    # Check step size statistics
    dt = np.diff(t_raw)
    assert np.max(dt) <= options["MaxStep"] + 1e-12, f"Max step {np.max(dt)} exceeds MaxStep {options['MaxStep']}"
    
    # Check that steps are non-uniform (ratio of max to min step > 1.1)
    # For a short run with tight tolerances, we expect some variation
    if len(dt) > 1:
        step_ratio = np.max(dt) / np.min(dt)
        assert step_ratio > 1.1, f"Step sizes are too uniform: ratio={step_ratio}"


def test_symplectic_raw_steps():
    """Test that symplectic integrator returns raw steps with fixed step size."""
    # Short horizon: 600 seconds
    t_final = 600.0
    t_eval = np.array([0.0, t_final])  # endpoints-only
    
    # Initial state: circular orbit at 500 km altitude
    y0 = np.array([6371000.0 + 500000.0, 0.0, 0.0, 0.0, 7600.0, 0.0])  # [x, y, z, vx, vy, vz]
    
    options = {
        "SymplecticStep": 1.0,  # 1 second steps
        "GaussLegendreTol": 1e-10,
        "GaussLegendreMaxFEV": 200,
        "GaussLegendreXtol": 1e-14,
    }
    
    t_raw, y_raw, stats = symplectic_integrate(gravity_ode, t_eval, y0, options)
    
    # Check that t_raw is strictly increasing
    assert np.all(np.diff(t_raw) > 0), "Time steps must be strictly increasing"
    
    # Check that it starts at t_eval[0] and ends at t_eval[-1] (within tolerance)
    assert abs(t_raw[0] - t_eval[0]) < 1e-12, "Initial time mismatch"
    assert abs(t_raw[-1] - t_eval[-1]) < 1e-6, "Final time mismatch"  # Allow small integration error
    
    # Check that accepted_steps matches the number of steps
    assert stats.accepted_steps == len(t_raw) - 1, "accepted_steps should equal number of steps"
    
    # Check step sizes: should be exactly SymplecticStep (except possibly the last step)
    dt = np.diff(t_raw)
    expected_step = options["SymplecticStep"]
    
    # All steps should be equal to expected_step, except the last step may be shorter if t_final is not a multiple
    # We'll allow the last step to be different
    if len(dt) > 1:
        # Check all but the last step
        np.testing.assert_allclose(dt[:-1], expected_step, rtol=1e-12, atol=1e-12)
        # The last step should be <= expected_step
        assert dt[-1] <= expected_step + 1e-12, f"Last step {dt[-1]} exceeds expected step {expected_step}"
    else:
        # Only one step: should be <= expected_step
        assert dt[0] <= expected_step + 1e-12, f"Single step {dt[0]} exceeds expected step {expected_step}"