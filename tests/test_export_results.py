import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.export_results import export_results
from src.orbital_parameters import orbital_parameters


def test_export_results_basic(tmp_path):
    """Test export with non-uniform time vector."""
    # Define a non-uniform time vector
    t_vector = np.array([0.0, 0.3, 1.7, 40.0, 40.5])
    # Create a simple y_matrix (e.g., constant state)
    y_matrix = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (len(t_vector), 1))
    
    orbit_params = orbital_parameters(verbose=False)
    output_dir = tmp_path / "test_export"
    
    # Call export_results
    export_results(t_vector, y_matrix, orbit_params, output_dir)
    
    # Check that the four expected files exist
    expected_files = [
        "orbit_cartesian.csv",
        "orbit_elements.csv",
        "orbit_energy.csv",
        "orbit_angular_momentum.csv",
    ]
    for fname in expected_files:
        fpath = output_dir / fname
        assert fpath.exists(), f"Missing file: {fpath}"
        
        # Check that the time_s column matches the input t_vector
        df = pd.read_csv(fpath)
        np.testing.assert_array_equal(df["time_s"].values, t_vector)
        
        # Check that the number of rows matches
        assert len(df) == len(t_vector)


def test_export_results_errors(tmp_path):
    """Test that export_results raises appropriate errors."""
    orbit_params = orbital_parameters(verbose=False)
    output_dir = tmp_path / "test_export"
    output_dir.mkdir()
    
    # Test length mismatch
    t_vector = np.array([0.0, 1.0, 2.0])
    y_matrix = np.zeros((2, 6))  # Only 2 rows
    with pytest.raises(ValueError, match="t_vector and y_matrix must contain the same number of samples"):
        export_results(t_vector, y_matrix, orbit_params, output_dir)
    
    # Test non-increasing times
    t_vector = np.array([0.0, 2.0, 1.0])  # Not increasing
    y_matrix = np.zeros((3, 6))
    with pytest.raises(ValueError, match="t_vector must be strictly increasing"):
        export_results(t_vector, y_matrix, orbit_params, output_dir)
    
    # Test duplicate times
    t_vector = np.array([0.0, 1.0, 1.0])
    y_matrix = np.zeros((3, 6))
    with pytest.raises(ValueError, match="t_vector must be strictly increasing"):
        export_results(t_vector, y_matrix, orbit_params, output_dir)
    
    # Test NaN times
    t_vector = np.array([0.0, 1.0, np.nan])
    y_matrix = np.zeros((3, 6))
    with pytest.raises(ValueError, match="t_vector contains NaN values"):
        export_results(t_vector, y_matrix, orbit_params, output_dir)
    
    # Test too few points
    t_vector = np.array([0.0])
    y_matrix = np.zeros((1, 6))
    with pytest.raises(ValueError, match="t_vector must contain at least two time points"):
        export_results(t_vector, y_matrix, orbit_params, output_dir)
