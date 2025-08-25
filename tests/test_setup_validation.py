"""Validation tests to verify the testing infrastructure is properly configured."""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch
import networkx as nx
from pathlib import Path


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure."""
    
    @pytest.mark.unit
    def test_pytest_is_working(self):
        """Basic test to verify pytest is installed and working."""
        assert True, "Pytest should be working"
    
    @pytest.mark.unit
    def test_python_path_configured(self):
        """Test that the project root is in the Python path."""
        project_root = Path(__file__).parent.parent
        assert str(project_root) in sys.path or str(project_root.absolute()) in sys.path, \
            "Project root should be in Python path"
    
    @pytest.mark.unit
    def test_required_packages_imported(self):
        """Test that all required packages can be imported."""
        required_packages = [
            'numpy', 'pandas', 'torch', 'networkx', 
            'pytest', 'pytest_cov', 'pytest_mock'
        ]
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Failed to import required package: {package}")
    
    @pytest.mark.unit
    def test_project_modules_importable(self):
        """Test that project modules can be imported."""
        try:
            import baseline
            import evaluation
            import preprocessing
        except ImportError as e:
            pytest.fail(f"Failed to import project modules: {e}")
    
    @pytest.mark.unit
    def test_fixtures_available(self, temp_dir, sample_graph, sample_config):
        """Test that custom fixtures are available and working."""
        assert temp_dir.exists(), "temp_dir fixture should create a directory"
        assert isinstance(sample_graph, nx.Graph), "sample_graph should be a NetworkX graph"
        assert isinstance(sample_config, dict), "sample_config should be a dictionary"
        assert 'dataset' in sample_config, "sample_config should have expected keys"
    
    @pytest.mark.unit
    def test_numpy_operations(self):
        """Test basic numpy operations."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        assert arr.shape == (5,)
    
    @pytest.mark.unit
    def test_pandas_operations(self):
        """Test basic pandas operations."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        assert list(df.columns) == ['a', 'b']
    
    @pytest.mark.unit
    def test_torch_operations(self):
        """Test basic PyTorch operations."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.mean().item() == pytest.approx(2.0)
        assert tensor.shape == torch.Size([3])
    
    @pytest.mark.unit
    def test_networkx_operations(self):
        """Test basic NetworkX operations."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2
    
    @pytest.mark.integration
    def test_temp_dir_cleanup(self, temp_dir):
        """Test that temporary directories are created and will be cleaned up."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
    
    @pytest.mark.integration
    def test_mock_functionality(self, mocker):
        """Test that pytest-mock is working correctly."""
        mock_func = mocker.Mock(return_value=42)
        result = mock_func()
        assert result == 42
        mock_func.assert_called_once()
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works (can be skipped with -m 'not slow')."""
        import time
        start = time.time()
        time.sleep(0.1)
        elapsed = time.time() - start
        assert elapsed >= 0.1, "Slow test should take at least 0.1 seconds"
    
    @pytest.mark.unit
    def test_coverage_configured(self):
        """Test that coverage is properly configured."""
        import coverage
        assert coverage.__version__, "Coverage should be installed"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("value,expected", [
        (1, 1),
        (2, 4),
        (3, 9),
        (4, 16),
    ])
    def test_parametrize_working(self, value, expected):
        """Test that pytest parametrize is working."""
        assert value ** 2 == expected


class TestFixturesValidation:
    """Test class to validate all custom fixtures."""
    
    @pytest.mark.unit
    def test_sample_adjacency_matrix(self, sample_adjacency_matrix):
        """Test the sample adjacency matrix fixture."""
        assert isinstance(sample_adjacency_matrix, np.ndarray)
        assert sample_adjacency_matrix.shape == (5, 5)
        assert np.allclose(sample_adjacency_matrix, sample_adjacency_matrix.T), \
            "Adjacency matrix should be symmetric"
    
    @pytest.mark.unit
    def test_sample_edge_list(self, sample_edge_list):
        """Test the sample edge list fixture."""
        assert isinstance(sample_edge_list, pd.DataFrame)
        assert list(sample_edge_list.columns) == ['source', 'target', 'weight']
        assert len(sample_edge_list) == 6
    
    @pytest.mark.unit
    def test_sample_node_features(self, sample_node_features):
        """Test the sample node features fixture."""
        assert isinstance(sample_node_features, torch.Tensor)
        assert sample_node_features.shape == (10, 32)
    
    @pytest.mark.unit
    def test_sample_edge_index(self, sample_edge_index):
        """Test the sample edge index fixture."""
        assert isinstance(sample_edge_index, torch.Tensor)
        assert sample_edge_index.shape == (2, 6)
        assert sample_edge_index.dtype == torch.long
    
    @pytest.mark.unit
    def test_mock_config_file(self, mock_config_file):
        """Test the mock config file fixture."""
        assert mock_config_file.exists()
        assert mock_config_file.suffix == '.json'
        import json
        with open(mock_config_file) as f:
            config = json.load(f)
        assert 'dataset' in config
        assert 'method' in config
    
    @pytest.mark.unit
    def test_sample_time_series_graphs(self, sample_time_series_graphs):
        """Test the sample time series graphs fixture."""
        assert isinstance(sample_time_series_graphs, list)
        assert len(sample_time_series_graphs) == 5
        for g in sample_time_series_graphs:
            assert isinstance(g, nx.Graph)
    
    @pytest.mark.unit
    def test_sample_embedding(self, sample_embedding):
        """Test the sample embedding fixture."""
        assert isinstance(sample_embedding, np.ndarray)
        assert sample_embedding.shape == (10, 128)
    
    @pytest.mark.unit
    def test_mock_data_dir(self, mock_data_dir):
        """Test the mock data directory fixture."""
        assert mock_data_dir.exists()
        assert (mock_data_dir / "graphs").exists()
        assert (mock_data_dir / "embeddings").exists()
        assert (mock_data_dir / "results").exists()
    
    @pytest.mark.unit
    def test_sample_csv_data(self, sample_csv_data):
        """Test the sample CSV data fixture."""
        assert sample_csv_data.exists()
        df = pd.read_csv(sample_csv_data)
        assert 'timestamp' in df.columns
        assert 'source' in df.columns
        assert 'target' in df.columns
        assert 'weight' in df.columns
        assert len(df) == 100
    
    @pytest.mark.unit
    def test_device_fixture(self, device):
        """Test the device fixture."""
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda']
    
    @pytest.mark.unit
    def test_sample_batch_data(self, sample_batch_data):
        """Test the sample batch data fixture."""
        assert isinstance(sample_batch_data, dict)
        assert 'x' in sample_batch_data
        assert 'edge_index' in sample_batch_data
        assert 'batch' in sample_batch_data
        assert 'y' in sample_batch_data
    
    @pytest.mark.unit
    def test_mock_model_checkpoint(self, mock_model_checkpoint):
        """Test the mock model checkpoint fixture."""
        assert mock_model_checkpoint.exists()
        checkpoint = torch.load(mock_model_checkpoint)
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'metrics' in checkpoint