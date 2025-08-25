import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator
import pytest
import numpy as np
import pandas as pd
import networkx as nx
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_graph() -> nx.Graph:
    """Create a sample NetworkX graph for testing."""
    G = nx.Graph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 2), (1, 3), (4, 5), (5, 6)
    ])
    return G


@pytest.fixture
def sample_directed_graph() -> nx.DiGraph:
    """Create a sample directed NetworkX graph for testing."""
    G = nx.DiGraph()
    G.add_edges_from([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (0, 2), (1, 3), (4, 5), (5, 6)
    ])
    return G


@pytest.fixture
def sample_adjacency_matrix() -> np.ndarray:
    """Create a sample adjacency matrix."""
    return np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ])


@pytest.fixture
def sample_edge_list() -> pd.DataFrame:
    """Create a sample edge list DataFrame."""
    edges = [
        (0, 1, 1.0),
        (1, 2, 0.8),
        (2, 3, 0.9),
        (3, 0, 0.7),
        (0, 2, 0.6),
        (1, 3, 0.5)
    ]
    return pd.DataFrame(edges, columns=['source', 'target', 'weight'])


@pytest.fixture
def sample_node_features() -> torch.Tensor:
    """Create sample node features tensor."""
    return torch.randn(10, 32)


@pytest.fixture
def sample_edge_index() -> torch.Tensor:
    """Create sample edge index tensor for PyTorch Geometric."""
    edge_index = torch.tensor([
        [0, 1, 2, 3, 0, 1],
        [1, 2, 3, 0, 2, 3]
    ], dtype=torch.long)
    return edge_index


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create a sample configuration dictionary."""
    return {
        "dataset": "test_dataset",
        "method": "gcn",
        "epochs": 10,
        "learning_rate": 0.01,
        "hidden_dim": 64,
        "batch_size": 32,
        "dropout": 0.5,
        "seed": 42,
        "device": "cpu"
    }


@pytest.fixture
def mock_config_file(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a mock configuration file."""
    config_path = temp_dir / "test_config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)
    return config_path


@pytest.fixture
def sample_time_series_graphs() -> list:
    """Create a list of graphs representing time series data."""
    graphs = []
    for t in range(5):
        G = nx.Graph()
        G.add_edges_from([
            (0, 1), (1, 2), (2, 3),
            (3, 0 if t % 2 == 0 else 4),
            (t, (t + 1) % 5)
        ])
        graphs.append(G)
    return graphs


@pytest.fixture
def sample_embedding() -> np.ndarray:
    """Create a sample node embedding matrix."""
    np.random.seed(42)
    return np.random.randn(10, 128)


@pytest.fixture
def mock_data_dir(temp_dir: Path) -> Path:
    """Create a mock data directory structure."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    (data_dir / "graphs").mkdir(exist_ok=True)
    (data_dir / "embeddings").mkdir(exist_ok=True)
    (data_dir / "results").mkdir(exist_ok=True)
    
    return data_dir


@pytest.fixture
def sample_csv_data(temp_dir: Path) -> Path:
    """Create a sample CSV file with graph data."""
    csv_path = temp_dir / "sample_data.csv"
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='h'),
        'source': np.random.randint(0, 10, 100),
        'target': np.random.randint(0, 10, 100),
        'weight': np.random.random(100)
    })
    data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch_data() -> Dict[str, torch.Tensor]:
    """Create sample batch data for model testing."""
    batch_size = 16
    num_nodes = 20
    num_features = 32
    
    return {
        'x': torch.randn(batch_size, num_nodes, num_features),
        'edge_index': torch.randint(0, num_nodes, (2, 50)),
        'batch': torch.repeat_interleave(torch.arange(batch_size), num_nodes // batch_size),
        'y': torch.randint(0, 2, (batch_size,))
    }


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path) -> Path:
    """Create a mock model checkpoint file."""
    checkpoint_path = temp_dir / "model_checkpoint.pt"
    checkpoint = {
        'epoch': 100,
        'model_state_dict': {'dummy': torch.randn(10, 10)},
        'optimizer_state_dict': {},
        'loss': 0.123,
        'metrics': {'accuracy': 0.95, 'f1': 0.93}
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def capture_stdout(monkeypatch):
    """Fixture to capture stdout output."""
    import io
    buffer = io.StringIO()
    
    def _capture():
        monkeypatch.setattr(sys, 'stdout', buffer)
        return buffer
    
    return _capture


@pytest.fixture(params=['gcn', 'gat', 'sage', 'gin'])
def model_type(request):
    """Parametrized fixture for different model types."""
    return request.param


@pytest.fixture(params=[16, 32, 64])
def hidden_dim(request):
    """Parametrized fixture for different hidden dimensions."""
    return request.param


@pytest.fixture
def cleanup_gpu():
    """Cleanup GPU memory after tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()