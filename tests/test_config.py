"""Tests for configuration module."""

import tempfile
from pathlib import Path

import pytest

from bridge_diffusion.config import (
    BridgeConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    SamplingConfig,
    TrainingConfig,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values (diffusers-style)."""
        config = ModelConfig()
        assert config.sample_size == 32
        assert config.block_out_channels == (128, 256, 256, 256)
        assert config.layers_per_block == 2

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ModelConfig(
            sample_size=64,
            block_out_channels=(64, 128, 256),
            attention_head_dim=16,
        )
        assert config.sample_size == 64
        assert config.block_out_channels == (64, 128, 256)
        assert config.attention_head_dim == 16


class TestBridgeConfig:
    """Tests for BridgeConfig."""

    def test_default_values(self) -> None:
        """Test default bridge parameters."""
        config = BridgeConfig()
        assert config.T == 0.1
        assert config.eps == 1e-7

    def test_custom_terminal_time(self) -> None:
        """Test custom terminal time."""
        config = BridgeConfig(T=1.0)
        assert config.T == 1.0


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self) -> None:
        """Test default training parameters."""
        config = TrainingConfig()
        assert config.batch_size == 128
        assert config.num_steps == 40000
        assert config.learning_rate == 1e-4

    def test_custom_values(self) -> None:
        """Test custom training configuration."""
        config = TrainingConfig(batch_size=64, learning_rate=2e-4)
        assert config.batch_size == 64
        assert config.learning_rate == 2e-4


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self) -> None:
        """Test default data configuration."""
        config = DataConfig()
        assert config.dataset == "mnist"
        assert config.image_size == 32

    def test_custom_values(self) -> None:
        """Test custom data configuration."""
        config = DataConfig(dataset="cifar10", image_size=64)
        assert config.dataset == "cifar10"
        assert config.image_size == 64


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_yaml_round_trip(self) -> None:
        """Test saving and loading from YAML."""
        config = ExperimentConfig(
            name="test_experiment",
            model=ModelConfig(sample_size=64),
            training=TrainingConfig(batch_size=64),
            bridge=BridgeConfig(T=0.5),
            sampling=SamplingConfig(num_steps=50),
            data=DataConfig(dataset="mnist"),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"

            # Save
            config.to_yaml(path)

            # Load
            loaded = ExperimentConfig.from_yaml(path)

            # Verify
            assert loaded.name == "test_experiment"
            assert loaded.model.sample_size == 64
            assert loaded.training.batch_size == 64
            assert loaded.bridge.T == 0.5
            assert loaded.data.dataset == "mnist"

    def test_nested_configs_preserved(self) -> None:
        """Test that nested configuration objects are preserved."""
        config = ExperimentConfig(
            model=ModelConfig(
                sample_size=64,
                block_out_channels=(64, 128, 256, 512),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            config.to_yaml(path)
            loaded = ExperimentConfig.from_yaml(path)

            # Note: YAML may convert tuple to list, so check values
            assert list(loaded.model.block_out_channels) == [64, 128, 256, 512]
