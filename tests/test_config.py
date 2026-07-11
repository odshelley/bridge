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


class TestTransportConfig:
    """Tests for transport-mode data configuration."""

    def test_transport_fields_default_to_none(self) -> None:
        config = DataConfig()
        assert config.classes is None
        assert config.source_dataset is None
        assert config.source_classes is None

    def test_transport_fields_set(self) -> None:
        config = DataConfig(
            dataset="afhq",
            classes=["dog"],
            source_dataset="afhq",
            source_classes=["cat"],
        )
        assert config.classes == ["dog"]
        assert config.source_dataset == "afhq"
        assert config.source_classes == ["cat"]

    def test_source_classes_without_source_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="source_classes requires source_dataset"):
            DataConfig(source_classes=["cat"])

    def test_ddpm_with_source_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="requires method='bridge'"):
            ExperimentConfig(method="ddpm", data=DataConfig(source_dataset="mnist"))

    def test_poisson_bridge_with_source_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="requires method='bridge'"):
            ExperimentConfig(method="poisson_bridge", data=DataConfig(source_dataset="mnist"))

    def test_yaml_round_trip_with_transport_fields(self, tmp_path) -> None:
        config = ExperimentConfig(
            data=DataConfig(
                dataset="afhq",
                classes=["dog"],
                source_dataset="afhq",
                source_classes=["cat"],
            )
        )
        path = tmp_path / "config.yaml"
        config.to_yaml(path)
        loaded = ExperimentConfig.from_yaml(path)
        assert loaded.data.classes == ["dog"]
        assert loaded.data.source_dataset == "afhq"
        assert loaded.data.source_classes == ["cat"]


class TestTransportYamlConfigs:
    """The shipped transport configs must load and be transport-mode."""

    @pytest.mark.parametrize(
        "name",
        [
            "cifar_cat2dog.yaml",
            "cifar_cat2dog_smoke.yaml",
            "afhq_cat2dog_64.yaml",
            "afhq_cat2dog_smoke.yaml",
        ],
    )
    def test_config_loads_and_is_transport(self, name) -> None:
        from pathlib import Path as _Path

        path = _Path(__file__).parent.parent / "configs" / name
        config = ExperimentConfig.from_yaml(path)
        assert config.method == "bridge"
        assert config.data.source_dataset is not None
        assert config.data.classes == ["dog"]
        assert config.data.source_classes == ["cat"]


class TestNewDatasetConfigs:
    """The generation configs added for CIFAR-10/AFHQ ship loadable."""

    CONFIGS_DIR = Path(__file__).parent.parent / "configs"

    @pytest.mark.parametrize(
        "name,method,dataset,raw",
        [
            ("afhq_64", "bridge", "afhq", False),
            ("afhq_poisson_64", "poisson_bridge", "afhq", True),
            ("cifar10_poisson", "poisson_bridge", "cifar10", True),
            ("cifar10_poisson_smoke", "poisson_bridge", "cifar10", True),
        ],
    )
    def test_config_loads(self, name: str, method: str, dataset: str, raw: bool) -> None:
        config = ExperimentConfig.from_yaml(self.CONFIGS_DIR / f"{name}.yaml")
        assert config.method == method
        assert config.data.dataset == dataset
        assert config.data.raw_pixels is raw
        assert config.data.source_dataset is None  # generation, not transport


class TestRawPixelsValidation:
    """poisson_bridge and raw_pixels must always agree."""

    def test_poisson_bridge_without_raw_pixels_raises(self) -> None:
        with pytest.raises(
            ValueError, match="method='poisson_bridge' requires data.raw_pixels=true"
        ):
            ExperimentConfig(method="poisson_bridge", data=DataConfig(raw_pixels=False))

    def test_raw_pixels_with_non_poisson_method_raises(self) -> None:
        with pytest.raises(
            ValueError, match="data.raw_pixels=true requires method='poisson_bridge'"
        ):
            ExperimentConfig(method="bridge", data=DataConfig(raw_pixels=True))

    def test_poisson_bridge_with_raw_pixels_is_valid(self) -> None:
        config = ExperimentConfig(method="poisson_bridge", data=DataConfig(raw_pixels=True))
        assert config.method == "poisson_bridge"
        assert config.data.raw_pixels is True


class TestUnknownTopLevelKeys:
    """from_yaml should reject configs with unrecognised top-level keys."""

    def test_unknown_key_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("name: test\nmethod: bridge\nbogus_key: 123\n")
        with pytest.raises(ValueError, match=r"Unknown top-level config keys.*bogus_key"):
            ExperimentConfig.from_yaml(path)

    def test_known_keys_do_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("name: test\nmethod: bridge\n")
        ExperimentConfig.from_yaml(path)  # should not raise

    def test_empty_file_raises_value_error(self, tmp_path: Path) -> None:
        """An empty YAML file parses to None; from_yaml must reject it clearly."""
        path = tmp_path / "config.yaml"
        path.write_text("")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            ExperimentConfig.from_yaml(path)


class TestAllShippedConfigsLoad:
    """Every YAML config shipped in configs/ must load without error."""

    CONFIGS_DIR = Path(__file__).parent.parent / "configs"

    @pytest.mark.parametrize(
        "path",
        sorted(CONFIGS_DIR.glob("*.yaml")),
        ids=lambda p: p.name,
    )
    def test_config_loads(self, path: Path) -> None:
        ExperimentConfig.from_yaml(path)
