"""Tests for scripts/evaluate_fid.py's sample generation (transport awareness)."""

import importlib.util
from pathlib import Path

import pytest
import torch

from bridge_diffusion.config import (
    BridgeConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
)
from bridge_diffusion.models import BridgeDiffusion, DiffusersUNetWrapper
from bridge_diffusion.sampling import Sampler

_SCRIPT = Path(__file__).parent.parent / "scripts" / "evaluate_fid.py"


@pytest.fixture(scope="module")
def evaluate_fid():
    spec = importlib.util.spec_from_file_location("evaluate_fid", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_model(image_size: int = 16) -> BridgeDiffusion:
    model_config = ModelConfig(
        in_channels=3,
        out_channels=3,
        sample_size=image_size,
        block_out_channels=(32, 64),
        layers_per_block=1,
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
    )
    network = DiffusersUNetWrapper(model_config)
    return BridgeDiffusion(network, BridgeConfig())


def _transport_config(afhq_dir) -> ExperimentConfig:
    return ExperimentConfig(
        name="eval_fid_test",
        output_dir=Path("outputs/eval_fid_test"),
        method="bridge",
        model=ModelConfig(
            in_channels=3,
            out_channels=3,
            sample_size=16,
            block_out_channels=(32, 64),
            layers_per_block=1,
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
        ),
        data=DataConfig(
            dataset="afhq",
            classes=["dog"],
            source_dataset="afhq",
            source_classes=["cat"],
            data_dir=afhq_dir,
            image_size=16,
            num_workers=0,
        ),
    )


class TestGenerateSamplesTransport:
    def test_transport_checkpoint_translates_source_images(
        self, evaluate_fid, afhq_dir, monkeypatch
    ) -> None:
        """Transport configs must feed real source val images into the sampler
        as x0 instead of sampling from Gaussian noise."""
        captured = {}
        original = Sampler.sample_batch

        def spy(self, total_samples, shape, batch_size=64, num_steps=None, x0=None):
            captured["x0"] = x0
            return original(
                self, total_samples, shape, batch_size=batch_size, num_steps=num_steps, x0=x0
            )

        monkeypatch.setattr(Sampler, "sample_batch", spy)

        config = _transport_config(afhq_dir)
        samples = evaluate_fid.generate_samples(
            model=_tiny_model(),
            config=config,
            num_samples=4,
            num_steps=2,
            device=torch.device("cpu"),
            batch_size=2,
        )

        assert samples.shape == (4, 3, 16, 16)
        assert torch.isfinite(samples).all()
        assert captured["x0"] is not None, "transport run sampled from noise, not sources"
        assert captured["x0"].shape == (4, 3, 16, 16)

    def test_transport_tiles_sources_when_val_split_is_small(
        self, evaluate_fid, afhq_dir, monkeypatch
    ) -> None:
        """The AFHQ val split has few images; requesting more samples than
        sources must tile them rather than truncate the sample count."""
        captured = {}
        monkeypatch.setattr(
            Sampler,
            "sample_batch",
            lambda self, total_samples, shape, batch_size=64, num_steps=None, x0=None: (
                captured.update(x0=x0),
                torch.zeros(total_samples, *shape),
            )[1],
        )

        config = _transport_config(afhq_dir)  # fixture has 4 val cats
        evaluate_fid.generate_samples(
            model=_tiny_model(),
            config=config,
            num_samples=6,
            num_steps=2,
            device=torch.device("cpu"),
            batch_size=6,
        )

        assert captured["x0"].shape[0] == 6
        # Tiled: images 4 and 5 repeat images 0 and 1.
        assert torch.equal(captured["x0"][4], captured["x0"][0])
        assert torch.equal(captured["x0"][5], captured["x0"][1])

    def test_transport_with_ode_raises(self, evaluate_fid, afhq_dir) -> None:
        config = _transport_config(afhq_dir)
        with pytest.raises(ValueError, match="--ode"):
            evaluate_fid.generate_samples(
                model=_tiny_model(),
                config=config,
                num_samples=2,
                num_steps=2,
                device=torch.device("cpu"),
                use_ode=True,
            )

    def test_generation_config_still_samples_from_noise(self, evaluate_fid, monkeypatch) -> None:
        captured = {}
        monkeypatch.setattr(
            Sampler,
            "sample_batch",
            lambda self, total_samples, shape, batch_size=64, num_steps=None, x0=None: (
                captured.update(x0=x0),
                torch.zeros(total_samples, *shape),
            )[1],
        )

        generation_config = ExperimentConfig(
            name="eval_fid_gen_test",
            output_dir=Path("outputs/eval_fid_gen_test"),
            method="bridge",
            model=ModelConfig(
                in_channels=3,
                out_channels=3,
                sample_size=16,
                block_out_channels=(32, 64),
                layers_per_block=1,
                down_block_types=("DownBlock2D", "DownBlock2D"),
                up_block_types=("UpBlock2D", "UpBlock2D"),
            ),
            data=DataConfig(dataset="cifar10", image_size=16, num_workers=0),
        )
        evaluate_fid.generate_samples(
            model=_tiny_model(),
            config=generation_config,
            num_samples=2,
            num_steps=2,
            device=torch.device("cpu"),
        )
        assert captured["x0"] is None
