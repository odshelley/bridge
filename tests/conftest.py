"""pytest configuration."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (e.g., tests that download datasets)",
    )


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is provided."""
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def afhq_dir(tmp_path):
    """Create a fake AFHQ directory tree with 4 tiny images per class per split."""
    from PIL import Image

    for split in ("train", "val"):
        for cls in ("cat", "dog", "wild"):
            d = tmp_path / "afhq" / split / cls
            d.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (32, 32), color=(i * 20, 100, 150)).save(d / f"{cls}_{i}.png")
    return tmp_path
