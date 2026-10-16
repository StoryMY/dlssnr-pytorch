"""Extraction of the library's second network. The tests that read the library need DLSSNR_TEST_DLL (the 310.9.16.0
nvngx_dlssnr.dll, or a zip holding it) and skip without it."""
import os
from pathlib import Path

import pytest
import torch

from dlssnr import extract
from dlssnr.weights import Weights

LIBRARY = os.environ.get("DLSSNR_TEST_DLL")
needs_library = pytest.mark.skipif(not (LIBRARY and Path(LIBRARY).exists()), reason="needs DLSSNR_TEST_DLL")


def test_the_variants_name_the_two_resources():
    assert extract.RESOURCES == {"ht": "WEIGHTS_HT", "small": "WEIGHTS_CC_SMALL_FP8"}


def test_an_unknown_variant_is_refused(tmp_path):
    with pytest.raises(extract.ExtractError, match="unknown network"):
        extract.extract(str(tmp_path / "x.dll"), str(tmp_path / "w.safetensors"), "huge")


@needs_library
def test_the_small_network_is_extracted(tmp_path):
    weights = extract.extract(LIBRARY, str(tmp_path / "small.safetensors"), "small")
    assert weights.metadata["network"] == "small" and weights.metadata["records"] == "77"
    assert len(weights.tensors) == 302
    blocks = {int(key.split(".")[0][5:]) for key in weights.tensors}
    assert blocks == set(range(31))
    assert Weights.load(str(tmp_path / "small.safetensors")).metadata["network"] == "small"


@needs_library
def test_the_original_network_of_the_new_library_is_the_one_we_had(tmp_path):
    """310.9.16.0's WEIGHTS_HT is byte for byte the 310.8.0 weights this project was built on."""
    ours = Path("weights.safetensors")
    if not ours.exists():
        pytest.skip("needs weights.safetensors")
    new = extract.extract(LIBRARY, str(tmp_path / "ht.safetensors"), "ht")
    old = Weights.load(str(ours))
    assert new.tensors.keys() == old.tensors.keys()
    assert all(torch.equal(new.tensors[k], old.tensors[k]) for k in old.tensors)


SMALL_WEIGHTS = Path("weights-small.safetensors")


@pytest.mark.skipif(not SMALL_WEIGHTS.exists(), reason="needs weights-small.safetensors")
def test_the_small_network_runs_and_follows_its_weights_file():
    import numpy as np

    from dlssnr import int4
    from dlssnr.benchmark import synthetic_frame
    from dlssnr.io import Controls
    from dlssnr.model import NRNet

    weights = Weights.load(str(SMALL_WEIGHTS))
    frame = synthetic_frame(192, 128)
    net = NRNet(weights, 192, 128)
    assert net.network == "small" and len(net.plan) == 76
    out = net(frame, Controls())
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - frame).mean() > 0.5                   # the network does something to the frame
    class Tables:
        directory = "unused"
    int4.tables = Tables()
    try:
        with pytest.raises(NotImplementedError, match="calibrated on the original network"):
            NRNet(weights, 192, 128)
    finally:
        int4.tables = None
