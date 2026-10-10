"""The int4 mixed network's decoder, quantiser and transform. The tests marked "tables" need DLSSNR-AMD's checkout."""
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from dlssnr import int4

AMD = Path(__file__).resolve().parents[2] / "DLSSNR-AMD"
TABLES = AMD / "linux" / "data" / "int4" / "rdna4"
INSTALLED = AMD / "build" / "test" / "inst-int4" / "dlssnr-amd" / "dlssnr-int4.bin"


def hadamard_matrix(n):
    h = np.array([[1.0]])
    while len(h) < n:
        h = np.block([[h, h], [h, -h]])
    return h


@pytest.mark.parametrize("block", [1, 16, 64])
def test_fwht_is_the_blockwise_hadamard_matrix(block):
    x = torch.randn(3, 128)
    expected = np.concatenate([x.numpy()[:, i:i + block] @ hadamard_matrix(block) for i in range(0, 128, block)], axis=1)
    assert np.allclose(int4.fwht(x, block).numpy(), expected, atol=1e-4)


def matrix(block, k4=64, gain=1.0, zero=8.0):
    aq = np.stack([np.full(k4, gain, np.float32), np.full(k4, zero, np.float32)], axis=1)
    return int4.Int4Matrix(np.zeros((2, k4), np.int8), np.ones(2, np.float32), np.zeros(2, np.float32), aq, block)


def test_quantise_rounds_to_even_and_clamps():
    m = matrix(1)
    x = torch.tensor([[0.5, 1.5, 2.5, -0.5, 100.0, -100.0] + [0.0] * 58])
    code = m.quantise(x)[0, :6]
    assert code.tolist() == [0.0, 2.0, 2.0, 0.0, 7.0, -8.0]                  # rint(x) + 8 - 8, ties to even, clamp to 0..15
    assert (m.quantise(x)[0, 6:] == 0).all()                                 # the zero point 8 is the signed zero


def test_quantise_rotates_before_quantising():
    m = matrix(16, gain=2.0, zero=8.0)
    x = torch.zeros(1, 64)
    x[0, 0] = 1.0                                                            # H16 of a unit vector: 1 / 4 on all 16 values
    expected = torch.round(torch.tensor(1.0 * 0.25 * 2.0)) + 0              # rint(0.5) = 0 with the zero point 8 removed
    assert m.quantise(x)[0, :16].tolist() == [expected.item()] * 16


def write_tables(directory, n, k4, block, clip, correction, escapes, mu_x, nu, shift=None):
    codes = np.zeros(((n * k4 + 3) // 4) * 4, np.uint8)
    codes[: n * k4] = np.where(correction.reshape(-1) == 1, 1, np.where(correction.reshape(-1) == -1, 2, 0))
    packed = (codes.reshape(-1, 4) << np.array([0, 2, 4, 6], np.uint8)).sum(axis=1).astype(np.uint8)
    fix = struct.pack("<6I", 0x58463449, n, k4, block, len(escapes), 0) + clip.astype(np.uint8).tobytes() + packed.tobytes()
    for index, delta in escapes:
        fix += struct.pack("<Ii", index, delta)
    mu = np.concatenate([mu_x, nu] + ([shift] if shift is not None else [])).astype(np.float32).tobytes()
    (directory / "m.fix.bin").write_bytes(fix)
    (directory / "m.mu.bin").write_bytes(mu)
    return directory / "m"


@pytest.mark.parametrize("block", [1, 16])
def test_decode_follows_the_documented_formulas(tmp_path, block):
    rng = np.random.default_rng(0)
    n, k4 = 6, 32
    codes = rng.integers(0, 0x7E, (n, k4)).astype(np.uint8)                  # positive and small e4m3 values
    aq = np.stack([rng.uniform(1, 4, k4), rng.integers(6, 10, k4)], 1).astype(np.float32)
    clip = rng.integers(0, 9, n)
    correction = rng.integers(-1, 2, (n, k4))
    mu_x, nu = rng.normal(size=k4), rng.normal(size=k4)
    shift = rng.normal(size=n)
    escapes = [(5, 2), (k4 + 7, -3)]
    prefix = write_tables(tmp_path, n, k4, block, clip, correction, escapes, mu_x, nu, shift)
    q, s, c = int4.decode(prefix, codes, aq, block)

    weights = int4.E4M3[codes]
    rotated = weights @ np.kron(np.eye(k4 // block), hadamard_matrix(block) / np.sqrt(block)) if block > 1 else weights
    rotated = rotated * (np.float32(1.0) / aq[:, 0]).astype(np.float64)[None]       # the step is float(1.0f / aq)
    scale = (np.abs(rotated).max(axis=1) * np.array(int4.CLIPS)[clip] / 7).astype(np.float32)
    expected = np.clip(np.rint(rotated / scale[:, None].astype(np.float64)), -8, 7) + correction
    for index, delta in escapes:
        row, col = divmod(index, k4)
        expected[row, col] = np.clip(np.rint(rotated[row, col] / float(scale[row])), -8, 7) + delta
    assert np.array_equal(q, expected.astype(np.int8)) and np.array_equal(s, scale)
    constant = weights @ mu_x.astype(np.float32).astype(np.float64) - scale * (expected @ nu.astype(np.float32).astype(np.float64)) + shift.astype(np.float32).astype(np.float64)
    assert np.allclose(c, constant, rtol=1e-6, atol=1e-6)


def test_decode_rejects_tables_of_another_shape(tmp_path):
    n, k4 = 2, 16
    prefix = write_tables(tmp_path, n, k4, 1, np.zeros(n), np.zeros((n, k4), int), [], np.zeros(k4), np.zeros(k4))
    with pytest.raises(ValueError):
        int4.decode(prefix, np.zeros((n, k4 + 16), np.uint8), np.ones((k4 + 16, 2), np.float32), 1)


@pytest.mark.skipif(not (INSTALLED.exists() and TABLES.exists() and Path("weights.safetensors").exists()),
                    reason="needs DLSSNR-AMD's tables, the weights file its installer wrote, and our weights")
def test_decoded_weights_are_those_of_the_installer(tmp_path):
    """tables: every one of the 83 matrices is rebuilt bit for bit (q, s and c) as DLSSNR-AMD's installer wrote it."""
    from dlssnr.weights import Weights
    data = INSTALLED.read_bytes()
    count = struct.unpack_from("<I", data, 8)[0]
    offset, entries = 20, {}
    for _ in range(count):
        (length,) = struct.unpack_from("<I", data, offset)
        name = data[offset + 4: offset + 4 + length].decode()
        offset += 4 + length
        n, k4 = struct.unpack_from("<II", data, offset)
        offset += 16
        packed = np.frombuffer(data, np.uint8, n * k4 // 2, offset)
        offset += n * k4 // 2
        sc = np.frombuffer(data, np.float32, 2 * n, offset)
        offset += 8 * n
        q = np.stack([(packed << 4).astype(np.int8) >> 4, packed.astype(np.int8) >> 4], 1).reshape(n, k4)
        entries[name] = (q, sc[:n], sc[n:])
    assert len(entries) == 83
    int4.enable(TABLES)
    weights = Weights.load("weights.safetensors")
    try:
        for name, (q, s, c) in entries.items():
            matrix = site_matrix(weights, name)
            assert np.array_equal(matrix.q.numpy().astype(np.int8), q), name
            assert np.array_equal(matrix.scale.numpy(), s) and np.array_equal(matrix.constant.numpy(), c), name
    finally:
        int4.disable()


def site_matrix(weights, name):
    """The decoded matrix the layers would hold for an installer entry name (``b15l0.ex``, ``b32.ex``, ``b32``...)."""
    import re
    block = int(re.match(r"b(\d+)", name).group(1))
    if re.match(r"b\d+l0\.", name):
        return int4.swin_matrix(weights, block, name.split(".")[1])
    if name.endswith(".attn"):
        return int4.attention_matrix(weights, block, 2)
    if name.endswith(".qkv"):
        return int4.vit_matrix(weights, block, "qkv")
    if name.endswith(".ex"):
        return int4.vit_ffn(weights, block).expand
    return int4.vit_ffn(weights, block).contract


def test_hard_swish_is_a_gated_identity():
    x = torch.tensor([-10.0, -2.74, 0.0, 2.74, 10.0, 1.0])
    a, b = 0.182412, 0.500465
    y = int4.hard_swish(x, a, b)
    gate = (x * np.float32(a) + np.float32(b)).clamp(0, 1)
    assert torch.allclose(y, x * gate, atol=1e-6)
    assert y[0] == 0 and y[4] == 10.0                  # fully closed below -B/A, fully open above (1 - B)/A, no scale


@pytest.mark.skipif(not TABLES.exists(), reason="needs DLSSNR-AMD's tables")
def test_swin_blocks_take_what_the_tables_name():
    """tables: hard-swish and fold per block as settings.txt lists them; int4 only where the tables are."""
    int4.enable(TABLES)
    try:
        class Missing:
            def raw(self, *args):
                raise AssertionError("no weights needed for a block without int4 tables")
        site = lambda block, channels, heads: int4.swin_site(Missing(), block, 0, channels, heads)
        assert site(3, 32, 1) is None                                # exact SiLU: not in NR_HS_FOLD_BLOCKS
        assert site(10, 128, 4) is None
        for block, channels, heads, a in ((0, 32, 1, 0.187367), (70, 32, 1, 0.187367), (4, 32, 1, 0.187367),
                                          (6, 64, 2, 0.182412), (65, 64, 2, 0.182412)):
            found = site(block, channels, heads)
            assert found.hs[0] == a and found.ex is None and found.qkv is None, block
        assert site(0, 32, 1).fold == pytest.approx(1.789667) and site(6, 64, 2).fold == pytest.approx(1.785119)
    finally:
        int4.disable()
