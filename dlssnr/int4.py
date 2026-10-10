"""The int4 mixed network's weights: rebuilt from the e4m3 weights and the recovery tables of DLSSNR-AMD.

DLSSNR-AMD's int4 weights are GPTQ solutions of the library's own weights, so they cannot be shipped. What ships
(``<prefix>.fix.bin``, ``<prefix>.mu.bin``, ``<prefix>.aq.bin``) holds no weight value: per row the index of a
clip step, per weight a correction of -1, 0 or +1 against plain rounding (and a few larger "escapes"), and per
input channel the calibration means and the input quantiser. ``decode`` rebuilds from the weights at hand::

    Wr  = (W . H_B) * step                    block Hadamard of size B, step = the input quantiser's step
    s_n = max|Wr_n| * clip[ci_n] / 7          (float32)
    q   = clamp(rint(Wr / s), -8, 7) + correction
    c_n = W_n . muX - s_n * q_n . nu + dh_n   (the constant, float32)

so that the layer computes ``s_n * (q_n . x_int4) + c_n`` for the integer-quantised input ``x_int4``. The result
is the solved one bit for bit; ``tests/test_int4.py`` checks it against the file DLSSNR-AMD's installer writes.
This module follows ``i4_obtain`` in DLSSNR-AMD's ``linux/src/core/nr_graph.cpp`` operation for operation, and the
shaders' quantisers and epilogues for the runtime; DLSSNR-AMD is MIT licensed, Copyright (c) 2026 mochizuki0323.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)
_MAGIC = 0x58463449          # "I4FX"


def _e4m3_table() -> np.ndarray:
    """The solver's e4m3 decode: both NaN codes (0x7f, 0xff) read as 0."""
    table = np.zeros(256)
    for code in range(256):
        if code & 0x7F == 0x7F:
            continue
        exponent, mantissa = (code >> 3) & 15, code & 7
        value = mantissa * 2.0 ** -9 if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        table[code] = -value if code & 0x80 else value
    return table


E4M3 = _e4m3_table()


def hadamard(rows: np.ndarray, block: int) -> np.ndarray:
    """The normalised Walsh-Hadamard transform of every ``block`` consecutive columns (float64, as the solver does)."""
    if block == 1:
        return rows
    n, k = rows.shape
    x = rows.reshape(n, k // block, block).copy()
    h = 1
    while h < block:
        y = x.reshape(n, k // block, block // (2 * h), 2, h)
        a, b = y[..., 0, :].copy(), y[..., 1, :].copy()
        y[..., 0, :], y[..., 1, :] = a + b, a - b
        h *= 2
    return (x * (1.0 / np.sqrt(float(block)))).reshape(n, k)


def _sequential_dot(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """``matrix @ vector`` summed in column order in float64, as the solver's scalar loop sums it."""
    total = np.zeros(matrix.shape[0])
    for k in range(matrix.shape[1]):
        total += matrix[:, k] * vector[k]
    return total


def decode(prefix: str | Path, wsel: np.ndarray, aq: np.ndarray, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild one matrix. ``wsel`` is the e4m3 codes ``[N, K4]`` (uint8) of the columns that run in int4, ``aq`` the
    input quantiser ``[K4, 2]`` float32 (1 / step, zero). Returns ``(q int8 [N, K4], s float32 [N], c float32 [N])``."""
    prefix = str(prefix)
    n_rows, k4 = wsel.shape
    fix = Path(prefix + ".fix.bin").read_bytes()
    mu = Path(prefix + ".mu.bin").read_bytes()
    header = struct.unpack_from("<6I", fix, 0)
    codes_size = (n_rows * k4 + 3) // 4
    if (header[0] != _MAGIC or header[1] != n_rows or header[2] != k4 or header[3] != block
            or len(fix) != 24 + n_rows + codes_size + header[4] * 8 or len(mu) not in (k4 * 8, k4 * 8 + n_rows * 4)):
        raise ValueError(f"int4 table {prefix} does not match this model")
    clip_index = np.frombuffer(fix, np.uint8, n_rows, 24)
    if clip_index.max() > 8:
        raise ValueError(f"int4 table {prefix}: clip index")
    packed = np.frombuffer(fix, np.uint8, codes_size, 24 + n_rows)
    correction = np.stack([(packed >> (2 * i)) & 3 for i in range(4)], axis=1).reshape(-1)[: n_rows * k4]
    correction = np.where(correction == 1, 1, np.where(correction == 2, -1, 0)).reshape(n_rows, k4)
    escapes = np.frombuffer(fix, np.uint32, header[4] * 2, 24 + n_rows + codes_size).reshape(-1, 2)
    escape_index, escape_delta = escapes[:, 0], escapes[:, 1].view(np.int32)
    means = np.frombuffer(mu, np.float32, k4 * 2)
    mu_x, nu = means[:k4].astype(np.float64), means[k4:].astype(np.float64)
    shift = (np.frombuffer(mu, np.float32, n_rows, k4 * 8).astype(np.float64) if len(mu) > k4 * 8 else np.zeros(n_rows))

    weights = E4M3[wsel]
    rotated = hadamard(weights, block)
    step = (np.float32(1.0) / aq[:, 0].astype(np.float32)).astype(np.float64)       # float(1.0f / aq)
    rotated = rotated * step[None, :]
    peak = np.maximum(np.abs(rotated).max(axis=1), 1e-30)
    scale = (peak * np.array(CLIPS)[clip_index] / 7.0).astype(np.float32)             # float(am * clip / 7)

    def reference(rows, s):
        return np.clip(np.rint(rotated[rows] / s.astype(np.float64)[:, None]), -8, 7)

    q = reference(slice(None), scale) + correction
    for index, delta in zip(escape_index, escape_delta):
        row, column = divmod(int(index), k4)
        q[row, column] = np.clip(np.rint(rotated[row, column] / float(scale[row])), -8, 7) + int(delta)
    constant = (_sequential_dot(weights, mu_x) - scale.astype(np.float64) * _sequential_dot(q, nu) + shift).astype(np.float32)
    return q.astype(np.int8), scale, constant


# ---- runtime ------------------------------------------------------------------------------------------------------

import math
import os

import torch
from torch import nn

MAGIC = 12582912.0           # 1.5 * 2**23: adding it to a float32 rounds the value to an integer

tables: "Tables | None" = None
side: dict = {}              # f16 values a producer layer leaves for the layer that quantises them (cleared per frame)

DEFAULT_TABLES = os.environ.get("DLSSNR_INT4_TABLES", str(Path(__file__).resolve().parents[2] / "DLSSNR-AMD" / "linux" / "data" / "int4" / "rdna4"))


class Tables:
    """A directory of DLSSNR-AMD's int4 recovery tables: ``vit/``, ``swin/`` and ``settings.txt``."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if not (self.directory / "vit").is_dir():
            raise FileNotFoundError(f"{self.directory} is not an int4 table directory (no vit/); set --int4-tables or "
                                    f"DLSSNR_INT4_TABLES to DLSSNR-AMD's linux/data/int4/rdna4")
        self.settings = {}
        settings = self.directory / "settings.txt"
        if settings.exists():
            for line in settings.read_text().splitlines():
                key, _, value = line.strip().partition("=")
                if key and not key.startswith("#") and value:
                    self.settings[key] = value

    def path(self, *parts: str) -> Path:
        return self.directory.joinpath(*parts)

    def has(self, *parts: str) -> bool:
        return self.path(*parts).exists()


def enable(directory: str | Path | None = None) -> None:
    global tables
    tables = Tables(directory or DEFAULT_TABLES)


def disable() -> None:
    global tables
    tables = None
    side.clear()


def active() -> bool:
    return tables is not None


def reset() -> None:
    """Forget the side copies of the previous frame."""
    side.clear()


def fwht(x: torch.Tensor, block: int) -> torch.Tensor:
    """The unnormalised Walsh-Hadamard transform of every ``block`` consecutive values of the last dimension, in float32
    with the butterfly stages in ascending order (the shaders' order)."""
    if block == 1:
        return x
    shape, k = x.shape, x.shape[-1]
    y = x.reshape(-1, k // block, block)
    h = 1
    while h < block:
        y = y.reshape(-1, k // block, block // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return y.reshape(shape)


class Int4Matrix(nn.Module):
    """One int4 matrix with its input quantiser: ``y = s * (q . x_int4) + c`` for ``x_int4 = quantise(x)``.

    ``half`` is the Swin sites' form, whose quantiser (1 / step, zero) and affine (s, c) DLSSNR-AMD stores as float16;
    the ViT sites keep them float32.
    """

    def __init__(self, q: np.ndarray, scale: np.ndarray, constant: np.ndarray, aq: np.ndarray, block: int, half: bool = False):
        super().__init__()
        self.block, self.half = block, half
        self.register_buffer("q", torch.from_numpy(q.astype(np.float32)))                    # [N, K4], -8..7
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))
        self.register_buffer("constant", torch.from_numpy(constant.astype(np.float32)))
        inverse = torch.from_numpy(aq[:, 0].astype(np.float32))
        zero = torch.from_numpy(aq[:, 1].astype(np.float32))
        if half:
            gain, zero = inverse.half().float(), zero.half().float()
        else:
            # the shaders' (a, z + magic): a = inverse step / sqrt(B) in float32; the zero point rounds to an integer
            gain = inverse * torch.tensor(1.0 / math.sqrt(block), dtype=torch.float32) if block > 1 else inverse
            zero = (zero + MAGIC) - MAGIC
        self.register_buffer("gain", gain)
        self.register_buffer("zero", zero)

    def quantise(self, value: torch.Tensor) -> torch.Tensor:
        """``[T, K4]`` float32 -> the int4 values ``clamp(rint(H(x) * a + z), 0, 15) - 8`` as float32 integers."""
        rotated = fwht(value, self.block)
        # one rounding of rotated * a + z, as the shader's fma, taken exactly in float64
        code = torch.round(rotated.double() * self.gain.double() + self.zero.double()).clamp_(0, 15)
        return (code - 8).float()

    def accumulate(self, quantised: torch.Tensor) -> torch.Tensor:
        """``float(int accumulator) * s + c`` for already quantised input ``[T, K4]`` -> ``[T, N]`` float32."""
        accumulator = quantised @ self.q.t()                      # integers below 2**24: exact in any order
        scale, constant = (self.scale.half().float(), self.constant.half().float()) if self.half else (self.scale, self.constant)
        return (accumulator.double() * scale.double() + constant.double()).float()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.accumulate(self.quantise(value))


def raw_codes(weights, block: int, layer: int, *names: str, shape: tuple[int, int]) -> np.ndarray:
    """The e4m3 bytes of one or more weight tensors stacked into ``shape`` (rows, columns)."""
    parts = [weights.raw(block, layer, name).numpy().reshape(-1, shape[1]) for name in names]
    return np.concatenate(parts)


def _aq(path: Path, k4: int) -> np.ndarray:
    return np.fromfile(path, np.float32).reshape(k4, 2)


def build_matrix(prefix: Path, wsel: np.ndarray, aq_path: Path, block: int, half: bool = False) -> Int4Matrix:
    aq = _aq(aq_path, wsel.shape[1])
    q, scale, constant = decode(prefix, wsel, aq, block)
    return Int4Matrix(q, scale, constant, aq, block, half)


# ---- the vision-transformer blocks ----------------------------------------------------------------------------------

class VitFfn(nn.Module):
    """Block ``b``'s feed-forward in int4: the hidden channels in the order of ``b<b>.perm.bin`` (the first ``k4`` run in
    int4 in the contraction, the rest in e4m3), the expansion's int4 matrix when the tables have one."""

    def __init__(self, weights, block: int):
        super().__init__()
        vit = tables.path("vit")
        perm = np.fromfile(vit / f"b{block}.perm.bin", np.uint16).astype(np.int64)
        aq_size = (vit / f"b{block}.aq.bin").stat().st_size
        self.k4 = aq_size // 8
        self.register_buffer("perm", torch.from_numpy(perm))
        self.expand = None
        if tables.has("vit", f"b{block}.ex.fix.bin"):
            wsel = raw_codes(weights, block, 0, "weight", shape=(4096, 1024))[perm]
            self.expand = build_matrix(vit / f"b{block}.ex", wsel, vit / f"b{block}.ex.aq.bin", 16)
        full = raw_codes(weights, block, 1, "weight", shape=(1024, 4096))
        rotation = int((vit / "rot.txt").read_text()) if (vit / "rot.txt").exists() else 64
        self.contract = build_matrix(vit / f"b{block}", full[:, perm[: self.k4]], vit / f"b{block}.aq.bin", rotation)
        tail = torch.from_numpy(E4M3[full[:, perm[self.k4:]]].astype(np.float32))              # the e4m3 channels, exact
        self.register_buffer("tail", tail)                                                      # [1024, 4096 - k4]

    def expand_preactivation(self, tokens: torch.Tensor, block: int, exact) -> torch.Tensor:
        """``W x`` of the expansion, ``[T, 4096]`` in the original channel order: from the previous projection's int4
        copy when the tables have this matrix, else ``exact(tokens)``."""
        if self.expand is None:
            return exact(tokens)
        permuted = self.expand(side[("stream", block)])
        out = torch.empty_like(permuted)
        out[:, self.perm] = permuted
        return out

    def contraction(self, hidden_e4m3: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """The contraction's matrix product: the int4 channels from the unrounded activation ``hidden``, the others from the
        e4m3 values ``hidden_e4m3`` (both ``[T, 4096]``, original order); the sum, in float32."""
        total = self.contract(hidden[:, self.perm[: self.k4]])
        return total + hidden_e4m3[:, self.perm[self.k4:]] @ self.tail.t()


def vit_ffn(weights, block: int) -> VitFfn | None:
    if tables is None or not tables.has("vit", f"b{block}.perm.bin"):
        return None
    return VitFfn(weights, block)


def vit_matrix(weights, block: int, name: str) -> Int4Matrix | None:
    """The int4 QKV (``name`` "qkv") of ViT block ``block``; ``None`` without tables."""
    if tables is None or not tables.has("vit", f"b{block}.{name}.fix.bin"):
        return None
    vit = tables.path("vit")
    wsel = raw_codes(weights, block, 2, "q", "k", "v", shape=(3072, 1024))
    return build_matrix(vit / f"b{block}.qkv", wsel, vit / f"b{block}.qkv.aq.bin", 16)


def attention_matrix(weights, block: int, layer: int) -> Int4Matrix | None:
    """The int4 QKV of a C=512 split-Swin attention block, ``[1536, 512]``; ``None`` without tables."""
    if tables is None or not tables.has("vit", f"b{block}.attn.fix.bin"):
        return None
    vit = tables.path("vit")
    wsel = raw_codes(weights, block, layer, "qkv", shape=(1536, 512))
    return build_matrix(vit / f"b{block}.attn", wsel, vit / f"b{block}.attn.aq.bin", 16)


# ---- the Swin blocks -------------------------------------------------------------------------------------------------

HARD_SWISH = {32: (0.187367, 0.500139)}       # (A, B) of x * clamp(A x + B, 0, 1) by channel count; the others take WIDE
HARD_SWISH_WIDE = (0.182412, 0.500465)
FIRST_AND_LAST = (0, 70)                       # the pre and post blocks take their own fold constant


def hard_swish(x: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """``x * clamp(fma(x, A, B), 0, 1)`` in float32 (the product and sum of the fma rounded once)."""
    gate = (x.double() * np.float32(a) + np.float32(b)).float().clamp(0.0, 1.0)
    return x * gate


class SwinSite(nn.Module):
    """What the int4 mixed network changes in one Swin block: the activation (hard-swish when ``hs``), the scale folded
    into the weights that read the hidden, and the int4 matrices of the MLP expansion and the QKV."""

    def __init__(self, weights, block: int, layer: int, channels: int, heads: int):
        super().__init__()
        settings = tables.settings
        listed = settings.get("NR_HS_FOLD_BLOCKS")
        blocks = {int(b) for b in listed.split("+")} if listed else None

        def constant(name: str, default: float) -> float:
            return float(np.float32(settings[name])) if name in settings else default

        wide = constant("NR_HS_FOLD", 0.0)
        narrow = constant("NR_HS_FOLD32", wide)                        # C = 32; the host's defaults chain the same way
        first_last = constant("NR_HS_FOLDPP", narrow)
        fold = first_last if block in FIRST_AND_LAST else narrow if channels == 32 else wide
        if blocks is not None and block not in blocks:
            fold = 0.0
        self.fold = fold
        self.hs = (HARD_SWISH.get(channels, HARD_SWISH_WIDE)) if fold else None
        self.ex = self.qkv = None
        name = f"b{block}l{layer}"
        swin = tables.path("swin")
        if heads > 1 and tables.has("swin", f"{name}.x.aq.bin") and tables.has("swin", f"{name}.y.aq.bin"):
            self.ex = build_matrix(swin / f"{name}.ex", raw_codes(weights, block, layer, "mlp_expand", shape=(4 * channels, channels)),
                                   swin / f"{name}.x.aq.bin", 1, half=True)
            self.qkv = build_matrix(swin / f"{name}.qkv", raw_codes(weights, block, layer, "qkv", shape=(3 * channels, channels)),
                                    swin / f"{name}.y.aq.bin", 1, half=True)


def swin_site(weights, block: int, layer: int, channels: int, heads: int) -> SwinSite | None:
    if tables is None:
        return None
    site = SwinSite(weights, block, layer, channels, heads)
    return site if (site.hs or site.ex is not None) else None


def swin_matrix(weights, block: int, site: str) -> Int4Matrix | None:
    """The int4 ``ex`` or ``qkv`` matrix of Swin block ``block`` (for tests)."""
    channels = 256 if 15 <= block <= 22 or 48 <= block <= 55 else 128
    found = swin_site(weights, block, 0, channels, 8 if channels == 256 else 4)
    return None if found is None else getattr(found, site)
