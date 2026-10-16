"""Which window-shift pattern does the small network use? The library builds the network in code, so the pattern is
not in the weights; a wrong one should match NVIDIA's output worse than the right one.

    python scripts/small_shift_probe.py --weights weights-small.safetensors --inputs DIR --references DIR [--sizes ...]

``--inputs`` holds ``<size>.png`` (the verification frames) and ``--references`` ``<size>_nvidia.png``: NVIDIA's
310.9.16 output of the new model, from DLSSNR-AMD's ``docs/ngx-verification/310.9.16/single-frame-outputs`` (the inputs
are in its ``310.8.0/single-frame-inputs``). Each hypothesis replaces the shifts of the small plan and scores the
frames (PSNR against NVIDIA's output, and the correlation of the edit the network makes with NVIDIA's edit).
"""
import argparse
from pathlib import Path

import numpy as np

from dlssnr import plan as plan_module
from dlssnr.io import Controls, load_image
from dlssnr.metrics import psnr
from dlssnr.model import NRNet
from dlssnr.runtime import resolve_device
from dlssnr.weights import Weights

CYCLE = plan_module._SHIFT_CYCLE
p = argparse.ArgumentParser()
p.add_argument("--weights", required=True)
p.add_argument("--inputs", required=True)
p.add_argument("--references", required=True)
p.add_argument("--sizes", nargs="+", default=["1920x1080"])
args = p.parse_args()

original = list(plan_module.ROWS["small"])
SWIN_DECODER = range(22, 30)
SPLIT_DECODER = (20, 21)


def variant(encoder=None, decoder=None, split_decoder=None, split_encoder=None, none=False):
    """Rows with the shifts replaced: each argument is a function (index in the level or section) -> shift."""
    rows = []
    for row in original:
        shift = row.shift
        if none and row.cls != "CCTinlayoutFusedPostBlockSwin1H":
            shift = (0, 0)
        elif row.cls.startswith("CCTinlayoutFusedSwin") and row.block <= 8 and encoder:
            shift = encoder((row.block - 1) % 2)
        elif row.cls.startswith("CCTinlayoutFusedSwin") and row.block in SWIN_DECODER and decoder:
            shift = decoder((row.block - 22) % 2)
        elif row.cls == "CCSplitSwin16HQKVAttn" and row.block in SPLIT_DECODER and split_decoder:
            shift = split_decoder(row.block - 20)
        elif row.cls == "CCSplitSwin16HQKVAttn" and row.block in (9, 10) and split_encoder:
            shift = split_encoder(row.block - 9)
        rows.append(row._replace(shift=shift))
    return rows


HYPOTHESES = {
    "as built (levels restart; the decoder continues the cycle: starts at 2)": variant(),
    "decoder levels start at 0": variant(decoder=lambda i: CYCLE[i % 4]),
    "decoder levels start at 1": variant(decoder=lambda i: CYCLE[(1 + i) % 4]),
    "decoder levels start at 3": variant(decoder=lambda i: CYCLE[(3 + i) % 4]),
    "decoder split blocks restart the cycle (0, 1)": variant(split_decoder=lambda i: CYCLE[i % 4]),
    "split blocks unshifted": variant(split_decoder=lambda i: (0, 0), split_encoder=lambda i: (0, 0)),
    "encoder levels unshifted, decoder unshifted": variant(encoder=lambda i: (0, 0), decoder=lambda i: (0, 0)),
    "no shifts anywhere": variant(none=True),
}

device = resolve_device("auto")
weights = Weights.load(args.weights)
print(f"{'hypothesis':70s}" + "".join(f"{s:>22s}" for s in args.sizes))
for name, rows in HYPOTHESES.items():
    plan_module.ROWS["small"] = rows
    cells = []
    for size in args.sizes:
        frame = load_image(Path(args.inputs) / f"{size}.png")
        nvidia = load_image(Path(args.references) / f"{size}_nvidia.png")
        out = NRNet(weights, frame.shape[1], frame.shape[0], network="small").to(device)(frame, Controls())
        correlation = np.corrcoef((out.astype(float) - frame).ravel(), (nvidia.astype(float) - frame).ravel())[0, 1]
        cells.append(f"{psnr(out, nvidia):6.2f} dB, corr {correlation:.4f}")
    print(f"{name:70s}" + "".join(f"{c:>22s}" for c in cells), flush=True)
plan_module.ROWS["small"] = original
