"""Does the int4 mixed network reproduce DLSSNR-AMD's? Score the three verification frames against NVIDIA's output and
against the pictures DLSSNR-AMD's int4 mixed network makes of them.

    python scripts/int4_check.py [--verification ../DLSSNR-AMD/docs/ngx-verification] [--int4-tables DIR]

DLSSNR-AMD reports 41.92, 43.36 and 43.62 dB against NVIDIA at 1080p, 1440p and 4K for its int4 mixed network (45.55,
48.21 and 49.05 dB for the exact one). The score against its own pictures says how closely the arithmetic is reproduced.
"""
import argparse
import time
from pathlib import Path

from dlssnr import int4
from dlssnr.io import Controls, load_image
from dlssnr.metrics import psnr, ssim
from dlssnr.model import NRNet
from dlssnr.runtime import resolve_device
from dlssnr.weights import Weights

p = argparse.ArgumentParser()
p.add_argument("--verification", default=str(Path(__file__).resolve().parents[2] / "DLSSNR-AMD" / "docs" / "ngx-verification"))
p.add_argument("--int4-tables")
p.add_argument("--weights", default="weights.safetensors")
p.add_argument("--sizes", nargs="+", default=["1920x1080", "2560x1440", "3840x2160"])
args = p.parse_args()

device = resolve_device("auto")
weights = Weights.load(args.weights)
folder = Path(args.verification)
int4.enable(args.int4_tables)
print(f"{'frame':10s} {'vs NVIDIA':>10s} {'vs golden int4':>15s} {'golden vs NVIDIA':>17s} {'SSIM vs golden':>15s}  seconds")
for size in args.sizes:
    frame = load_image(folder / "single-frame-inputs" / f"{size}.png")
    nvidia = load_image(folder / "single-frame-outputs" / f"{size}_nvidia.png")
    golden = load_image(folder / "single-frame-outputs" / f"{size}_dlssnr-amd-rdna3-int4.png")
    started = time.time()
    net = NRNet(weights, frame.shape[1], frame.shape[0]).to(device)
    out = net(frame, Controls())
    print(f"{size:10s} {psnr(out, nvidia):9.2f}  {psnr(out, golden):14.2f}  {psnr(golden, nvidia):16.2f}  {ssim(out, golden):14.4f}  {time.time() - started:6.1f}",
          flush=True)
    del net
