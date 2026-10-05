"""Compare final latent rows of a Neuron run against a reference run (the FastH3 parity gate).

    python examples/minimax_h3/eval/gate_compare.py --ref ref_fp32_256.pt --run dump.pt [--out gate.json]

Both files hold ``video_rows`` (Nv, 96) and ``audio_rows`` (Na, 32): the reference from the fp32 CPU run of the
earlier port (``ref_fp32_256_neuronprompt.pt``), the run from ``MINIMAX_H3_DUMP_LATENTS``. Gate (FastH3 256p, the
standard of both earlier ports): final video rel-L2 <= 27.2 % and cosine >= 0.963 (bf16 CPU diffusers: 27.2 % /
0.963; trn1 bf16 port: 23.5 % / 0.972; inf2 W8 port: 25.0 % / 0.9685).
"""

from __future__ import annotations

import argparse
import json
import sys
import types

import torch

GATE_REL, GATE_COS = 0.272, 0.963


def _load(path: str) -> dict:
    # the reference pickles the earlier port's layout class; stub it so the tensors load without that package
    for name in ("fasth3_neuron", "fasth3_neuron.layout"):
        sys.modules.setdefault(name, types.ModuleType(name))

    class H3Layout:
        def __setstate__(self, s):
            self.__dict__.update(s)

    sys.modules["fasth3_neuron.layout"].H3Layout = H3Layout
    return torch.load(path, map_location="cpu", weights_only=False)


def _rc(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    return {
        "rel_l2": ((a - b).norm() / b.norm()).item(),
        "cos": torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
    }


def compare(ref: dict, run: dict, floor: dict | None = None) -> dict:
    """Final latents (e2e tier) and, when both files carry per-step velocities, the step-0 velocity (single-step
    tier: same input noise, no compounding). ``floor`` = a CPU bf16 run of the same request: the project rule is
    device-vs-fp32 <= bf16-vs-fp32 (both reported). Without a floor the fixed FastH3 4-step 256p bar applies."""
    out = {
        "video": _rc(run["video_rows"], ref["video_rows"]),
        "audio": _rc(run["audio_rows"], ref["audio_rows"]),
    }
    if run.get("steps") and ref.get("steps"):
        out["step0"] = {
            "video": _rc(run["steps"][0]["vel_video"], ref["steps"][0]["vel_video"]),
            "audio": _rc(run["steps"][0]["vel_audio"], ref["steps"][0]["vel_audio"]),
        }
    v = out["video"]
    if floor is not None:
        f = {
            "video": _rc(floor["video_rows"], ref["video_rows"]),
            "audio": _rc(floor["audio_rows"], ref["audio_rows"]),
        }
        if floor.get("steps") and ref.get("steps"):
            f["step0"] = {
                "video": _rc(floor["steps"][0]["vel_video"], ref["steps"][0]["vel_video"])
            }
        out["cpu_bf16_floor"] = f
        # fleet bar: device-vs-fp32 <= 2 x (CPU-bf16-vs-fp32) + 0.5 %, on the e2e latents and (if present) step 0
        bar = lambda x: 2.0 * x + 0.005  # noqa: E731
        ok = v["rel_l2"] <= bar(f["video"]["rel_l2"])
        if "step0" in out and "step0" in f:
            ok = ok and out["step0"]["video"]["rel_l2"] <= bar(f["step0"]["video"]["rel_l2"])
        out["gate"] = {
            "rule": "device <= 2 x CPU-bf16 floor + 0.5% (video rel-L2; e2e and step 0)",
            "e2e_bar": bar(f["video"]["rel_l2"]),
            "pass": ok,
        }
        if "step0" in f:
            out["gate"]["step0_bar"] = bar(f["step0"]["video"]["rel_l2"])
    else:
        out["gate"] = {
            "rel_max": GATE_REL,
            "cos_min": GATE_COS,
            "pass": v["rel_l2"] <= GATE_REL and v["cos"] >= GATE_COS,
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument(
        "--floor",
        default=None,
        help="CPU bf16 run of the same request (reference_cpu.py --dtype bf16)",
    )
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = compare(_load(a.ref), _load(a.run), _load(a.floor) if a.floor else None)
    print(json.dumps(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    sys.exit(0 if res["gate"]["pass"] else 1)
