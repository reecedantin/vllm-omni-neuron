# SPDX-License-Identifier: Apache-2.0
"""Build a DROID-shaped observation NPZ from the demo clip shipped with the policy weights.

Use it when no recorded DROID episode is at hand. Three robot-arm tiles are cropped from one frame of
``<policy>/assets/flux-3-action-demo.mp4`` and resized to the DROID camera size (360x640); the state is a
plausible DROID joint configuration. The same inputs always give the same NPZ, so CPU reference runs and
Neuron runs see bit-identical observations. For a recorded DROID episode, use the upstream
``examples/droid/make_observation.py`` instead (same NPZ format).

    python examples/flux3_action/make_observation.py --policy <flux-3-action-droid dir> --output obs.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

DEMO_CLIP = os.path.join("assets", "flux-3-action-demo.mp4")
FRAME = 60
# tile boxes (x0, y0, x1, y1) in the 2040x800 source frame
BOXES = {
    "images.wrist": (961, 483, 1188, 598),
    "images.left": (1191, 150, 1414, 282),
    "images.right": (1543, 485, 1733, 600),
}
STATE = [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8, 0.0]
TASK = "put the screwdriver in the box"


def make_observation(video: str) -> dict[str, np.ndarray]:
    import av
    from PIL import Image

    img = None
    with av.open(video) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i == FRAME:
                img = f.to_image().convert("RGB")
                break
    if img is None:
        raise ValueError(f"{video} has fewer than {FRAME + 1} frames")
    out: dict[str, np.ndarray] = {}
    for key, box in BOXES.items():
        tile = img.crop(box).resize((640, 360), Image.BICUBIC)
        out[key] = np.asarray(tile, dtype=np.uint8).copy()  # (360, 640, 3) HWC
    out["state"] = np.asarray(STATE, dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--policy",
        default=os.environ.get("FLUX3_ACTION_DROID"),
        help="flux-3-action-droid package dir (reads assets/flux-3-action-demo.mp4)",
    )
    ap.add_argument("--video", default=None, help="explicit clip path (overrides --policy)")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    video = a.video or (os.path.join(a.policy, DEMO_CLIP) if a.policy else None)
    if not video:
        ap.error("pass --policy or --video (or set FLUX3_ACTION_DROID)")
    np.savez_compressed(a.output, **make_observation(video))
    with open(os.path.splitext(a.output)[0] + ".json", "w") as f:
        json.dump(
            {"task": TASK, "source": os.path.basename(video), "frame": FRAME, "boxes": BOXES},
            f,
            indent=1,
        )
    print(a.output)


if __name__ == "__main__":
    main()
