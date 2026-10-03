"""HTTP client for a Cosmos3-Edge `vllm serve --omni` server (T2I).

    python examples/cosmos3_edge/t2i_request.py --port 8091 --out edge_t2i.png
"""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8091)
parser.add_argument("--model", default="cosmos3-edge")
parser.add_argument("--prompt", default="A red sports car parked on a wet city street at golden hour, photorealistic")
parser.add_argument("--size", default="640x640")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--out", default="cosmos3_edge_t2i.png")
args = parser.parse_args()

body = {"model": args.model, "prompt": args.prompt, "size": args.size, "seed": args.seed,
        "num_inference_steps": args.steps, "response_format": "b64_json", "n": 1}
req = urllib.request.Request(f"http://{args.host}:{args.port}/v1/images/generations", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as resp:
    data = json.loads(resp.read())
elapsed = time.time() - t0
with open(args.out, "wb") as f:
    f.write(base64.b64decode(data["data"][0]["b64_json"]))
print(json.dumps({"out": args.out, "client_s": round(elapsed, 3), "size": args.size, "steps": args.steps}))
