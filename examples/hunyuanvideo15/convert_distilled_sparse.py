"""Build a Diffusers-layout HunyuanVideo-1.5 720p I2V distilled-sparse checkpoint.

The Tencent repo ships only the transformer for ``720p_i2v_distilled_sparse`` (original naming, one fp32
safetensors file). Everything else is shared with the 720p I2V Diffusers checkpoint, so the output directory is:

* ``transformer/``: the converted weights (diffusers' own ``convert_hyvideo15_transformer_to_diffusers``), saved
  sharded in the source dtype, plus a config that is the 720p I2V config with the upstream sparse-attention
  parameters kept verbatim under ``attn_mode`` / ``attn_param`` (ignored by diffusers, read by the Neuron port);
* ``guider/``: classifier-free guidance disabled (``guidance_scale`` 1.0, the model is CFG-distilled);
* ``scheduler/``: flow shift 7 (same as 720p I2V; written explicitly);
* every other component (VAE, Qwen2.5-VL, byT5, SigLIP, tokenizers, feature extractor): symlinks into the base.

Usage::

    python examples/hunyuanvideo15/convert_distilled_sparse.py \
        --upstream <dir with config.json + diffusion_pytorch_model.safetensors> \
        --base <HunyuanVideo-1.5-Diffusers-720p_i2v> --out <output dir> \
        --diffusers-script <diffusers>/scripts/convert_hunyuan_video1_5_to_diffusers.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys


def _header_shapes(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return {k: (tuple(v["shape"]), v["dtype"]) for k, v in h.items()}


def _dir_shapes(d: str) -> dict:
    out = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".safetensors"):
            out.update(_header_shapes(os.path.join(d, fn)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diffusers-script", required=True)
    ap.add_argument("--shard-size", default="10GB")
    args = ap.parse_args()

    from types import SimpleNamespace

    import torch
    from safetensors.torch import load_file, save_file  # noqa: F401

    spec = importlib.util.spec_from_file_location("hv15_convert", args.diffusers_script)
    conv = importlib.util.module_from_spec(spec)
    sys.argv = sys.argv[:1]
    spec.loader.exec_module(conv)

    with open(os.path.join(args.upstream, "config.json")) as f:
        up_cfg = json.load(f)
    with open(os.path.join(args.base, "transformer", "config.json")) as f:
        cfg = json.load(f)

    src = os.path.join(args.upstream, "diffusion_pytorch_model.safetensors")
    sd = load_file(src)
    n_src = len(sd)
    out_sd = conv.convert_hyvideo15_transformer_to_diffusers(sd, config=SimpleNamespace(**cfg))
    if sd:
        raise SystemExit(f"unconverted upstream keys: {sorted(sd)[:20]} ...")

    base_shapes = _dir_shapes(os.path.join(args.base, "transformer"))
    new_shapes = {k: (tuple(v.shape), str(v.dtype)) for k, v in out_sd.items()}
    missing = sorted(set(base_shapes) - set(new_shapes))
    extra = sorted(set(new_shapes) - set(base_shapes))
    bad = sorted(
        k for k in set(base_shapes) & set(new_shapes) if base_shapes[k][0] != new_shapes[k][0]
    )
    report = {
        "upstream_keys": n_src,
        "converted_keys": len(out_sd),
        "base_keys": len(base_shapes),
        "missing_vs_base": missing,
        "extra_vs_base": extra,
        "shape_mismatch": bad,
        "dtypes": sorted({v[1] for v in new_shapes.values()}),
        "params": sum(int(torch.Size(v[0]).numel()) for v in new_shapes.values()),
    }
    print(
        json.dumps(
            {
                k: (v if not isinstance(v, list) or len(v) < 20 else v[:20])
                for k, v in report.items()
            }
        )
    )
    if missing or extra or bad:
        raise SystemExit("shape/key mismatch vs the 720p I2V transformer")

    tdir = os.path.join(args.out, "transformer")
    os.makedirs(tdir, exist_ok=True)
    from huggingface_hub import split_torch_state_dict_into_shards

    split = split_torch_state_dict_into_shards(
        out_sd,
        filename_pattern="diffusion_pytorch_model{suffix}.safetensors",
        max_shard_size=args.shard_size,
    )
    for fn, keys in split.filename_to_tensors.items():
        save_file(
            {k: out_sd[k].contiguous() for k in keys},
            os.path.join(tdir, fn),
            metadata={"format": "pt"},
        )
    if split.is_sharded:
        with open(os.path.join(tdir, "diffusion_pytorch_model.safetensors.index.json"), "w") as f:
            json.dump(
                {"metadata": split.metadata, "weight_map": split.tensor_to_filename}, f, indent=2
            )
    cfg.update(
        {
            "attn_mode": up_cfg["attn_mode"],
            "attn_param": up_cfg["attn_param"],
            "_source": "tencent/HunyuanVideo-1.5 transformer/720p_i2v_distilled_sparse",
        }
    )
    with open(os.path.join(tdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for name in ("guider", "scheduler"):
        os.makedirs(os.path.join(args.out, name), exist_ok=True)
    with open(os.path.join(args.base, "guider", "guider_config.json")) as f:
        g = json.load(f)
    g["guidance_scale"] = 1.0
    with open(os.path.join(args.out, "guider", "guider_config.json"), "w") as f:
        json.dump(g, f, indent=2)
    with open(os.path.join(args.base, "scheduler", "scheduler_config.json")) as f:
        s = json.load(f)
    s["shift"] = 7.0
    with open(os.path.join(args.out, "scheduler", "scheduler_config.json"), "w") as f:
        json.dump(s, f, indent=2)
    for name in sorted(os.listdir(args.base)):
        if name in ("transformer", "guider", "scheduler") or name.startswith("."):
            continue
        dst = os.path.join(args.out, name)
        if not os.path.lexists(dst):
            os.symlink(os.path.join(os.path.abspath(args.base), name), dst)
    with open(os.path.join(args.out, "conversion_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[convert] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
