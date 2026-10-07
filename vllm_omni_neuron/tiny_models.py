# SPDX-License-Identifier: Apache-2.0
"""Tiny random-weight checkpoints in a real model's exact on-disk layout (M-tiny).

Given a real Diffusers/transformers checkpoint directory, write a copy whose every component keeps the
real module / parameter names, dtypes, folder layout (``model_index.json`` + component folders,
tokenizers, schedulers, processors) and configs -- with fewer layers and, optionally, smaller dims --
so loading, weight-name mapping, sharding, compile and one device forward can be proven in minutes.
Says nothing about quality: the weights are random.

Two modes per component (a folder, or the checkpoint root, holding ``config.json`` + weights):

``synthesize`` (default; needs no model code)
    Reads the source safetensors HEADERS only (names, shapes, dtypes), keeps the first ``layers``
    entries of every indexed block list whose length matches a layer-count field of the config
    (``num_layers``, ``num_hidden_layers``, ... also inside nested ``text_config`` etc.), and writes
    random tensors of the original shapes and dtypes. Widths are unchanged, so it works for any
    architecture, including classes the installed diffusers/transformers do not have.

``instantiate`` (needs the class: diffusers ``_class_name`` or transformers ``architectures[0]``, or
a ``class_resolver``)
    Applies the layer shrink plus explicit config ``overrides`` (dims, heads...), builds the model
    with random init, casts each tensor to the source dtype of the same-named tensor, and saves it
    with ``save_pretrained``. Use it to shrink widths; :func:`compare_layout` then checks the
    parameter names still match the real checkpoint.

Non-weight files (tokenizers, scheduler/processor configs, templates) are copied verbatim when they
are smaller than ``copy_limit_mb``. Never commit generated weights; commit the script/call that
makes them.

CLI::

    python -m vllm_omni_neuron.tiny_models SRC OUT --layers 2 \\
        --mode transformer=instantiate --set transformer.num_attention_heads=4 \\
        --set transformer.attention_head_dim=32
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

WEIGHT_EXTS = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".msgpack", ".h5")
LAYER_COUNT_KEYS = (
    "num_layers",
    "num_hidden_layers",
    "n_layers",
    "n_layer",
    "num_decoder_layers",
    "num_encoder_layers",
    "num_single_layers",
    "num_refiner_layers",
    "num_blocks",
    "depth",
    "num_transformer_blocks",
    "num_single_transformer_blocks",
    "num_double_layers",
)
_INDEX = re.compile(r"\.(\d+)\.")

ClassResolver = Callable[[str, dict], Any]


# ----------------------------------------------------------------------------- source inspection


def weight_files(folder: str) -> list[str]:
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".safetensors") and os.path.isfile(os.path.join(folder, f))
    )


def read_headers(folder: str) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``{name: (shape, dtype)}`` for every tensor in ``folder``'s safetensors files (headers only)."""
    from safetensors import safe_open

    out = {}
    for path in weight_files(folder):
        with safe_open(path, "pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                out[k] = (tuple(sl.get_shape()), _st_dtype(sl.get_dtype()))
    return out


_ST_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}


def _st_dtype(name: str) -> torch.dtype:
    return _ST_DTYPES[name]


def normalize_key(name: str) -> str:
    """Parameter name with block indices erased (``blocks.3.attn`` -> ``blocks.#.attn``)."""
    return _INDEX.sub(".#.", f".{name}.")[1:-1]


def layer_lists(names) -> dict[str, int]:
    """``{prefix: length}`` of every indexed module list (``prefix.<i>.``) found in ``names``."""
    lens: dict[str, int] = {}
    for n in names:
        for m in re.finditer(r"(?:^|\.)(\d+)\.", n):
            prefix = n[: m.start(1)].rstrip(".")
            lens[prefix] = max(lens.get(prefix, 0), int(m.group(1)) + 1)
    return lens


# ----------------------------------------------------------------------------- config shrinking


def shrink_layer_counts(cfg: dict, layers: int) -> tuple[dict, dict[int, int]]:
    """Copy of ``cfg`` with every layer-count field (recursively) capped at ``layers``.

    Returns ``(new_cfg, {old_count: new_count})`` for the counts that changed.
    """
    changed: dict[int, int] = {}

    def walk(d):
        for k, v in list(d.items()):
            if isinstance(v, dict):
                walk(v)
            elif (
                k in LAYER_COUNT_KEYS
                and isinstance(v, int)
                and not isinstance(v, bool)
                and v > layers
            ):
                changed[v] = layers
                d[k] = layers

    new = copy.deepcopy(cfg)
    walk(new)
    return new, changed


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Set dotted keys (``text_config.hidden_size``) in a copy of ``cfg``."""
    new = copy.deepcopy(cfg)
    for dotted, value in overrides.items():
        d = new
        *parents, leaf = dotted.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = value
    return new


# ----------------------------------------------------------------------------- tensors


def _init_tensor(
    name: str, shape: tuple[int, ...], dtype: torch.dtype, gen: torch.Generator
) -> torch.Tensor:
    if not dtype.is_floating_point:
        return torch.zeros(shape, dtype=dtype)
    leaf = name.rsplit(".", 1)[-1]
    is_norm = "norm" in name.lower() or ".ln" in name or name.startswith("ln")
    if leaf == "bias" or len(shape) == 0:
        t = 0.02 * torch.randn(shape, generator=gen)
    elif is_norm and len(shape) == 1:
        t = 1.0 + 0.1 * torch.randn(
            shape, generator=gen
        )  # non-trivial, so a wrong norm load is visible
    else:
        fan_in = shape[-1] if len(shape) > 1 else shape[0]
        if len(shape) > 2:  # conv: in_ch * kernel
            fan_in = 1
            for s in shape[1:]:
                fan_in *= s
        t = torch.randn(shape, generator=gen) / max(fan_in, 1) ** 0.5
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.clamp(-1, 1).to(dtype)
    return t.to(dtype)


def _weights_basename(folder: str) -> str:
    files = [os.path.basename(p) for p in weight_files(folder)]
    idx = [f for f in os.listdir(folder) if f.endswith(".safetensors.index.json")]
    name = idx[0][: -len(".index.json")] if idx else (files[0] if files else "model.safetensors")
    return re.sub(r"-\d{5}-of-\d{5}", "", name)


# ----------------------------------------------------------------------------- components


@dataclass
class ComponentReport:
    name: str
    mode: str
    params: int = 0
    tensors: int = 0
    layer_lists: dict[str, tuple[int, int]] = field(default_factory=dict)
    config_changes: dict[int, int] = field(default_factory=dict)


def _load_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def synthesize_component(
    src: str, out: str, layers: int, changed: dict[int, int], *, seed: int = 0, name: str = ""
) -> ComponentReport:
    """Random tensors with the source's names/shapes/dtypes, keeping the first ``layers`` entries of
    every block list whose length is one of the ``changed`` layer counts."""
    headers = read_headers(src)
    lists = layer_lists(headers)
    shrink = {p: layers for p, n in lists.items() if n in changed}
    gen = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    rep = ComponentReport(
        name,
        "synthesize",
        config_changes=dict(changed),
        layer_lists={p: (lists[p], k) for p, k in shrink.items()},
    )
    for key in sorted(headers):
        if not _keep(key, shrink):
            continue
        shape, dtype = headers[key]
        tensors[key] = _init_tensor(key, shape, dtype, gen)
        rep.params += tensors[key].numel()
    rep.tensors = len(tensors)
    from safetensors.torch import save_file

    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, _weights_basename(src)), metadata={"format": "pt"})
    return rep


def _keep(key: str, shrink: dict[str, int]) -> bool:
    for prefix, keep in shrink.items():
        head = f"{prefix}." if prefix else ""
        if key.startswith(head):
            rest = key[len(head) :]
            m = re.match(r"(\d+)\.", rest)
            if m and int(m.group(1)) >= keep:
                return False
    return True


def default_class_resolver(library_hint: str, cfg: dict):
    """diffusers ``_class_name`` or transformers ``architectures[0]``; None when not installed."""
    if "_class_name" in cfg:
        import diffusers

        return getattr(diffusers, cfg["_class_name"], None)
    archs = cfg.get("architectures") or []
    if archs:
        import transformers

        return getattr(transformers, archs[0], None)
    return None


def instantiate_component(
    src: str,
    out: str,
    layers: int,
    *,
    overrides: dict[str, Any] | None = None,
    seed: int = 0,
    name: str = "",
    class_resolver: ClassResolver = default_class_resolver,
) -> ComponentReport:
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    new_cfg, changed = shrink_layer_counts(cfg, layers)
    new_cfg = apply_overrides(new_cfg, overrides or {})
    cls = class_resolver(name, new_cfg)
    if cls is None:
        raise ValueError(
            f"{name or src}: no class for this config; use mode 'synthesize' or a class_resolver"
        )
    torch.manual_seed(seed)
    if "_class_name" in new_cfg:
        model = cls.from_config({k: v for k, v in new_cfg.items() if not k.startswith("_")})
    else:
        model = cls(cls.config_class.from_dict(new_cfg))
    src_dtypes = {normalize_key(k): dt for k, (_, dt) in read_headers(src).items()}
    with torch.no_grad():
        for pname, t in list(model.named_parameters()) + list(model.named_buffers()):
            dt = src_dtypes.get(normalize_key(pname))
            if dt is not None and t.dtype != dt:
                t.data = t.data.to(dt)
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    rep = ComponentReport(name, "instantiate", config_changes=changed)
    rep.params = sum(p.numel() for p in model.parameters())
    rep.tensors = len(model.state_dict())
    return rep


# ----------------------------------------------------------------------------- pipeline


def make_tiny_checkpoint(
    src: str,
    out: str,
    *,
    layers: int = 2,
    modes: dict[str, str] | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    seed: int = 0,
    copy_limit_mb: float = 64.0,
    class_resolver: ClassResolver = default_class_resolver,
) -> dict[str, Any]:
    """Write a tiny copy of the checkpoint at ``src`` to ``out`` (must not exist or be empty).

    ``modes`` / ``overrides`` are keyed by component folder name (``"."`` = the checkpoint root).
    Returns a manifest (also written to ``out/tiny_manifest.json``).
    """
    modes, overrides = modes or {}, overrides or {}
    if os.path.exists(out) and os.listdir(out):
        raise FileExistsError(f"{out} exists and is not empty")
    os.makedirs(out, exist_ok=True)
    walk = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames.sort()
        walk.append((os.path.relpath(dirpath, src), sorted(filenames)))
    walk.sort()

    # Layer counts per folder: its own config.json plus every ancestor's (a weights folder without its
    # own config, e.g. a vision encoder, is described by the root config's vision_config).
    own: dict[str, dict[int, int]] = {}
    for rel, files in walk:
        cfg = _load_json(os.path.join(src, rel, "config.json")) if "config.json" in files else None
        own[rel] = shrink_layer_counts(cfg, layers)[1] if cfg is not None else {}

    def counts(rel: str) -> dict[int, int]:
        if _load_json(os.path.join(src, rel, "config.json")) is not None:
            return own.get(rel, {})  # a folder with its own config is described by it alone
        parts = [] if rel == "." else rel.split(os.sep)
        for i in range(len(parts) - 1, -1, -1):  # nearest ancestor with a config
            anc = os.path.join(*parts[:i]) if i else "."
            if _load_json(os.path.join(src, anc, "config.json")) is not None:
                return own.get(anc, {})
        return {}

    reports: list[ComponentReport] = []
    copied: list[str] = []
    indexes: list[str] = []
    for i, (rel, files) in enumerate(walk):
        sdir, dst = os.path.join(src, rel), os.path.join(out, rel)
        os.makedirs(dst, exist_ok=True)
        has_weights = any(f.endswith(".safetensors") for f in files)
        mode = modes.get(rel, "synthesize")
        instantiated = False
        if has_weights:
            kw = dict(seed=seed + i, name=rel)
            if mode == "instantiate":
                if "config.json" not in files:
                    raise ValueError(f"{rel}: 'instantiate' needs a config.json in the folder")
                rep = instantiate_component(
                    sdir,
                    dst,
                    layers,
                    overrides=overrides.get(rel),
                    class_resolver=class_resolver,
                    **kw,
                )
                instantiated = True
            elif mode == "synthesize":
                rep = synthesize_component(sdir, dst, layers, counts(rel), **kw)
            else:
                raise ValueError(f"unknown mode {mode!r} for {rel}")
            reports.append(rep)
        for fn in files:
            path = os.path.join(sdir, fn)
            relfn = os.path.normpath(os.path.join(rel, fn))
            if fn.endswith(".safetensors.index.json"):
                indexes.append(relfn)
                continue
            if fn == "config.json" and not instantiated:
                cfg = _load_json(path)
                if cfg is not None:  # every config gets the same layer-count shrink as the weights
                    with open(os.path.join(dst, fn), "w") as f:
                        json.dump(shrink_layer_counts(cfg, layers)[0], f, indent=2)
                    continue
            if instantiated and fn == "config.json":
                continue
            if fn.endswith(WEIGHT_EXTS) or os.path.getsize(path) > copy_limit_mb * 2**20:
                continue
            shutil.copy2(path, os.path.join(dst, fn))
            copied.append(relfn)
    for relfn in indexes:
        _rewrite_index(src, out, relfn)
    manifest = {
        "source": os.path.abspath(src),
        "layers": layers,
        "seed": seed,
        "components": {
            r.name: {
                "mode": r.mode,
                "params": r.params,
                "tensors": r.tensors,
                "layer_lists": r.layer_lists,
                "config_layer_counts": r.config_changes,
            }
            for r in reports
        },
        "copied_files": len(copied),
        "indexes": indexes,
    }
    with open(os.path.join(out, "tiny_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


def _rewrite_index(src: str, out: str, relfn: str) -> None:
    """Point a ``*.safetensors.index.json`` at the tiny files: each referenced shard becomes the single
    file written in that shard's folder, and only tensors that still exist are listed."""
    with open(os.path.join(src, relfn)) as f:
        index = json.load(f)
    base = os.path.dirname(relfn)
    from safetensors import safe_open

    present: dict[str, dict[str, str]] = {}  # folder -> {tensor name: tiny file name}
    weight_map = {}
    for key, shard in index.get("weight_map", {}).items():
        folder = os.path.normpath(os.path.join(base, os.path.dirname(shard)))
        if folder not in present:
            names: dict[str, str] = {}
            tiny_dir = os.path.join(out, folder)
            for p in weight_files(tiny_dir) if os.path.isdir(tiny_dir) else []:
                with safe_open(p, "pt") as sf:
                    names.update(dict.fromkeys(sf.keys(), os.path.basename(p)))
            present[folder] = names
        fname = present[folder].get(key)
        if fname is not None:
            weight_map[key] = os.path.normpath(os.path.join(os.path.dirname(shard), fname))
    total = sum(
        os.path.getsize(p) for folder in present for p in weight_files(os.path.join(out, folder))
    )
    index["weight_map"] = dict(sorted(weight_map.items()))
    index.setdefault("metadata", {})["total_size"] = total
    with open(os.path.join(out, relfn), "w") as f:
        json.dump(index, f, indent=2)


def compare_layout(tiny: str, src: str, component: str = ".") -> dict[str, Any]:
    """Compare a tiny component against the real one by block-index-normalised parameter names and
    dtypes. ``ok`` means the same set of names with the same dtypes."""
    a = {normalize_key(k): dt for k, (_, dt) in read_headers(os.path.join(tiny, component)).items()}
    b = {normalize_key(k): dt for k, (_, dt) in read_headers(os.path.join(src, component)).items()}
    missing = sorted(set(b) - set(a))
    extra = sorted(set(a) - set(b))
    dtype_diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    return {
        "ok": not (missing or extra or dtype_diff),
        "missing": missing,
        "extra": extra,
        "dtype_mismatch": dtype_diff,
    }


def _parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mode", action="append", default=[], metavar="COMPONENT=synthesize|instantiate"
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="COMPONENT.key=value",
        help="config override for an instantiated component (JSON value)",
    )
    ap.add_argument("--copy-limit-mb", type=float, default=64.0)
    args = ap.parse_args(argv)
    modes = dict(m.split("=", 1) for m in args.mode)
    overrides: dict[str, dict[str, Any]] = {}
    for s in args.set:
        lhs, value = s.split("=", 1)
        comp, key = lhs.split(".", 1)
        overrides.setdefault(comp, {})[key] = _parse_value(value)
    manifest = make_tiny_checkpoint(
        args.src,
        args.out,
        layers=args.layers,
        modes=modes,
        overrides=overrides,
        seed=args.seed,
        copy_limit_mb=args.copy_limit_mb,
    )
    for comp in manifest["components"]:
        cmp = compare_layout(args.out, args.src, comp)
        manifest["components"][comp]["layout_ok"] = cmp["ok"]
    print(json.dumps(manifest["components"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
