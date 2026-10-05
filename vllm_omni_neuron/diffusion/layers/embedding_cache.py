# SPDX-License-Identifier: Apache-2.0
"""Text-encoder phase with a prompt-embedding cache.

Run the text encoder as its own phase so its weights never have to co-reside with the DiT, and
cache prompt embeddings on the host so repeated prompts (CFG negatives, benchmarks, livestream /
interactive prompts, robot instructions) skip it entirely.

* :class:`PromptEmbeddingCache` -- bounded LRU of host (CPU) tensors keyed on the encoder identity,
  the prompt and the encode settings, with an optional on-disk layer. Values are stored detached on
  CPU, so a cached embedding never pins device memory; callers move them to the device per request.
* :class:`TextEncoderPhase` -- wraps an ``encode(prompts) -> tensor | tuple`` callable: serves hits
  from the cache, encodes only the misses in one batched call, and can release the encoder (drop its
  weights / free its cores) after a precompute pass via a user callback, reloading lazily on a miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

import torch

logger = logging.getLogger(__name__)

Embedding = torch.Tensor | tuple[torch.Tensor, ...]


def _to_host(value: Embedding) -> Embedding:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous()
    return tuple(_to_host(v) for v in value)


def _nbytes(value: Embedding) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    return sum(_nbytes(v) for v in value)


def embedding_key(
    encoder_id: str, prompt: str | Sequence[int], settings: dict[str, Any] | None = None
) -> str:
    """Key for one prompt: encoder identity (model path/revision + dtype), the prompt text or token
    ids, and any encode settings that change the output (max length, padding bucket, template...)."""
    payload = json.dumps(
        {
            "enc": encoder_id,
            "p": prompt if isinstance(prompt, str) else list(prompt),
            "s": settings or {},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


class PromptEmbeddingCache:
    """LRU of host embeddings bounded by entry count and total bytes, plus an optional disk layer."""

    def __init__(
        self, max_entries: int = 256, max_bytes: int = 4 << 30, cache_dir: str | None = None
    ) -> None:
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.cache_dir = cache_dir or ""
        self._lru: OrderedDict[str, Embedding] = OrderedDict()
        self._bytes = 0
        self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._lru)

    def __contains__(self, key: str) -> bool:
        return key in self._lru or bool(self._path(key) and os.path.exists(self._path(key)))

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"emb_{key}.pt") if self.cache_dir else ""

    def get(self, key: str) -> Embedding | None:
        value = self._lru.get(key)
        if value is not None:
            self._lru.move_to_end(key)
            self.hits += 1
            return value
        path = self._path(key)
        if path and os.path.exists(path):
            value = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(value, list):
                value = tuple(value)
            self._insert(key, value)
            self.hits += 1
            return value
        self.misses += 1
        return None

    def put(self, key: str, value: Embedding) -> Embedding:
        value = _to_host(value)
        self._insert(key, value)
        path = self._path(key)
        if path and not os.path.exists(path):
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            torch.save(list(value) if isinstance(value, tuple) else value, tmp)
            os.replace(tmp, path)
        return value

    def _insert(self, key: str, value: Embedding) -> None:
        if key in self._lru:
            self._bytes -= _nbytes(self._lru.pop(key))
        self._lru[key] = value
        self._bytes += _nbytes(value)
        while self._lru and (len(self._lru) > self.max_entries or self._bytes > self.max_bytes):
            if len(self._lru) == 1:  # a single entry larger than max_bytes is still kept
                break
            _, old = self._lru.popitem(last=False)
            self._bytes -= _nbytes(old)

    def clear(self) -> None:
        self._lru.clear()
        self._bytes = 0


class TextEncoderPhase:
    """Cached, batched text encoding with optional release of the encoder between phases.

    ``encode_fn(prompts)`` returns a tensor batched on dim 0 (or a tuple of such tensors, e.g.
    ``(embeds, mask)``). ``load_fn`` (optional) is called before encoding when the encoder is not
    loaded; ``release_fn`` (optional) is called by :meth:`release`.
    """

    def __init__(
        self,
        encode_fn: Callable[[list[str]], Embedding],
        encoder_id: str,
        *,
        settings: dict[str, Any] | None = None,
        cache: PromptEmbeddingCache | None = None,
        load_fn: Callable[[], None] | None = None,
        release_fn: Callable[[], None] | None = None,
        loaded: bool = True,
    ) -> None:
        self.encode_fn = encode_fn
        self.encoder_id = encoder_id
        self.settings = settings or {}
        self.cache = cache if cache is not None else PromptEmbeddingCache()
        self.load_fn, self.release_fn = load_fn, release_fn
        self.loaded = loaded
        self.encoder_calls = 0

    def key(self, prompt: str) -> str:
        return embedding_key(self.encoder_id, prompt, self.settings)

    def _ensure_loaded(self) -> None:
        if not self.loaded:
            if self.load_fn is None:
                raise RuntimeError("text encoder was released and no load_fn was given")
            self.load_fn()
            self.loaded = True

    def release(self) -> None:
        if self.loaded and self.release_fn is not None:
            self.release_fn()
            self.loaded = False

    def encode(self, prompts: Sequence[str]) -> list[Embedding]:
        """Per-prompt host embeddings (each with the batch dim of 1 kept), cache hits first."""
        keys = [self.key(p) for p in prompts]
        out: list[Embedding | None] = [self.cache.get(k) for k in keys]
        miss = [i for i, v in enumerate(out) if v is None]
        if miss:
            uniq: dict[str, int] = {}
            for i in miss:
                uniq.setdefault(keys[i], i)
            self._ensure_loaded()
            batch = self.encode_fn([prompts[i] for i in uniq.values()])
            self.encoder_calls += 1
            fresh = {k: self.cache.put(k, _slice(batch, j)) for j, k in enumerate(uniq)}
            for i in miss:
                out[i] = fresh[keys[i]]
        return out  # type: ignore[return-value]

    def precompute(self, prompts: Sequence[str], release_after: bool = False) -> None:
        self.encode(prompts)
        if release_after:
            self.release()


def _slice(value: Embedding, j: int) -> Embedding:
    if isinstance(value, torch.Tensor):
        return value[j : j + 1]
    return tuple(_slice(v, j) for v in value)


def stack_embeddings(
    items: Sequence[Embedding], device: torch.device | str | None = None
) -> Embedding:
    """Concatenate per-prompt embeddings back into a batch (and optionally move to ``device``)."""
    first = items[0]
    if isinstance(first, torch.Tensor):
        out = torch.cat(list(items), dim=0)
        return out.to(device) if device is not None else out
    return tuple(stack_embeddings([it[k] for it in items], device) for k in range(len(first)))
