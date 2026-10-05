# SPDX-License-Identifier: Apache-2.0
"""Neuron LTX-2.5 served pipeline: the diffusers CPU pipeline (text encoder, connectors,
scheduler, VAE, vocoder, all unmodified upstream math) with ``transformer`` swapped for
:class:`NeuronLTX2Transformer` -- the same pattern as Cosmos3-Edge (reuse the pure-math
pipeline, swap only the device-touching components), not a subclass of vLLM-Omni's
``LTX2Pipeline``: that module imports ``cache_dit`` unconditionally (not a dependency of this
plugin) and its transformer is a CUDA/vLLM-TP implementation with no Neuron path.

Engine contract this class implements for :class:`DiffusionWorker`:
``__init__(od_config, prefix)``, ``load_weights(weights=None)``, ``compile(...)``,
``forward(req: DiffusionRequestBatch, ...) -> DiffusionOutput``. ``supports_request_batch = False``
(like upstream's own LTX2Pipeline): the engine calls forward with one request at a time.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

PIPELINE_REGISTRY = [
    {
        # model_index.json `_class_name` of Lightricks/LTX-2.5-Diffusers
        "model_arch": "LTX2Pipeline",
        "class_name": "NeuronLTX25Pipeline",
        "pre_process_func_name": "get_ltx25_pre_process_func",
        "post_process_func_name": "get_ltx25_post_process_func",
    },
]


def host_sync(group) -> None:
    """Barrier over a host (gloo) group as a CPU all-reduce. ``dist.barrier`` allocates its token
    tensor on ``torch._C._get_accelerator()``, which under the Neuron Lite runtime is a PrivateUse1
    device without ``isAvailable`` hooks and raises ``NotImplementedError``."""
    import torch.distributed as dist

    dist.all_reduce(torch.zeros(1, dtype=torch.int32), group=group)


def bootstrap_collective(stage_cpu_group, tp_group, tp_size, device, dtype, compile_fn) -> None:
    """Every stage rank: host barrier, one ``compile_fn``-compiled all-reduce over ``tp_group`` on
    ``device`` (checked), host barrier. See ``NeuronLTX25Pipeline._bootstrap_device_collectives``."""
    import torch.distributed as dist

    host_sync(stage_cpu_group)  # all ranks enter together

    def reduce_one(x):
        dist.all_reduce(x, group=tp_group)
        return x

    out = compile_fn(reduce_one)(torch.ones(1, 8, dtype=dtype).to(device)).cpu()
    if float(out[0, 0]) != float(tp_size):
        raise RuntimeError(f"device all-reduce bootstrap: {out[0, 0]} != {tp_size}")
    host_sync(stage_cpu_group)


@contextmanager
def host_threads(n: int):
    """Run host-CPU work with up to ``n`` torch threads, restoring the caller's setting after.

    The Omni diffusion worker pins torch to 1 thread under Lite (``_limit_lite_worker_threads``).
    The host may be shared with other workloads (measured on a 192-vCPU host at a 1-min load of
    120-160), and OpenMP threads spin at barriers, so asking for more threads than there are idle
    cores makes a host stage slower, not faster (standalone vocoder at load ~140: 6.6-8.7 s with
    24 threads, 16.9 s with 48). The count is capped at the cores actually free (this process's
    affinity minus the 1-min load), floor 4, but only with ``LTX25_HOST_THREADS_ADAPTIVE=1``:
    the CPU kernels' results depend on the thread count (the connectors differ by ~0.3% rel-L2
    between 1 and 4 threads, and the distilled denoise amplifies that to ~10% on the video), so by
    default the count is exactly ``n`` and a request's output does not depend on the host load.
    Yields the count used."""
    k = n
    if os.environ.get("LTX25_HOST_THREADS_ADAPTIVE", "0") == "1":
        cpus = len(os.sched_getaffinity(0))
        k = max(4, min(n, int(cpus - os.getloadavg()[0])))
    prev = torch.get_num_threads()
    torch.set_num_threads(k)
    try:
        yield k
    finally:
        torch.set_num_threads(prev)


def get_ltx25_pre_process_func(od_config):
    def pre_process_func(prompts, sampling_params=None, **_kwargs):
        return prompts

    return pre_process_func


def get_ltx25_post_process_func(od_config):
    def post_process_func(output, output_type: str = "np", sampling_params=None):
        return output

    return post_process_func


class _ConditioningFromRank0(nn.Module):
    """Connectors stand-in on a non-output rank: the conditioning arrives from rank 0 (see
    ``NeuronLTX25Pipeline._shared_text_conditioning``) and is returned by the forward wrapper."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("text conditioning is computed on rank 0 and broadcast")


class NeuronLTX25Pipeline(nn.Module):
    """Positive-only-guidance distilled T2V, served via the vLLM-Omni diffusion engine."""

    supports_request_batch = False
    _dit_modules = ["transformer"]
    _encoder_modules = ["text_encoder"]
    _vae_modules = ["vae", "audio_vae"]
    dummy_run_num_frames = 9  # smallest shape that still exercises the real graphs on warmup

    def __init__(self, *, od_config, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = torch.device("cpu")  # host-resident text enc / VAE / vocoder; DiT is on-core
        self.dtype = getattr(od_config, "dtype", torch.bfloat16)
        model_config = od_config.model_config or {}
        self.blocks_per_graph = int(model_config.get("blocks_per_graph", 4))
        self.transformer_subfolder = model_config.get("transformer_subfolder", "transformer")

        model_path = self._resolve_model_dir(od_config.model, self.transformer_subfolder)
        self.model_dir = model_path
        # Stage rank, for the output-rank-only stages (rank 0 encodes the prompt, merges the VAE
        # tiles, runs the audio decode and returns the output).
        self.rank = self._tp_rank()
        prompt_cache = os.environ.get("LTX25_PROMPT_CACHE", "1") == "1"
        # With the conditioning broadcast from rank 0, the other ranks never run the text
        # encoder (12 B) or the connectors (3.2 B): they do not load them (~30 GB of host RAM
        # per rank, which at 16-64 ranks does not fit the host).
        # Gemma text tower on the NeuronCores (text_encoder.py), TP-sharded over rank 0's TP
        # group; LTX25_TEXT_ENCODER_DEVICE=0 keeps transformers' model on the host (rank 0).
        # (needs the prompt cache path, which is where the conditioning is computed and shared)
        self.text_on_device = (
            prompt_cache and os.environ.get("LTX25_TEXT_ENCODER_DEVICE", "1") == "1"
        )
        self._device_text = None
        self._build_components(
            model_path,
            lean=prompt_cache and self.rank != 0,
            skip_text_encoder=self.text_on_device,
        )
        self._weights_loaded = False
        # Prompt-conditioning cache (shared layers/embedding_cache.py): a repeated prompt skips the
        # Gemma text encoder and the text connectors. Host tensors (the connector outputs), LRU,
        # optional disk layer (LTX25_EMB_CACHE_DIR). LTX25_PROMPT_CACHE=0 disables it.
        self._emb_cache = None
        if os.environ.get("LTX25_PROMPT_CACHE", "1") == "1":
            from vllm_omni_neuron.diffusion.layers.embedding_cache import PromptEmbeddingCache

            self._emb_cache = PromptEmbeddingCache(
                max_entries=64, cache_dir=os.environ.get("LTX25_EMB_CACHE_DIR") or None
            )
        self._encoder_id = f"{os.path.abspath(model_path)}/text_encoder:{self.dtype}"
        self.host_threads = int(os.environ.get("LTX25_HOST_THREADS", "24"))
        # The pipeline's own connector call returns the cached conditioning of the request in
        # flight (set in forward) instead of recomputing it.
        self._connector_out = None
        if self.connectors is not None:
            self._connectors_forward = self.connectors.forward

            def connectors_forward(*args, _orig=self._connectors_forward, **kw):
                if self._connector_out is not None:
                    return self._connector_out
                return _orig(*args, **kw)

            self.connectors.forward = connectors_forward

    def _cached_text_conditioning(self, prompt: str, max_sequence_length: int = 1024):
        """The DiT's text conditioning for ``prompt`` -- the connector outputs
        ``(video_embeds, audio_embeds, attention_mask)`` -- from the cache, or one Gemma call plus
        one connector call (diffusers' own ``encode_prompt`` and ``connectors``, so the tensors
        are exactly what ``__call__`` would have computed). Returns ``(conditioning, hit)``.

        The connectors (3.2 B parameters) are cached too, not only the Gemma output: they are a
        pure function of the prompt, and on the 1-thread worker they cost ~14 s per request
        (1.3 s at 24 threads). The cached value is ~13 MB per prompt instead of the ~385 MB
        49-layer Gemma hidden-state stack."""
        from vllm_omni_neuron.diffusion.layers.embedding_cache import embedding_key

        key = embedding_key(
            self._encoder_id, prompt, {"max_len": max_sequence_length, "out": "connectors"}
        )
        hit = self._emb_cache.get(key)
        if hit is not None:
            return tuple(hit), True
        with torch.no_grad(), host_threads(getattr(self, "host_threads", 24)):
            embeds, mask = self._encode_prompt_embeds(prompt, max_sequence_length)
            tok = getattr(self._pipe, "tokenizer", None)
            side = getattr(tok, "padding_side", "left") if tok is not None else "left"
            cond = self._run_connectors(embeds, mask, padding_side=side)
        cond = self._emb_cache.put(key, tuple(t.contiguous() for t in cond))
        return tuple(cond), False

    def _encode_prompt_embeds(self, prompt: str, max_sequence_length: int):
        """The Gemma hidden-state stack ``(prompt_embeds, mask)``: on the NeuronCores, or with
        diffusers' ``encode_prompt`` (host)."""
        if getattr(self, "_device_text", None) is not None:
            tok = self._pipe.tokenizer
            return self._device_text.encode_prompt(tok, prompt, max_sequence_length)
        embeds, mask, _, _ = self._pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=torch.device("cpu"),
        )
        return embeds, mask

    def _run_connectors(self, embeds, mask, padding_side="left"):
        fn = getattr(self, "_connectors_forward", None) or self._pipe.connectors
        return fn(embeds, mask, padding_side=padding_side)

    def _bootstrap_device_collectives(self) -> None:
        """Every rank of a multi-rank stage runs one small compiled TP all-reduce, once, before
        the first request's text conditioning.

        The Neuron runtime builds its device communicator lazily, at the first collective any
        rank executes, and that bootstrap needs EVERY rank of the stage. With the device text
        encoder under CP > 1 the first collective is the encoder's, which only rank 0's TP group
        runs, while the other ranks wait for its broadcast on the host: the bootstrap never
        completes (``16-ranks bootstrap: rank 0 awaiting root parameters``) and the host
        broadcast times out. One collective on all ranks first makes the encoder's run on the
        already-built communicator."""
        if getattr(self, "_device_comm_ready", False):
            return
        self._device_comm_ready = True
        group = self._stage_group()
        dev = torch.device(getattr(self.od_config, "device", None) or self._dit_device())
        if group is None or dev.type == "cpu":
            return
        from vllm.distributed.parallel_state import get_tp_group

        tp = get_tp_group()
        if tp.world_size == 1:
            return
        from vllm_neuron.envs import get_compile_backend_name

        backend = get_compile_backend_name()
        bootstrap_collective(
            group.cpu_group,
            tp.device_group,
            tp.world_size,
            dev,
            self.dtype,
            lambda f: torch.compile(f, backend=backend, fullgraph=True),
        )

    def _shared_text_conditioning(self, prompt: str, max_sequence_length: int = 1024):
        """Like :meth:`_cached_text_conditioning`, but under TP only rank 0 runs the encoder.

        Every TP rank needs the same conditioning for the sharded DiT. Encoding on all of them
        put three single-threaded Gemma + connector runs on ranks 1-3 that rank 0's first DiT
        collective then waited for. Now: the ranks agree (CPU all-reduce) whether all of them
        have the prompt cached; if so each uses its own copy; otherwise rank 0 encodes (threaded)
        and broadcasts the tensors over the stage group's CPU (gloo) group, and every rank caches
        them. Returns ``(conditioning, status)``."""
        from vllm_omni_neuron.diffusion.layers.embedding_cache import embedding_key

        group = NeuronLTX25Pipeline._stage_group()
        if group is None:
            cond, hit = self._cached_text_conditioning(prompt, max_sequence_length)
            return cond, "hit" if hit else "miss"
        import torch.distributed as dist

        key = embedding_key(
            self._encoder_id, prompt, {"max_len": max_sequence_length, "out": "connectors"}
        )
        have = torch.tensor([1 if key in self._emb_cache else 0], dtype=torch.int32)
        dist.all_reduce(have, op=dist.ReduceOp.MIN, group=group.cpu_group)
        if int(have.item()) == 1:
            return tuple(self._emb_cache.get(key)), "hit"
        names = ("video", "audio", "mask")
        if group.rank_in_group == 0:
            cond, _ = self._cached_text_conditioning(prompt, max_sequence_length)
            group.broadcast_tensor_dict(dict(zip(names, cond, strict=True)), src=0)
        else:
            if getattr(self, "_device_text", None) is not None:  # rank 0's TP peers: its shards
                with torch.no_grad():
                    self._device_text.encode_prompt(
                        self._pipe.tokenizer, prompt, max_sequence_length
                    )
            d = group.broadcast_tensor_dict(None, src=0)
            cond = tuple(self._emb_cache.put(key, tuple(d[n] for n in names)))
        return cond, "miss-broadcast"

    @staticmethod
    def _resolve_model_dir(model: str, transformer_subfolder: str) -> str:
        """A local directory, or a Hugging Face repo id downloaded to the local HF cache (only the
        components the distilled T2V pipeline loads)."""
        if os.path.isdir(model):
            return model
        from huggingface_hub import snapshot_download

        patterns = [
            "model_index.json",
            "scheduler/*",
            "tokenizer/*",
            "text_encoder/*",
            "connectors/*",
            "vae/*",
            "audio_vae/*",
            "vocoder/*",
            f"{transformer_subfolder}/*",
        ]
        return snapshot_download(model, allow_patterns=patterns)

    @staticmethod
    def _stage_group():
        """The stage's world GroupCoordinator (every TP x CP rank) when it has more than one
        rank, else None. Rank 0 of it is the output rank."""
        g = None
        try:
            from vllm_omni.diffusion.distributed.parallel_state import get_world_group

            g = get_world_group()
        except (AssertionError, ImportError, AttributeError):
            try:
                from vllm.distributed.parallel_state import get_world_group

                g = get_world_group()
            except (AssertionError, ImportError):
                g = None
        return g if g is not None and g.world_size > 1 else None

    @classmethod
    def _tp_rank(cls) -> int:
        """This rank's index in the stage group (0 = the output rank)."""
        g = cls._stage_group()
        return int(g.rank_in_group) if g is not None else 0

    # -- construction ---------------------------------------------------------------------
    def _build_components(
        self, model_path: str, lean: bool = False, skip_text_encoder: bool = False
    ) -> None:
        """Build every component except the DiT from the diffusers pipeline (CPU, host dtype);
        the DiT is built separately in ``load_weights`` once this rank's device is known.
        ``lean``: skip the text encoder and the connectors (a non-output rank that receives the
        conditioning from rank 0); a stub stands in for the connectors. ``skip_text_encoder``:
        skip only the text encoder (it is built on the NeuronCores in ``load_weights``)."""
        import inspect
        import json

        from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformer3DModel
        from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline

        with open(os.path.join(model_path, self.transformer_subfolder, "config.json")) as f:
            tf_cfg = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        supported = inspect.signature(LTX2VideoTransformer3DModel.__init__).parameters
        placeholder = LTX2VideoTransformer3DModel(
            **{**{k: v for k, v in tf_cfg.items() if k in supported}, "num_layers": 0}
        ).to(self.dtype)
        extra = {"text_encoder": None, "connectors": None} if lean else {}
        if skip_text_encoder:
            extra["text_encoder"] = None
        pipe = LTX2Pipeline.from_pretrained(
            model_path, torch_dtype=self.dtype, transformer=placeholder, **extra
        )
        del placeholder
        if lean:
            pipe.connectors = _ConditioningFromRank0()
        self._pipe = pipe
        for name in (
            "tokenizer",
            "text_encoder",
            "connectors",
            "vae",
            "audio_vae",
            "vocoder",
            "scheduler",
        ):
            setattr(self, name, getattr(pipe, name, None))
        self.vae_spatial_compression_ratio = pipe.vae_spatial_compression_ratio
        self.vae_temporal_compression_ratio = pipe.vae_temporal_compression_ratio

    # -- engine surface ---------------------------------------------------------------------
    def load_weights(self, weights=None) -> None:
        """Build + load the real DiT on this rank's device (TP shard via vLLM's TP group, if any).

        Weights are not routed through the engine's ``weights`` iterator (same as Cosmos3-Edge):
        the DiT does its own sharded safetensors load keyed on the TP group's rank/size.
        """
        from vllm_omni_neuron.diffusion.models.ltx2.ltx2_transformer import NeuronLTX2Transformer

        dev = getattr(self.od_config, "device", None) or self._dit_device()
        transformer_dir = os.path.join(self.model_dir, self.transformer_subfolder)
        t0 = time.time()
        self.transformer = NeuronLTX2Transformer.from_dir(
            transformer_dir,
            dtype=self.dtype,
            blocks_per_graph=self.blocks_per_graph,
            device=dev,
        )
        self._pipe.transformer = self.transformer
        # plain print, not logger.info: this module's logger sits outside vLLM's dictConfig'd
        # hierarchy (vllm_omni_neuron.*, not vllm.*/vllm_omni.*) and silently drops at the root
        # logger's default WARNING level inside the engine's worker subprocess -- confirmed
        # missing from a served-stage job log despite the pipeline running correctly.
        print(f"[NeuronLTX25Pipeline] DiT loaded in {time.time() - t0:.1f}s ({dev})", flush=True)
        self._weights_loaded = True
        # Device-tiled VAE decode: one compiled (TILE_H, TILE_W) tile graph serves every tile.
        # With more than one rank (TP x CP) the tiles are spread over all of them (each on its own
        # core; rank 0 merges);
        # LTX2_VAE_PARALLEL=0 decodes every tile on rank 0. LTX2_VAE_DEVICE=0 keeps the CPU decode
        # (the fallback / reference path).
        self._tiled_vae = None
        dev = torch.device(dev)
        group = self._stage_group()
        parallel = group is not None and os.environ.get("LTX2_VAE_PARALLEL", "1") == "1"
        if (
            (self._tp_rank() == 0 or parallel)
            and dev.type != "cpu"
            and os.environ.get("LTX2_VAE_DEVICE", "1") == "1"
        ):
            from vllm_omni_neuron.diffusion.models.ltx2.vae_tiling import TiledLTX2VideoDecoder

            self._tiled_vae = TiledLTX2VideoDecoder(
                self.vae, dev, group=group if parallel else None
            )
            if self._tp_rank() == 0:
                self._tiled_vae.install(self.vae)
        # The vocoder runs as time spans, one per rank, on the first LTX25_VOCODER_SPANS (4) ranks'
        # host CPUs (audio_chunks.py). Its own gloo group: rank 0 drives it from the audio thread
        # while the video decode uses the stage group. LTX25_VOCODER_PARALLEL=0: one call, rank 0.
        self._chunked_vocoder = None
        if group is not None and os.environ.get("LTX25_VOCODER_PARALLEL", "1") == "1":
            n = min(group.world_size, int(os.environ.get("LTX25_VOCODER_SPANS", "4")))
            if n > 1:
                import torch.distributed as dist

                from vllm_omni_neuron.diffusion.models.ltx2.audio_chunks import ChunkedVocoder

                ranks = list(group.ranks[:n])
                span_group = dist.new_group(ranks=ranks, backend="gloo")  # every rank calls it
                if group.rank_in_group < n:
                    self._chunked_vocoder = ChunkedVocoder(n, ranks=ranks, group=span_group)
                    if group.rank_in_group != 0:
                        self.vocoder.float()
        self._build_text_encoder(dev)
        if self._tp_rank() == 0:
            self._install_host_stages()

    def _build_text_encoder(self, dev: torch.device) -> None:
        """Device Gemma on the ranks of rank 0's TP group (the other TP groups get the
        conditioning by broadcast); host Gemma on rank 0 when the flag is off or the DiT is on
        the CPU."""
        if not self.text_on_device:
            return
        te_dir = os.path.join(self.model_dir, "text_encoder")
        if dev.type == "cpu":
            if self._tp_rank() == 0:  # host fallback: the transformers model, as diffusers loads it
                from transformers.models.gemma4_unified.modeling_gemma4_unified import (
                    Gemma4UnifiedForConditionalGeneration,
                )

                self.text_encoder = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                    te_dir, torch_dtype=self.dtype
                ).eval()
                self._pipe.text_encoder = self.text_encoder
            return
        stage = self._stage_group()
        if stage is not None:
            from vllm.distributed.parallel_state import get_tp_group

            if stage.ranks[0] not in get_tp_group().ranks:
                return
        from vllm_omni_neuron.diffusion.models.ltx2.text_encoder import NeuronGemmaTextEncoder

        t0 = time.time()
        self._device_text = NeuronGemmaTextEncoder(te_dir, dev, dtype=self.dtype)
        print(
            f"[NeuronLTX25Pipeline] text encoder loaded in {time.time() - t0:.1f}s ({dev}, "
            f"{self._device_text.num_local_bytes() / 2**30:.2f} GiB per core)",
            flush=True,
        )

    def _install_host_stages(self) -> None:
        """Rank 0: run the host-CPU stages with up to ``LTX25_HOST_THREADS`` torch threads (default
        24, see :func:`host_threads`): the Gemma text encoder and connectors (cache miss), the
        tile merge of the video decode, the frame post-processing, the audio VAE and the BWE
        vocoder. Each stage's thread count, host load and wall time land in ``host_stage_log``.

        The Omni diffusion worker pins torch to 1 thread under Lite (diffusion_worker.py
        ``_limit_lite_worker_threads``): in the served worker the vocoder took 45 s where the same
        call standalone takes 4.4 s. Non-output ranks never reach these stages.

        The vocoder also runs in fp32 (the official LTX pipeline and vLLM-Omni's
        ``_run_ltx_vocoder`` do; measured bf16-vs-fp32 waveform rel-L2 0.179 at the real 121-frame
        mel): its weights/buffers are upcast once and the bf16 mel input cast per call.
        """
        n = self.host_threads
        self.host_stage_log: dict = {}

        def threaded(fn, name, cast_fp32=False):
            def wrapper(*args, **kwargs):
                t0, c0 = time.perf_counter(), time.process_time()
                load = os.getloadavg()[0]
                with host_threads(n) as k:
                    try:
                        if cast_fp32:
                            args = tuple(
                                a.float() if torch.is_tensor(a) and a.is_floating_point() else a
                                for a in args
                            )
                        return fn(*args, **kwargs)
                    finally:
                        self.host_stage_log[name] = {
                            "threads": k,
                            "load1": round(load, 1),
                            "wall_s": round(time.perf_counter() - t0, 3),
                            "cpu_s": round(time.process_time() - c0, 2),
                        }

            return wrapper

        if self.text_encoder is not None:
            self.text_encoder.forward = threaded(self.text_encoder.forward, "text_encoder")
        if self.vae is not None:
            self.vae.decode = threaded(self.vae.decode, "vae_decode")
        vp = getattr(self._pipe, "video_processor", None)
        if vp is not None:
            vp.postprocess_video = threaded(vp.postprocess_video, "postprocess_video")
        if self.audio_vae is not None:
            self.audio_vae.decode = threaded(self.audio_vae.decode, "audio_vae_decode")
        if self.vocoder is not None:
            if os.environ.get("LTX25_VOCODER_FP32", "1") == "1":
                self.vocoder.float()
                self.vocoder.forward = threaded(self.vocoder.forward, "vocoder", cast_fp32=True)
            else:
                self.vocoder.forward = threaded(self.vocoder.forward, "vocoder")
        if getattr(self, "_chunked_vocoder", None) is not None:
            spans_fn, chunked = self.vocoder.forward, self._chunked_vocoder
            self.vocoder.forward = lambda mel, _f=spans_fn, _c=chunked: _c(_f, mel)
        if os.environ.get("LTX25_OVERLAP_AUDIO", "1") == "1":
            self._install_audio_overlap()

    def _install_audio_overlap(self) -> None:
        """Rank 0: decode the audio (audio VAE + vocoder, host) while the video VAE runs.

        The diffusers pipeline unpacks the audio latents before the video decode and runs the
        audio decode after it, sequentially. The video decode is device tiles plus a host merge
        and the audio decode is host CPU, so when the video decode starts, the audio decode of the
        already-unpacked latents is started on a worker thread, with exactly the calls the
        pipeline would make (``audio_vae.decode(latents.to(audio_vae.dtype))`` then
        ``vocoder(mel)``); the pipeline's own two calls then return those results.
        ``LTX25_OVERLAP_AUDIO=0`` keeps them sequential."""
        import threading

        pipe, state = self._pipe, {}
        unpack = pipe._unpack_audio_latents  # staticmethod, called as self._unpack_audio_latents
        vae_decode, audio_decode, vocoder_fwd = (
            self.vae.decode,
            self.audio_vae.decode,
            self.vocoder.forward,
        )

        def unpack_audio(*args, **kwargs):
            state.clear()
            state["latents"] = out = unpack(*args, **kwargs)
            return out

        def run_audio(latents):
            try:
                with torch.no_grad():
                    mel = audio_decode(latents.to(self.audio_vae.dtype), return_dict=False)[0]
                    state["result"] = (mel, vocoder_fwd(mel))
            except BaseException as exc:  # noqa: BLE001  re-raised in the pipeline's thread
                state["error"] = exc

        def video_decode(*args, **kwargs):
            lat = state.get("latents")
            if lat is not None and "thread" not in state:
                state["thread"] = t = threading.Thread(target=run_audio, args=(lat,), daemon=True)
                t.start()
            return vae_decode(*args, **kwargs)

        def _result():
            t = state.pop("thread", None)
            if t is None:
                return None
            t.join()
            if "error" in state:
                raise state.pop("error")
            return state.pop("result")

        def audio_vae_decode(z, *args, **kwargs):
            res = _result()
            if res is None:
                return audio_decode(z, *args, **kwargs)
            state["pending_wav"] = res[1]
            mel = res[0]
            if not kwargs.get("return_dict", True):
                return (mel,)
            from diffusers.models.autoencoders.vae import DecoderOutput

            return DecoderOutput(sample=mel)

        def vocoder_forward(mel, *args, **kwargs):
            wav = state.pop("pending_wav", None)
            return wav if wav is not None else vocoder_fwd(mel, *args, **kwargs)

        pipe._unpack_audio_latents = unpack_audio
        self.vae.decode = video_decode
        self.audio_vae.decode = audio_vae_decode
        self.vocoder.forward = vocoder_forward

    @staticmethod
    def _dit_device() -> torch.device:
        try:
            from vllm_omni.platforms import current_omni_platform

            return current_omni_platform.get_torch_device()
        except Exception:  # noqa: BLE001  fall back for a bare offline run
            return (
                torch.device("neuron", 0) if os.path.exists("/dev/neuron0") else torch.device("cpu")
            )

    def compile(self, *args, backend: str | None = None, **kwargs) -> "NeuronLTX25Pipeline":
        if backend is None:
            from vllm_neuron.envs import get_compile_backend_name

            backend = get_compile_backend_name()
        self.transformer.compile(backend)
        if getattr(self, "_device_text", None) is not None:
            self._device_text.compile(backend)
        if getattr(self, "_tiled_vae", None) is not None:
            self._tiled_vae.compile(backend)
        return self

    def to(self, *args, **kwargs):
        return self  # components already placed; DiT device is fixed at load_weights time

    class _StageTimer:
        """Wraps text_encoder / transformer / vae.decode / vocoder with perf_counter timers.

        The DiT is called once per denoise step, so its total is reported alongside a per-step
        average; everything else runs once per request."""

        def __init__(self, pipe: "NeuronLTX25Pipeline"):
            self.pipe, self.totals, self.counts, self._originals = pipe, {}, {}, {}
            self._wrap("text_encoder", pipe.text_encoder, "forward")
            self._wrap("connectors", pipe.connectors, "forward")
            self._wrap("transformer", pipe.transformer, "forward")
            self._wrap("vae_decode", pipe.vae, "decode")
            self._wrap("audio_vae_decode", pipe.audio_vae, "decode")
            self._wrap("vocoder", pipe.vocoder, "forward")

        def _wrap(self, name, obj, attr):
            if obj is None or not hasattr(obj, attr):
                return
            original = getattr(obj, attr)
            self._originals[(obj, attr)] = original
            self.totals[name] = 0.0
            self.counts[name] = 0

            def timed(*args, _orig=original, _name=name, **kw):
                t0 = time.perf_counter()
                out = _orig(*args, **kw)
                self.totals[_name] += time.perf_counter() - t0
                self.counts[_name] += 1
                return out

            setattr(obj, attr, timed)

        def report(self) -> dict:
            out = {}
            for name, total in self.totals.items():
                n = self.counts[name]
                out[name] = {
                    "total_s": round(total, 3),
                    "calls": n,
                    "per_call_s": round(total / n, 4) if n else None,
                }
            return out

        def restore(self) -> None:
            for (obj, attr), original in self._originals.items():
                setattr(obj, attr, original)

    def _time_stages(self) -> "NeuronLTX25Pipeline._StageTimer":
        return self._StageTimer(self)

    def forward(self, req, **overrides):
        """Engine entry point: ``req`` is a single-request :class:`DiffusionRequestBatch`
        (``supports_request_batch = False``). Positive-only guidance (``do_classifier_free_guidance``
        must be False / unset -- CFG/STG/modality-isolation are not supported by the Neuron DiT)."""
        from vllm_omni.diffusion.data import DiffusionOutput

        sp = req.sampling_params
        prompt = req.prompts[0]
        prompt = prompt if isinstance(prompt, str) else prompt.get("prompt", "")
        kwargs = dict(
            prompt=prompt,
            height=sp.height or 512,
            width=sp.width or 768,
            num_frames=sp.num_frames or 121,
            frame_rate=sp.frame_rate or 24.0,
            num_inference_steps=sp.num_inference_steps or 8,
            guidance_scale=1.0,
            stg_scale=0.0,
            modality_scale=1.0,
            audio_guidance_scale=1.0,
            audio_stg_scale=0.0,
            audio_modality_scale=1.0,
            use_cross_timestep=True,
            output_type=sp.output_type or "np",
            return_dict=True,
        )
        if sp.seed is not None:
            kwargs["generator"] = torch.Generator().manual_seed(sp.seed)
        elif sp.generator is not None:
            kwargs["generator"] = sp.generator
        kwargs.update(overrides)
        if kwargs.pop("do_classifier_free_guidance", False):
            raise NotImplementedError("CFG is not supported by the Neuron LTX-2.5 DiT yet")

        # Decode (VAE + vocoder) only on the output rank. Every TP rank runs the full sharded DiT
        # denoise (the collectives need all of them), but the VAE/vocoder are host-CPU and NOT
        # sharded -- running them on all 4 ranks at once just 4-way-thrashes the host (measured:
        # a VAE decode that is ~63s uncontended on one core balloons to ~390s under 4-way
        # contention). Non-output ranks stop at the latent (output_type="latent" returns before
        # decode), so only rank 0 pays the decode; the engine collects rank 0's output anyway.
        is_output_rank = self._tp_rank() == 0
        # tile-parallel VAE: the other ranks decode their share of the tiles after the denoise
        serve_vae = (
            not is_output_rank
            and kwargs.get("output_type") != "latent"
            and getattr(self, "_tiled_vae", None) is not None
            and self._tiled_vae.group is not None
        )
        # span-parallel vocoder: the first few ranks vocode one time span each
        serve_audio = (
            not is_output_rank
            and kwargs.get("output_type") != "latent"
            and getattr(self, "_chunked_vocoder", None) is not None
        )
        if not is_output_rank:
            kwargs["output_type"] = "latent"

        stages = (
            self._time_stages()
            if (os.environ.get("LTX25_TIME_STAGES") == "1" and is_output_rank)
            else None
        )
        prompt_cache = None
        t_req = time.perf_counter()
        if self._emb_cache is not None and isinstance(kwargs.get("prompt"), str):
            self._bootstrap_device_collectives()  # once; no-op on one rank or the CPU
            cond, prompt_cache = self._shared_text_conditioning(kwargs.pop("prompt"))
            # The pipeline still wants a prompt_embeds tensor (batch size, dtype); its connector
            # call returns the cached conditioning, so a [B, L, 1] placeholder stands in for the
            # Gemma hidden-state stack.
            kwargs["prompt_embeds"] = torch.zeros(1, cond[2].shape[1], 1, dtype=self.dtype)
            kwargs["prompt_attention_mask"] = cond[2]
            self._connector_out = cond
        t_cond = time.perf_counter() - t_req
        try:
            with torch.no_grad():
                out = self._pipe(**kwargs)
        finally:
            self._connector_out = None
        if serve_vae:
            self._tiled_vae.serve()
        if serve_audio:
            with host_threads(self.host_threads), torch.no_grad():
                self._chunked_vocoder.serve(lambda m: self.vocoder(m.float()))
        if stages is not None:
            print(
                f"[NeuronLTX25Pipeline] stage timings (s): {stages.report()} "
                f"prompt_cache={prompt_cache} "
                f"text_conditioning_s={t_cond:.3f} pipeline_s={time.perf_counter() - t_req:.3f} "
                f"host_threads={getattr(self, 'host_threads', None)} "
                f"worker_threads={torch.get_num_threads()} "
                f"host_stages={getattr(self, 'host_stage_log', None)}",
                flush=True,
            )
            stages.restore()
        if not is_output_rank:
            # Latent-only: nothing to return as a frame; the engine ignores non-output ranks.
            from vllm_omni.diffusion.data import DiffusionOutput as _DO

            return _DO(output={"video": None})
        frames = out.frames[0] if hasattr(out, "frames") else out[0][0]
        audio = getattr(out, "audio", None)
        payload: dict = {"video": frames}
        if audio is not None:
            a0 = audio[0]
            payload["audio"] = a0.float().cpu() if torch.is_tensor(a0) else a0
            payload["audio_sample_rate"] = getattr(
                self.vocoder.config, "output_sampling_rate", 48000
            )
        return DiffusionOutput(output=payload)
