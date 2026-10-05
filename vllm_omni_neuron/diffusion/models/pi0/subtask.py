# SPDX-License-Identifier: Apache-2.0
"""pi0.52 subtask generation: the greedy "User: {task}\\n" -> low-level subtask decode that
``_prepare_action_batch`` runs before each action chunk. Mirrors LeRobot's
``PI052Policy.select_message`` / ``_generate_low_level_subtask`` (non-joint path: no state in the
prompt, no FAST/loc suppression needed because the subtask vocabulary doesn't drift there in
practice for this checkpoint — ``suppress_loc_tokens=True`` is applied regardless, matching
upstream, since it is cheap and exactly reproduces the training-time behavior).

Two device paths. Default (:meth:`Pi052SubtaskGenerator.generate_kv`): one prefill over the
image + prompt prefix fills a shared ``StaticKVCache`` per LM layer, then each generated token is
one position-static ``decode_step`` graph. Fallback (:meth:`Pi052SubtaskGenerator.generate`):
the text-prefix graph recomputes the WHOLE sequence on every new token, bucketed by total length
so the compiled graph is reused; ``lang_tokens`` grows by one real token per step. Both apply
upstream's prefix-LM mask (generated tokens attend causally, the prompt never sees them).
"""

from __future__ import annotations

import bisect

import torch

# PaliGemma's reserved <locDDDD> ids (text_processor_pi052.register_paligemma_loc_tokens).
_LOC_ID_LO, _LOC_ID_HI = 256000, 257024
_FAST_ACTION_VOCAB_SIZE = 1024


def format_subtask_prompt(task: str) -> str:
    """LeRobot's deployed subtask-generation prefill (``_generate_low_level_subtask`` ->
    ``_build_text_batch`` with ``add_generation_prompt=True``): the user turn plus the generation
    prompt the decoder continues from — ``"User: {task}\\nAssistant:"``.

    No trailing space after ``Assistant:`` — SentencePiece folds a trailing space into the first
    generated token, so a space here is a token the model never saw at this position in training."""
    return f"User: {(task or '').strip()}\nAssistant:"


def _token_bucket(n: int, buckets: tuple[int, ...]) -> int:
    i = bisect.bisect_left(buckets, n)
    if i == len(buckets):
        raise ValueError(f"sequence length {n} exceeds the largest subtask bucket {buckets[-1]}")
    return buckets[i]


class Pi052SubtaskGenerator:
    """Greedy subtask decode, bucketed by length. Uses a cached image embedding when
    ``embed_images_graph`` is set (SigLIP + projector run once per request; each decode re-prefill
    then runs ``text_graph_cached`` over the precomputed image tokens), else the full
    :class:`.graphs.Pi05TextPrefixGraph` that recomputes the vision tower every token."""

    def __init__(
        self,
        text_graph,
        tokenizer,
        num_cameras: int,
        image_resolution: int,
        fast_skip_tokens: int = 1152,
        buckets: tuple[int, ...] = (64, 96, 128, 192, 256),
        embed_images_graph=None,
        text_graph_cached=None,
    ):
        self.text_graph = text_graph
        self.embed_images_graph = embed_images_graph
        self.text_graph_cached = text_graph_cached
        self.tokenizer = tokenizer
        self.num_cameras = num_cameras
        self.image_resolution = image_resolution
        self.fast_skip_tokens = fast_skip_tokens
        self.buckets = tuple(sorted(buckets))
        self.eos_id = tokenizer.eos_token_id
        self.special_ids = {int(i) for i in (tokenizer.all_special_ids or []) if i is not None}
        if self.eos_id is not None:
            self.special_ids.add(int(self.eos_id))
        self._ids_cache: dict[str, list[int]] = {}
        self._bias_cache: dict[tuple, torch.Tensor] = {}
        self._prefill_cache: dict[tuple, tuple] = {}

    def _lm_head(self):
        g = self.text_graph_cached if self.text_graph_cached is not None else self.text_graph
        return g.lm_head

    def _vocab_size(self) -> int:
        """Full vocabulary size (``lm_head`` holds a 1/TP slice of it under tensor parallelism)."""
        g = self.text_graph_cached if self.text_graph_cached is not None else self.text_graph
        return self._lm_head().weight.shape[0] * int(getattr(g.prefix, "tp_size", 1))

    @torch.no_grad()
    def generate(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        task: str,
        device: torch.device,
        vision_dtype: torch.dtype,
        max_new_tokens: int = 128,
        suppress_loc_tokens: bool = True,
        min_new_tokens: int = 0,
        return_ids: bool = False,
        step_hook=None,
    ):
        """Re-prefill decode: the whole [image, prompt + generated] sequence runs through the LM
        again for every token (bidirectional prefix, causal generated tokens: upstream's
        ``use_kv_cache=False`` path). :meth:`generate_kv` is the KV-cached path."""
        prompt = format_subtask_prompt(task)
        ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors=None)["input_ids"]
        n_prompt = len(ids)
        if n_prompt + max_new_tokens > self.buckets[-1]:
            raise ValueError(
                f"prompt ({n_prompt} tok) + max_new_tokens ({max_new_tokens}) "
                f"exceeds the largest subtask bucket {self.buckets[-1]}"
            )

        pix = torch.stack([im.detach().cpu() for im in images], dim=1)
        pix = (
            pix.reshape(len(images) * pix.shape[0], *pix.shape[2:])
            .to(vision_dtype)
            .contiguous()
            .to(device)
        )
        img_valid = torch.stack([m.detach().cpu() for m in image_masks], dim=1).float().to(device)

        # Image-prefix cache: SigLIP + projector are identical for every decode step, so run them
        # once up front and feed the cached embedding to the (vision-tower-free) cached text graph.
        cached = self.embed_images_graph is not None and self.text_graph_cached is not None
        img_emb = self.embed_images_graph(pix) if cached else None

        generated: list[int] = []
        cur_bucket = _token_bucket(n_prompt, self.buckets)
        tokens = torch.zeros(1, cur_bucket, dtype=torch.long)
        tokens[0, :n_prompt] = torch.tensor(ids, dtype=torch.long)
        valid = torch.zeros(1, cur_bucket, dtype=torch.float32)
        valid[0, :n_prompt] = 1.0
        causal = torch.zeros(1, cur_bucket, dtype=torch.float32)  # 1.0 on generated tokens
        vocab_size = self._vocab_size()
        fast_lo = vocab_size - 1 - self.fast_skip_tokens - (_FAST_ACTION_VOCAB_SIZE - 1)

        for step in range(max_new_tokens):
            n_live = n_prompt + step
            need_bucket = _token_bucket(n_live, self.buckets)
            if need_bucket != cur_bucket:
                new_tokens = torch.zeros(1, need_bucket, dtype=torch.long)
                new_valid = torch.zeros(1, need_bucket, dtype=torch.float32)
                new_causal = torch.zeros(1, need_bucket, dtype=torch.float32)
                new_tokens[0, :n_live] = tokens[0, :n_live]
                new_valid[0, :n_live] = valid[0, :n_live]
                new_causal[0, :n_live] = causal[0, :n_live]
                tokens, valid, causal = new_tokens, new_valid, new_causal
                cur_bucket = need_bucket
            if cached:
                logits = self.text_graph_cached(
                    img_emb, img_valid, tokens.to(device), valid.to(device), causal.to(device)
                )
            else:
                logits = self.text_graph(
                    pix, img_valid, tokens.to(device), valid.to(device), causal.to(device)
                )
            # The graph lays out [image tokens, text tokens], so the last REAL text token's logits
            # are at absolute position (n_image + n_live - 1), not n_live - 1.
            n_image = logits.shape[1] - cur_bucket
            step_logits = logits[0, n_image + n_live - 1].float().cpu()
            forced_tok = step_hook(step, step_logits.clone()) if step_hook else None
            if 0 < fast_lo < _LOC_ID_LO:
                step_logits[fast_lo:_LOC_ID_LO] = float("-inf")
            if suppress_loc_tokens:
                step_logits[_LOC_ID_LO:_LOC_ID_HI] = float("-inf")
            if step < min_new_tokens:
                for sid in self.special_ids:
                    if sid < step_logits.shape[-1]:
                        step_logits[sid] = float("-inf")
            tok_id = int(step_logits.argmax().item())  # greedy (temperature=0, matches the caller)
            if forced_tok is not None:
                tok_id = int(forced_tok)
            generated.append(tok_id)
            if self.eos_id is not None and tok_id == self.eos_id:
                break
            tokens[0, n_live] = tok_id
            valid[0, n_live] = 1.0
            causal[0, n_live] = 1.0

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return (text, generated) if return_ids else text

    def _suppress(self, step_logits: torch.Tensor, suppress_loc_tokens: bool) -> torch.Tensor:
        vocab_size = step_logits.shape[-1]
        fast_lo = vocab_size - 1 - self.fast_skip_tokens - (_FAST_ACTION_VOCAB_SIZE - 1)
        if 0 < fast_lo < _LOC_ID_LO:
            step_logits[fast_lo:_LOC_ID_LO] = float("-inf")
        if suppress_loc_tokens:
            step_logits[_LOC_ID_LO:_LOC_ID_HI] = float("-inf")
        return step_logits

    @torch.no_grad()
    def generate_kv(
        self,
        images: list[torch.Tensor],
        image_masks: list[torch.Tensor],
        task: str,
        device: torch.device,
        vision_dtype: torch.dtype,
        prefill_graph,
        decode_graph,
        cache_cfg,
        max_new_tokens: int = 128,
        suppress_loc_tokens: bool = True,
        min_new_tokens: int = 0,
        return_ids: bool = False,
        step_hook=None,
        img_emb: torch.Tensor | None = None,
        sync_every: int = 1,
    ):
        """Prefill once + KV-cached decode (LeRobot ``select_message(use_kv_cache=True)``): the
        image + prompt prefix is one bidirectional pass whose per-layer K/V (compacted to the real
        tokens) seed one shared ``StaticKVCache`` per layer; each generated token is then ONE
        position-static decode step through the shared decode-attention layer. Each new token sees
        every real prefix token, the earlier generated tokens and itself (prefix-LM), exactly
        upstream's cached path. ``min_new_tokens`` masks EOS/special ids for that many steps
        (upstream's knob; used to time a fixed-length decode). ``step_hook(step, logits)`` (both
        paths; diagnostics) sees each step's raw fp32 logits and may return a token id to feed
        instead of the greedy pick (teacher forcing).

        Without a ``step_hook`` the greedy pick and the suppression run inside the prefill/decode
        graphs and each step's token output feeds the next step on the device; the host only
        reads the 8-byte token ids, every ``sync_every`` steps, to stop at EOS (steps launched
        past an EOS are discarded). ``img_emb`` (``[ncam, N, W]`` on the device, from
        :class:`.graphs.Pi05EmbedImagesGraph`) skips the vision tower when the caller already
        ran it for this observation."""
        from vllm_omni_neuron.diffusion.attention.decode_attention import StaticKVCache

        ids = self.prompt_ids(task)
        n_prompt = len(ids)
        bucket = _token_bucket(n_prompt, self.buckets)

        img_valid_h = torch.stack([m.detach().cpu() for m in image_masks], dim=1).float()
        if img_emb is None:
            img_emb = self.embed_images_graph(_camera_stack(images, vision_dtype, device))
        n_img = int(img_emb.shape[1])

        n_fill, dev_inputs = self._prefill_inputs(
            ids, bucket, n_img, img_valid_h, cache_cfg.max_len, max_new_tokens, device
        )
        vocab = self._vocab_size()
        bias_free = self.suppress_bias(vocab, suppress_loc_tokens, False, device)
        bias_min = self.suppress_bias(vocab, suppress_loc_tokens, True, device)

        def bias_for(step):  # step = index of the token being produced
            return bias_min if step < min_new_tokens else bias_free

        out = prefill_graph(img_emb, *dev_inputs, bias_for(0))
        n_layers = (len(out) - 2) // 2
        logits, dev_tok = out[0], out[1]
        caches = []
        for i in range(n_layers):
            c = StaticKVCache(cache_cfg, "cpu")  # host zeros: no eager device op
            c.k, c.v = out[2 + i], out[2 + n_layers + i]
            c.pos = torch.full((1,), n_fill, dtype=torch.int32).to(device)
            caches.append(c)

        generated: list[int] = []
        if step_hook is None:
            # Device-resident greedy loop: token ids are read back only to find the EOS.
            pending: list[torch.Tensor] = []
            sync_every = max(1, int(sync_every))
            for step in range(max_new_tokens):
                pending.append(dev_tok)
                last_step = step == max_new_tokens - 1
                if last_step or len(pending) >= sync_every:
                    new = [int(t.cpu().reshape(-1)[0]) for t in pending]
                    pending = []
                    if self.eos_id is not None and self.eos_id in new:
                        generated += new[: new.index(self.eos_id) + 1]
                        break
                    generated += new
                if last_step:
                    break
                _, dev_tok = decode_graph(dev_tok, bias_for(step + 1), caches)
        else:
            for step in range(max_new_tokens):
                raw = logits.cpu().reshape(-1).float()
                forced_tok = step_hook(step, raw.clone())
                step_logits = self._suppress(raw, suppress_loc_tokens)
                if step < min_new_tokens:
                    for sid in self.special_ids:
                        if sid < step_logits.shape[-1]:
                            step_logits[sid] = float("-inf")
                tok_id = int(step_logits.argmax().item())  # greedy (temperature=0)
                if forced_tok is not None:
                    tok_id = int(forced_tok)
                generated.append(tok_id)
                if (
                    self.eos_id is not None and tok_id == self.eos_id
                ) or step == max_new_tokens - 1:
                    break
                tok = torch.tensor([[tok_id]], dtype=torch.long).to(device)
                logits, _ = decode_graph(tok, bias_for(step + 1), caches)

        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return (text, generated) if return_ids else text

    # -- host-side caches (a repeated instruction / camera layout costs no host work) ---------
    def prompt_ids(self, task: str) -> list[int]:
        """Token ids of the subtask prompt for ``task`` (memoized: a robot repeats its task)."""
        ids = self._ids_cache.get(task)
        if ids is None:
            prompt = format_subtask_prompt(task)
            ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors=None)["input_ids"]
            if len(self._ids_cache) >= 64:
                self._ids_cache.clear()
            self._ids_cache[task] = ids
        return ids

    def suppress_bias(self, vocab: int, suppress_loc: bool, mask_special: bool, device):
        """``[1, V]`` fp32 device bias for the in-graph greedy pick: :data:`MASK_VALUE` on the
        FAST action ids (and ``<loc>`` ids when ``suppress_loc``; EOS/special ids when
        ``mask_special``), 0 elsewhere -- the same ids :meth:`_suppress` fills with ``-inf``."""
        from .graphs import MASK_VALUE

        key = (vocab, bool(suppress_loc), bool(mask_special), str(device))
        b = self._bias_cache.get(key)
        if b is None:
            h = torch.zeros(vocab, dtype=torch.float32)
            fast_lo = vocab - 1 - self.fast_skip_tokens - (_FAST_ACTION_VOCAB_SIZE - 1)
            if 0 < fast_lo < _LOC_ID_LO:
                h[fast_lo:_LOC_ID_LO] = MASK_VALUE
            if suppress_loc:
                h[_LOC_ID_LO:_LOC_ID_HI] = MASK_VALUE
            if mask_special:
                for sid in self.special_ids:
                    if sid < vocab:
                        h[sid] = MASK_VALUE
            b = h[None].to(device)
            self._bias_cache[key] = b
        return b

    def _prefill_inputs(self, ids, bucket, n_img, img_valid_h, max_len, max_new_tokens, device):
        """Device inputs of the prefill graph after ``img_emb``: ``(image_valid, tokens, valid,
        sel, last)``, memoized per (prompt ids, camera validity) -- ``sel`` alone is a
        ``[max_len, P]`` fp32 matrix (~4 MB) that would otherwise be rebuilt and uploaded per
        request."""
        cams = tuple(bool(img_valid_h[0, c] > 0.5) for c in range(self.num_cameras))
        n_prompt = len(ids)
        key = (tuple(ids), cams, n_img, max_len, str(device))
        hit = self._prefill_cache.get(key)
        n_fill = sum(cams) * n_img + n_prompt
        if n_fill + max_new_tokens > max_len:
            raise ValueError(
                f"prefix ({n_fill} tok) + max_new_tokens ({max_new_tokens}) exceeds the subtask "
                f"KV cache ({max_len})"
            )
        if hit is not None:
            return n_fill, hit
        # Real prefix tokens in layout order: valid camera slots' image tokens, then the prompt.
        keep = [c * n_img + j for c in range(self.num_cameras) if cams[c] for j in range(n_img)]
        keep += [self.num_cameras * n_img + j for j in range(n_prompt)]
        p_len = self.num_cameras * n_img + bucket
        sel = torch.zeros(max_len, p_len, dtype=torch.float32)
        sel[torch.arange(n_fill), torch.tensor(keep)] = 1.0
        last = torch.zeros(1, p_len, dtype=torch.float32)
        last[0, keep[-1]] = 1.0
        tokens = torch.zeros(1, bucket, dtype=torch.long)
        tokens[0, :n_prompt] = torch.tensor(ids, dtype=torch.long)
        valid = torch.zeros(1, bucket, dtype=torch.float32)
        valid[0, :n_prompt] = 1.0
        hit = tuple(t.to(device) for t in (img_valid_h, tokens, valid, sel, last))
        if len(self._prefill_cache) >= 16:
            self._prefill_cache.clear()
        self._prefill_cache[key] = hit
        return n_fill, hit


def _camera_stack(images, vision_dtype, device) -> torch.Tensor:
    """``[ncam]`` x ``[B, 3, R, R]`` camera tensors -> the ``[B*ncam, 3, R, R]`` device stack."""
    pix = torch.stack([im.detach().cpu() for im in images], dim=1)
    pix = pix.reshape(pix.shape[0] * pix.shape[1], *pix.shape[2:]).to(vision_dtype).contiguous()
    return pix.to(device)
