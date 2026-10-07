# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 distilled text-to-video with synchronized audio on Neuron, offline, via the Omni entrypoint.

The default stage (``ltx2_stage.yaml``) runs the transformer tensor-parallel over four logical NeuronCores.
It is a multi-process stage, so select devices with ``NEURON_VISIBLE_DEVICES`` (comma list), not
``NEURON_RT_VISIBLE_CORES``.

Usage:
    unset NEURON_RT_VISIBLE_CORES; export NEURON_VISIBLE_DEVICES=0,1,2,3
    python examples/ltx2/run.py --height 512 --width 768 --num-frames 121 --steps 8 --output ltx25_t2v.mp4
    python examples/ltx2/run.py --height 64 --width 96 --num-frames 9 --steps 2 --output smoke.mp4   # smoke
"""

from __future__ import annotations

import argparse
import os
import time

import vllm_omni_neuron.bootstrap  # noqa: F401  isort: skip  must precede vllm imports
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

parser = argparse.ArgumentParser(description="LTX-2.5 distilled text-to-video (+ audio) on Neuron")
parser.add_argument(
    "--model-path", default=os.environ.get("LTX25_WEIGHTS", "Lightricks/LTX-2.5-Diffusers")
)
parser.add_argument("--stage-config", default=None)
parser.add_argument(
    "--prompt",
    default="A red fox walking through a snowy forest at dawn, the camera tracking alongside.",
)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--width", type=int, default=768)
parser.add_argument("--num-frames", type=int, default=121)
parser.add_argument("--steps", type=int, default=8)
parser.add_argument("--fps", type=float, default=24.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="ltx25_t2v.mp4")
parser.add_argument("--profile", action="store_true", help="time warm requests after the first")
parser.add_argument(
    "--warm-repeats", type=int, default=1, help="number of timed warm requests with --profile"
)
args = parser.parse_args()


def main() -> None:
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ltx2_stage.yaml"
    )
    # The first request at a new shape compiles the transformer and the VAE tile graph (~20 min cold).
    timeout = int(os.environ.get("LTX25_HANDSHAKE_TIMEOUT_S", "7200"))
    try:
        import vllm_omni.diffusion.stage_diffusion_proc as sdp

        sdp._HANDSHAKE_POLL_TIMEOUT_S = max(getattr(sdp, "_HANDSHAKE_POLL_TIMEOUT_S", 0), timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"[init] handshake timeout not patched: {exc!r}")
    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=timeout,
        init_timeout=timeout,
    )

    params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        frame_rate=args.fps,
        seed=args.seed,
        output_type="np",
    )
    print(f"[run] {args.height}x{args.width}x{args.num_frames}, {args.steps} steps")
    t0 = time.perf_counter()
    result = omni.generate({"prompt": args.prompt}, params)
    print(
        f"[run] first request (includes compile/load on a cold cache): {time.perf_counter() - t0:.2f}s"
    )
    if args.profile:
        for _ in range(max(1, args.warm_repeats)):
            t0 = time.perf_counter()
            result = omni.generate({"prompt": args.prompt}, params)
            print(f"[profile] warm request: {time.perf_counter() - t0:.2f}s")
    save(result)


def save(result) -> None:
    import numpy as np

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    request_output = result[0].request_output
    video = request_output.images[0]
    # The engine moves "audio" / "audio_sample_rate" from the pipeline output into the request's
    # multimodal side channel; images[0] is the video alone.
    mm = request_output.multimodal_output or {}
    audio = mm.get("audio")
    sr = mm.get("audio_sample_rate", 48000)

    video = np.asarray(video.detach().cpu().float() if hasattr(video, "detach") else video)
    # The pipeline returns float frames in [0, 1]. export_to_video scales by 255 itself, the muxer
    # takes uint8: keep one copy of each (scaling an already-uint8 array again wraps mod 256).
    if video.dtype == np.uint8:
        video_u8, video_01 = video, video.astype(np.float32) / 255.0
    else:
        video_01 = np.clip(video, 0.0, 1.0)
        video_u8 = (video_01 * 255).round().astype("uint8")

    if audio is not None:
        audio_arr = np.asarray(audio.detach().cpu().float() if hasattr(audio, "detach") else audio)
        try:
            from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

            data = mux_video_audio_bytes(
                video_u8, audio_arr, fps=float(args.fps), audio_sample_rate=int(sr)
            )
            with open(args.output, "wb") as f:
                f.write(data)
            print(f"[run] saved {args.output} (video + {sr} Hz audio)")
            return
        except Exception as exc:  # noqa: BLE001
            import wave

            wav_path = args.output.rsplit(".", 1)[0] + ".wav"
            print(f"[run] audio mux unavailable ({exc!r}); writing {wav_path} next to the video")
            with wave.open(wav_path, "wb") as w:
                w.setnchannels(audio_arr.shape[0] if audio_arr.ndim == 2 else 1)
                w.setsampwidth(2)
                w.setframerate(int(sr))
                pcm = np.clip(audio_arr.T if audio_arr.ndim == 2 else audio_arr, -1, 1)
                w.writeframes((pcm * 32767).astype("int16").tobytes())

    from diffusers.utils import export_to_video

    export_to_video(list(video_01), args.output, fps=args.fps)
    print(f"[run] saved {args.output}")


if __name__ == "__main__":
    main()
