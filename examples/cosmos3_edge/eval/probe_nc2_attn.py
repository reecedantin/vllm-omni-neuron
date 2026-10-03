"""Device check of the NC-v2 Edge attention kernel vs CPU fp32.  NEURON_RT_VISIBLE_CORES=0 python probe_nc2_attn.py SQ SK [H HK]"""
import sys
import time

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.envs import get_compile_backend_name

import vllm_omni_neuron.bootstrap  # noqa: F401
from vllm_omni_neuron.diffusion.models.cosmos3_edge import attention as A
from vllm_omni_neuron.diffusion.models.cosmos3_edge.nki_attention_nc2 import (
    edge_attn_fwd_nc2,
    prepare_nc2_inputs,
)
from vllm_omni_neuron.lite_compat import nki_op


@nki_op("edge_probe::nc2_attn")
def _k(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return wrap_nki(edge_attn_fwd_nc2)[1](q, k, v, b, t)

sq, sk = int(sys.argv[1]), int(sys.argv[2])
h, hk = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (16, 8)
torch.manual_seed(0)
q = torch.randn(1, h, sq, 128)
k = torch.randn(1, hk, sk, 128)
v = torch.randn(1, hk, sk, 128)
NT = int(sys.argv[5]) if len(sys.argv) > 5 else 512  # text-prefix length (bias only on the prefix, as in GEN)
valid = torch.ones(1, NT, dtype=torch.bool)
valid[0, NT - 60 :] = False
kb = A.key_padding_bias(valid)
kb_full = torch.cat([kb.reshape(1, -1), torch.zeros(1, sk - NT, dtype=kb.dtype)], -1)
ref = A.torch_attention(q, k, v, 128 ** -0.5, key_bias=kb_full, qblock=0)
dev = torch.device("neuron", 0)
def f(q_, k_, v_, b_):
    args, n = prepare_nc2_inputs(q_, k_, v_, 128 ** -0.5, b_)
    return _k(*args)[:, :n][None]
fn = torch.compile(f, backend=get_compile_backend_name(), fullgraph=True, dynamic=False,
                   options={"model_name": f"probe_nc2_{sq}_{sk}_{h}", "compiler_args": ["--model-type=transformer", "--auto-cast=none", "-O1"]})
args = [x.to(dev) for x in (q.bfloat16(), k.bfloat16(), v.bfloat16(), kb)]
t0 = time.time()
out = fn(*args).cpu().float()
first = time.time() - t0
ws = []
for _ in range(15):
    t0 = time.time()
    fn(*args).sum().item()
    ws.append(time.time() - t0)
pl = A.torch_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), 128 ** -0.5, key_bias=kb_full).float()
print(f"PROBE sq={sq} sk={sk} h={h} rel={((out-ref).norm()/ref.norm()).item():.4f} (cpu bf16 plain {((pl-ref).norm()/ref.norm()).item():.4f}) first={first:.1f}s warm_med={sorted(ws)[7]:.4f}s min={min(ws):.4f}s", flush=True)
