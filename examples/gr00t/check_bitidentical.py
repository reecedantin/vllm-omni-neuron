# SPDX-License-Identifier: Apache-2.0
"""Check that the host-baked AdaLN modulation tables reproduce the DEVICE modulation math.

The device matmul accumulates in fp32 and rounds once to bf16; plain CPU bf16 ``F.linear`` does
NOT (it rounds throughout), so the baked tables deliberately differ from a plain CPU bf16 forward.
``bake_linear_modulation`` models the device path (silu fp32 -> cast -> fp32-accumulated matmul ->
one rounding -> bias add), so the test is: baked table == an independent fp32-accumulate reference,
and baked table != plain-bf16, confirming the helper is doing the device-matching rounding A2 found.
Runs on CPU (``cpumode.sh``); no NeuronCore needed.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
args = ap.parse_args()

from vllm_omni_neuron.diffusion.models.gr00t.model import NeuronGr00tModel  # noqa: E402


def device_matmul_ref(temb, lin):
    """fp32-accumulated matmul + single rounding + bf16 bias add = the device's rounding."""
    x = F.silu(temb.float()).to(torch.bfloat16).float()
    y = (x @ lin.weight.to(torch.bfloat16).float().t()).to(torch.bfloat16)
    return (y + lin.bias.to(torch.bfloat16))[0]


m = NeuronGr00tModel.from_pretrained(args.model, dtype=torch.bfloat16)  # CPU
dit = m.head.model
bad_ref, bad_plain = [], 0
with torch.no_grad():
    for step, t in enumerate(m.head.timesteps()):
        temb = dit.temb(torch.full((1,), t, dtype=torch.long))
        for i, blk in enumerate(dit.transformer_blocks):
            baked = m.head.adaln_table[step, i]
            if not torch.equal(baked, device_matmul_ref(temb, blk.norm1.linear)):
                bad_ref.append((step, i))
            if not torch.equal(baked, blk.norm1.linear(F.silu(temb))[0]):
                bad_plain += 1
        if not torch.equal(m.head.adaln_out_table[step], device_matmul_ref(temb, dit.proj_out_1)):
            bad_ref.append((step, "out"))
n = len(m.head.timesteps()) * (len(dit.transformer_blocks) + 1)
print(
    f"matches_device_matmul_ref={not bad_ref} (checked {n}, mismatches {len(bad_ref)}); "
    f"differs_from_plain_bf16={bad_plain}/{n} (expected > 0: that is the fp32-accumulate fix)"
)
