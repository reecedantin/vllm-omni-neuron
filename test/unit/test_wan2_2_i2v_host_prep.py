"""TI2V image-to-video host-side condition prep (CPU, no device).

The Trn2 TI2V I2V path keeps every condition tensor on the host until it is finished: a host tensor
must be cast on the host and moved as one contiguous base tensor, never routed through the compiled
device cast, and the first-frame blend must equal upstream's arithmetic.
"""

import torch

from vllm_omni_neuron.diffusion.models.wan2_2.pipeline_wan2_2_i2v import (
    NeuronWanI2VPipeline,
    _ti2v_blend,
)


class _NoDeviceCast(NeuronWanI2VPipeline):
    def __init__(self):  # bypass the model setup; only the helpers are exercised
        torch.nn.Module.__init__(self)

    def _cast_device_dtype(self, tensor, dtype):
        raise AssertionError("a host tensor must not reach the compiled device cast")


def test_host_tensor_is_cast_on_host_and_made_contiguous():
    pipe = _NoDeviceCast()
    x = torch.randn(1, 3, 4, 6, dtype=torch.float32).transpose(2, 3)  # strided view
    assert not x.is_contiguous()
    y = pipe._to_device_dtype(x, "cpu", torch.bfloat16)
    assert y.dtype == torch.bfloat16 and y.is_contiguous() and y.storage_offset() == 0
    assert torch.equal(y, x.to(torch.bfloat16))


def test_ti2v_blend_matches_upstream_arithmetic():
    mask = torch.ones(1, 1, 3, 4, 4, dtype=torch.bfloat16)
    mask[:, :, 0] = 0
    cond = torch.randn(1, 48, 3, 4, 4).to(torch.bfloat16)
    lat = torch.randn(1, 48, 3, 4, 4).to(torch.bfloat16)
    out = _ti2v_blend(mask, cond, lat)
    assert out.dtype == cond.dtype
    assert torch.equal(out, (1 - mask) * cond + mask * lat)
    assert torch.equal(out[:, :, 0], cond[:, :, 0]) and torch.equal(out[:, :, 1:], lat[:, :, 1:])
    # Re-blending the blended latent (what upstream forward does after the loop) is a no-op.
    assert torch.equal(_ti2v_blend(mask, cond, out), out)


def _bcast_worker(rank, port, out):
    import os

    import torch.distributed as dist

    from vllm_omni_neuron.diffusion.models.wan2_2.pipeline_wan2_2_i2v import broadcast_from_rank0

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=2)
    if rank == 0:
        t = torch.arange(2 * 48 * 1 * 3 * 4, dtype=torch.bfloat16).view(2, 48, 1, 3, 4)
    else:  # placeholder with a different (full-length) shape, as a non-encoding rank builds it
        t = torch.zeros(2, 48, 5, 3, 4)
    torch.save(broadcast_from_rank0(t, dist.group.WORLD), f"{out}/r{rank}.pt")
    dist.destroy_process_group()


def test_condition_broadcast_carries_rank0_shape(tmp_path):
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_bcast_worker, args=(port, str(tmp_path)), nprocs=2)
    r0, r1 = torch.load(tmp_path / "r0.pt"), torch.load(tmp_path / "r1.pt")
    assert r1.shape == (2, 48, 1, 3, 4)
    assert torch.equal(r0, r1)
