import torch
from fused_bilagrid_cuda import bilagrid_sample_forward, bilagrid_sample_backward


class _FusedGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, grid, rgb):
        output = bilagrid_sample_forward(input, grid, rgb)
        ctx.save_for_backward(input, grid, rgb)
        return output

    @staticmethod
    def backward(ctx, v_output):
        input, grid, rgb = ctx.saved_tensors
        return bilagrid_sample_backward(input, grid, rgb, v_output)


def fused_bilagrid_sample(input, grid, rgb):
    return _FusedGridSample.apply(
        input.contiguous(),
        grid.contiguous(),
        rgb.contiguous()
    )
