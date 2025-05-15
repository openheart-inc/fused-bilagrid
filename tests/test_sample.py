import sys
import os
sys.path += [os.path.dirname(os.path.abspath(__file__))]
from util import *

import torch

from fused_bilagrid_cuda import bilagrid_sample_forward, bilagrid_sample_backward

def bilagrid_sample_torch(input, grid, rgb):
    grid = (grid - 0.5) * 2
    affine_mats = torch.nn.functional.grid_sample(
        input, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )  # (N, 12, m, h, w)
    affine_mats = affine_mats.permute(0, 2, 3, 4, 1)  # (N, m, h, w, 12)
    affine_mats = affine_mats.reshape(*affine_mats.shape[:-1], 3, 4)  # (N, m, h, w, 3, 4)
    return torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1) + affine_mats[..., 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid_sample():

    print("# Test grid sample")

    N, m = 3, 2
    L, H, W = 5, 7, 15
    h, w = 234, 567

    torch.random.manual_seed(42)

    input = torch.randn((N, 12, L, H, W)).cuda()
    grid = 0.5+0.5*torch.randn((N, m, h, w, 3)).cuda()
    rgb = 0.5+0.5*torch.randn((N, m, h, w, 3)).cuda()

    input = torch.nn.Parameter(input)
    grid = torch.nn.Parameter(grid)
    rgb = torch.nn.Parameter(rgb)

    output = bilagrid_sample_torch(input, grid, rgb)
    output.retain_grad()
    output.requires_grad_(True)

    output1 = bilagrid_sample_forward(input, grid, rgb)
    # print(output.shape, output1.shape)
    assert output.shape == output1.shape

    assert_close(output1, output, 1e-5, "output")

    weights = torch.randn_like(output)
    loss = (weights*output).mean()
    loss.backward()

    v_input, v_grid, v_rgb = bilagrid_sample_backward(input, grid, rgb, output.grad)

    assert_close(v_input, input.grad, 1e-8, "input.grad")
    assert_close(v_grid, grid.grad, 1e-8, "grid.grad")
    assert_close(v_rgb, rgb.grad, 1e-8, "rgb.grad")

    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def profile_bilagrid_sample():

    N, m = 1, 1
    L, H, W = 8, 16, 16
    h, w = 1080, 1440

    torch.random.manual_seed(42)

    input = torch.randn((N, 12, L, H, W)).cuda()
    grid = torch.randn((N, m, h, w, 3)).cuda()
    rgb = torch.randn((N, m, h, w, 3)).cuda()

    print("# Profile forward")
    timeit(lambda: bilagrid_sample_torch(input, grid, rgb), "forward torch")
    timeit(lambda: bilagrid_sample_forward(input, grid, rgb), "forward fused")
    print()

    input = torch.nn.Parameter(input)
    grid = torch.nn.Parameter(grid)
    rgb = torch.nn.Parameter(rgb)

    output = bilagrid_sample_torch(input, grid, rgb)
    output.retain_grad()
    output.requires_grad_(True)

    weight = torch.randn_like(output)
    loss = (weight*output).mean()

    print("# Profile backward")
    timeit(lambda: loss.backward(retain_graph=True), "backward torch")
    timeit(lambda: bilagrid_sample_backward(input, grid, rgb, output.grad), "backward fused")
    print()


if __name__ == "__main__":

    test_bilagrid_sample()

    profile_bilagrid_sample()
