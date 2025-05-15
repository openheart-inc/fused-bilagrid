import pytest
from time import perf_counter

import torch
import torch.nn.functional as F

from fused_bilagrid_cuda import grid_sample_forward, grid_sample_backward

def grid_sample_torch(input, grid, rgb):
    affine_mats = torch.nn.functional.grid_sample(
        input, grid, mode="bilinear", align_corners=True, padding_mode="border"
    )  # (N, 12, m, h, w)
    affine_mats = affine_mats.permute(0, 2, 3, 4, 1)  # (N, m, h, w, 12)
    affine_mats = affine_mats.reshape(*affine_mats.shape[:-1], 3, 4)  # (N, m, h, w, 3, 4)
    return torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1) + affine_mats[..., 3]


def assert_close(x, y, tol, name: str):
    err = torch.amax(torch.abs(y - x)).item()
    print(f"{name}: maxerr = {err:.2g}")
    assert err < tol


def timeit(fun, name: str):

    for i in range(2):
        fun()
    torch.cuda.synchronize()

    repeat = 20

    time0 = perf_counter()
    for i in range(repeat):
        fun()
        torch.cuda.synchronize()
    time1 = perf_counter()

    dt = 1e3 * (time1-time0) / repeat
    print(f"{name}: {dt:.2f} ms")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_grid_sample():

    print("# Test grid sample")

    N, m = 3, 2
    L, H, W = 5, 7, 15
    h, w = 234, 567

    torch.random.manual_seed(42)

    input = torch.randn((N, 12, L, H, W)).cuda()
    grid = torch.randn((N, m, h, w, 3)).cuda()
    rgb = torch.randn((N, m, h, w, 3)).cuda()

    input = torch.nn.Parameter(input)
    grid = torch.nn.Parameter(grid)
    rgb = torch.nn.Parameter(rgb)

    output = grid_sample_torch(input, grid, rgb)
    output.retain_grad()
    output.requires_grad_(True)

    output1 = grid_sample_forward(input, grid, rgb)
    # print(output.shape, output1.shape)
    assert output.shape == output1.shape

    assert_close(output1, output, 1e-5, "output")

    weights = torch.randn_like(output)
    loss = (weights*output).mean()
    loss.backward()

    v_input, v_grid, v_rgb = grid_sample_backward(input, grid, rgb, output.grad)

    assert_close(v_input, input.grad, 1e-5, "input.grad")
    assert_close(v_grid, grid.grad, 1e-5, "grid.grad")
    assert_close(v_rgb, rgb.grad, 1e-5, "rgb.grad")

    print()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def profile_grid_sample():

    N, m = 1, 1
    L, H, W = 8, 16, 16
    h, w = 1080, 1440

    torch.random.manual_seed(42)

    input = torch.randn((N, 12, L, H, W)).cuda()
    grid = torch.randn((N, m, h, w, 3)).cuda()
    rgb = torch.randn((N, m, h, w, 3)).cuda()

    print("# Profile forward")
    timeit(lambda: grid_sample_torch(input, grid, rgb), "forward torch")
    timeit(lambda: grid_sample_forward(input, grid, rgb), "forward fused")
    print()

    input = torch.nn.Parameter(input)
    grid = torch.nn.Parameter(grid)
    rgb = torch.nn.Parameter(rgb)

    output = grid_sample_torch(input, grid, rgb)
    output.retain_grad()
    output.requires_grad_(True)

    weight = torch.randn_like(output)
    loss = (weight*output).mean()

    print("# Profile backward")
    timeit(lambda: loss.backward(retain_graph=True), "backward torch")
    timeit(lambda: grid_sample_backward(input, grid, rgb, output.grad), "backward fused")
    print()


if __name__ == "__main__":

    test_grid_sample()

    profile_grid_sample()

"""
May 14 initial commit:

# Test grid sample
output: maxerr = 7.6e-06
input.grad: maxerr = 2.9e-10
grid.grad: maxerr = 1.2e-10
rgb.grad: maxerr = 5.4e-06

# Profile forward
forward torch: 4.40 ms
forward fused: 0.52 ms

# Profile backward
backward torch: 18.59 ms
backward fused: 12.05 ms

"""
