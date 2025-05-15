import sys
import os
sys.path += [os.path.dirname(os.path.abspath(__file__))]
from util import *

import torch

from nerfstudio.model_components import lib_bilagrid
import fused_bilagrid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
def test_bilagrid():

    print("# Test bilagrid")

    N = 3
    grid_shape = (15, 16, 7)
    h, w = 345, 678
    h, w = 5, 8
    idx = torch.tensor([0]).cuda()

    bilagrid0 = lib_bilagrid.BilateralGrid(N, *grid_shape).cuda()
    bilagrid1 = fused_bilagrid.BilateralGrid(N, *grid_shape).cuda()

    torch.random.manual_seed(42)
    grid_data = torch.randn_like(bilagrid0.grids.data)
    bilagrid0.grids.data = grid_data
    bilagrid1.grids.data = grid_data
    assert_close(bilagrid1.grids, bilagrid0.grids, 1e-5, "bilagrid")

    ni = len(idx) if idx is not None else N
    uv = 0.5 + 0.5 * torch.randn((ni, h, w, 2)).cuda()
    rgb = 0.5 + 0.5 * torch.randn((ni, h, w, 3)).cuda()
    uv0 = torch.nn.Parameter(uv.clone())
    rgb0 = torch.nn.Parameter(rgb.clone())
    uv1 = torch.nn.Parameter(uv.clone())
    rgb1 = torch.nn.Parameter(rgb.clone())

    output0 = lib_bilagrid.slice(bilagrid0, uv0, rgb0, idx)['rgb']
    output1 = fused_bilagrid.slice(bilagrid1, uv1, rgb1, idx)['rgb']

    assert_close(output1, output0, 1e-5, "output")

    weights = torch.randn_like(output0)
    (weights*output0).mean().backward()
    (weights*output1).mean().backward()

    assert_close(bilagrid1.grids.grad, bilagrid0.grids.grad, 1e-6, "bilagrid.grad")
    assert_close(uv1.grad, uv0.grad, 1e-6, "uv.grad")
    assert_close(rgb1.grad, rgb0.grad, 1e-6, "rgb.grad")
    print()



if __name__ == "__main__":

    test_bilagrid()

