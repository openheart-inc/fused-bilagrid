# Copyright 2024 Yuehao Wang (https://github.com/yuehaowang).
# This part of code is borrowed form ["Bilateral Guided Radiance Field Processing"](https://bilarfpro.github.io/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from torch import nn

from fused_bilagrid_cuda import (
    bilagrid_sample_backward,
    bilagrid_sample_forward,
    bilagrid_uniform_sample_forward,
    bilagrid_uniform_sample_backward,
    tv_loss_forward,
    tv_loss_backward,
)


class _FusedGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bilagrid, coords, rgb, compute_coords_grad=False):
        output = bilagrid_sample_forward(bilagrid, coords, rgb)
        ctx.save_for_backward(bilagrid, coords, rgb)
        ctx.compute_coords_grad = compute_coords_grad
        return output

    @staticmethod
    def backward(ctx, v_output):
        bilagrid, coords, rgb = ctx.saved_tensors
        return *bilagrid_sample_backward(
            bilagrid, coords, rgb, v_output,
            ctx.compute_coords_grad
        ), None


class _FusedUniformGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bilagrid, rgb):
        output = bilagrid_uniform_sample_forward(bilagrid, rgb)
        ctx.save_for_backward(bilagrid, rgb)
        return output

    @staticmethod
    def backward(ctx, v_output):
        bilagrid, rgb = ctx.saved_tensors
        return bilagrid_uniform_sample_backward(
            bilagrid, rgb, v_output,
        )


class _FusedTotalVariationLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bilagrid):
        assert bilagrid.ndim == 5 and bilagrid.shape[1] == 12, bilagrid.shape
        assert bilagrid.shape[-3] >= 2 and bilagrid.shape[-2] >= 2 and bilagrid.shape[-1] >= 2, bilagrid.shape

        ctx.save_for_backward(bilagrid)
        return tv_loss_forward(bilagrid)

    @staticmethod
    def backward(ctx, v_output):
        (bilagrid,) = ctx.saved_tensors
        return tv_loss_backward(bilagrid, v_output)


def fused_bilagrid_sample(bilagrid, coords, rgb, compute_coords_grad=False):
    if coords is not None:
        return _FusedGridSample.apply(
            bilagrid.contiguous(),
            coords.contiguous(),
            rgb.contiguous(),
            compute_coords_grad
        )
    else:
        return _FusedUniformGridSample.apply(
            bilagrid.contiguous(),
            rgb.contiguous(),
        )


def total_variation_loss(x: torch.Tensor):
    """Returns total variation on multi-dimensional tensors.

    Args:
        x (torch.Tensor): The input tensor with shape $(B, 12, L, H, W)$, where $B$ is the batch size.
    """
    return _FusedTotalVariationLoss.apply(x.contiguous())


def slice(bil_grids, xy, rgb, grid_idx, compute_coords_grad=False):
    """Slices a batch of 3D bilateral grids by pixel coordinates `xy` and gray-scale guidances of pixel colors `rgb`.

    Supports 2-D, 3-D, and 4-D input shapes. The first dimension of the input is the batch size
    and the last dimension is 2 for `xy`, 3 for `rgb`, and 1 for `grid_idx`.

    The return value is a dictionary containing the affine transformations `affine_mats` sliced from bilateral grids and
    the output color `rgb_out` after applying the afffine transformations.

    In the 2-D input case, `xy` is a $(N, 2)$ tensor, `rgb` is  a $(N, 3)$ tensor, and `grid_idx` is a $(N, 1)$ tensor.
    Then `affine_mats[i]` can be obtained via slicing the bilateral grid indexed at `grid_idx[i]` by `xy[i, :]` and `rgb2gray(rgb[i, :])`.
    For 3-D and 4-D input cases, the behavior of indexing bilateral grids and coordinates is the same with the 2-D case.

    .. note::
        This function can be regarded as a wrapper of `color_affine_transform` and `BilateralGrid` with a slight performance improvement.
        When `grid_idx` contains a unique index, only a single bilateral grid will used during the slicing. In this case, this function will not
        perform tensor indexing to avoid data copy and extra memory
        (see [this](https://discuss.pytorch.org/t/does-indexing-a-tensor-return-a-copy-of-it/164905)).

    Args:
        bil_grids (`BilateralGrid`): An instance of $N$ bilateral grids.
        xy (Optional[torch.Tensor]): The x-y coordinates of shape $(..., 2)$ in the range of $[0,1]$.
        rgb (torch.Tensor): The RGB values of shape $(..., 3)$ for computing the guidance coordinates, ranging in $[0,1]$.
        grid_idx (torch.Tensor): The indices of bilateral grids for each slicing. Shape: $(..., 1)$.

    Returns:
        {
            "rgb": Transformed RGB colors. Shape: (..., 3),
        }
    """

    sh_ = rgb.shape

    grid_idx_unique = torch.unique(grid_idx)
    if len(grid_idx_unique) == 1:
        # All pixels are from a single view.
        grid_idx = grid_idx_unique  # (1,)
        if xy is not None:
            xy = xy.unsqueeze(0)  # (1, ..., 2)
        rgb = rgb.unsqueeze(0)  # (1, ..., 3)
    else:
        # Pixels are randomly sampled from different views.
        if len(grid_idx.shape) == 4:
            grid_idx = grid_idx[:, 0, 0, 0]  # (chunk_size,)
        elif len(grid_idx.shape) == 3:
            grid_idx = grid_idx[:, 0, 0]  # (chunk_size,)
        elif len(grid_idx.shape) == 2:
            grid_idx = grid_idx[:, 0]  # (chunk_size,)
        else:
            raise ValueError("The input to bilateral grid slicing is not supported yet.")

    rgb = bil_grids(xy, rgb, grid_idx, compute_coords_grad)

    return {
        "rgb": rgb.reshape(*sh_),
    }


class BilateralGrid(nn.Module):
    """Class for 3D bilateral grids.

    Holds one or more than one bilateral grids.
    """

    def __init__(self, num, grid_X=16, grid_Y=16, grid_W=8):
        """
        Args:
            num (int): The number of bilateral grids (i.e., the number of views).
            grid_X (int): Defines grid width $W$.
            grid_Y (int): Defines grid height $H$.
            grid_W (int): Defines grid guidance dimension $L$.
        """
        super(BilateralGrid, self).__init__()

        self.grid_width = grid_X
        """Grid width. Type: int."""
        self.grid_height = grid_Y
        """Grid height. Type: int."""
        self.grid_guidance = grid_W
        """Grid guidance dimension. Type: int."""

        # Initialize grids.
        grid = self._init_identity_grid()
        self.grids = nn.Parameter(grid.tile(num, 1, 1, 1, 1))  # (N, 12, L, H, W)
        """ A 5-D tensor of shape $(N, 12, L, H, W)$."""

        # Weights of BT601 RGB-to-gray.
        self.register_buffer("rgb2gray_weight", torch.Tensor([[0.299, 0.587, 0.114]]))
        """ A function that converts RGB to gray-scale guidance in $[-1, 1]$."""

    def _init_identity_grid(self):
        grid = torch.eye(4)[:3].float()
        grid = grid.repeat([self.grid_guidance * self.grid_height * self.grid_width, 1])  # (L * H * W, 12)
        grid = grid.reshape(1, self.grid_guidance, self.grid_height, self.grid_width, -1)  # (1, L, H, W, 12)
        grid = grid.permute(0, 4, 1, 2, 3)  # (1, 12, L, H, W)
        return grid

    def tv_loss(self):
        """Computes and returns total variation loss on the bilateral grids."""
        return total_variation_loss(self.grids)

    def forward(self, grid_xy, rgb, idx=None, compute_coords_grad=False):
        """Bilateral grid slicing. Supports 2-D, 3-D, 4-D, and 5-D input.
        For the 2-D, 3-D, and 4-D cases, please refer to `slice`.
        For the 5-D cases, `idx` will be unused and the first dimension of `xy` should be
        equal to the number of bilateral grids. Then this function becomes PyTorch's
        [`F.bilagrid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.bilagrid_sample.html).

        Args:
            grid_xy (Optional[torch.Tensor]): The x-y coordinates in the range of $[0,1]$.
            rgb (torch.Tensor): The RGB values in the range of $[0,1]$.
            idx (torch.Tensor): The bilateral grid indices.

        Returns:
            Affine transformed RGB values with same shape as input `rgb`$.
        """

        grids = self.grids
        input_ndims = len(rgb.shape)
        assert len(rgb.shape) == input_ndims

        if input_ndims > 1 and input_ndims < 5:
            # Convert input into 5D
            for i in range(5 - input_ndims):
                if grid_xy is not None:
                    grid_xy = grid_xy.unsqueeze(1)
                rgb = rgb.unsqueeze(1)
            assert idx is not None
        elif input_ndims != 5:
            raise ValueError("Bilateral grid slicing only takes either 2D, 3D, 4D and 5D inputs")

        grids = self.grids
        if idx is not None:
            grids = grids[idx]

        rgb = fused_bilagrid_sample(grids, grid_xy, rgb, compute_coords_grad)

        for _ in range(5 - input_ndims):
            rgb = rgb.squeeze(1)

        return rgb
