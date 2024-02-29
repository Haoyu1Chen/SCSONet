import os
import sys
import inspect

from torch import nn
import torch


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # # only for inference
        # # x = x.clone()  # !!! Keep the original input intact for the residual connection later
        # x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        x1 = x[:, :self.dim_conv3, :, :]
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x[:, self.dim_conv3:, :, :]), dim=1)
        return x

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


if __name__ == '__main__':
    block = Partial_conv3(3, 2, 'slicing').cuda()
    input = torch.rand(1, 3, 64, 64).cuda()
    output = block(input)
    print(input.size(), output.size())

# import torch
# import torch.nn as nn
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from functools import partial
# from typing import List
# from torch import Tensor
# import copy
# import os
#
# try:
#     from mmdet.models.builder import BACKBONES as det_BACKBONES
#     from mmdet.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmdet = True
# except ImportError:
#     print("If for detection, please install mmdetection first")
#     has_mmdet = False
#

# class Partial_conv3(nn.Module):
#
#     def __init__(self, dim, n_div, forward):
#         super().__init__()
#         self.dim_conv3 = dim // n_div
#         self.dim_untouched = dim - self.dim_conv3
#         self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
#
#         if forward == 'slicing':
#             self.forward = self.forward_slicing
#         elif forward == 'split_cat':
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError
#
#     def forward_slicing(self, x: Tensor) -> Tensor:
#         # only for inference
#         x = x.clone()   # !!! Keep the original input intact for the residual connection later
#         x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
#
#         return x
#
#     def forward_split_cat(self, x: Tensor) -> Tensor:
#         # for training/inference
#         x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
#         x1 = self.partial_conv3(x1)
#         x = torch.cat((x1, x2), 1)
#
#         return x

#
# class MLPBlock(nn.Module):
#
#     def __init__(self,
#                  dim,
#                  n_div,
#                  mlp_ratio,
#                  drop_path,
#                  layer_scale_init_value,
#                  act_layer,
#                  norm_layer,
#                  pconv_fw_type
#                  ):
#
#         super().__init__()
#         self.dim = dim
#         self.mlp_ratio = mlp_ratio
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.n_div = n_div
#
#         mlp_hidden_dim = int(dim * mlp_ratio)
#
#         mlp_layer: List[nn.Module] = [
#             nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
#             norm_layer(mlp_hidden_dim),
#             act_layer(),
#             nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
#         ]
#
#         self.mlp = nn.Sequential(*mlp_layer)
#
#         self.spatial_mixing = Partial_conv3(
#             dim,
#             n_div,
#             pconv_fw_type
#         )
#
#         if layer_scale_init_value > 0:
#             self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             self.forward = self.forward_layer_scale
#         else:
#             self.forward = self.forward
#
#     def forward(self, x: Tensor) -> Tensor:
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(self.mlp(x))
#         return x
#
#     def forward_layer_scale(self, x: Tensor) -> Tensor:
#         shortcut = x
#         x = self.spatial_mixing(x)
#         x = shortcut + self.drop_path(
#             self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
#         return x
#
