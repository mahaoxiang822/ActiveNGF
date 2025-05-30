# This file is a part of ESLAM.
#
# ESLAM is a NeRF-based SLAM system. It utilizes Neural Radiance Fields (NeRF)
# to perform Simultaneous Localization and Mapping (SLAM) in real-time.
# This software is the implementation of the paper "ESLAM: Efficient Dense SLAM
# System Based on Hybrid Representation of Signed Distance Fields" by
# Mohammad Mahdi Johari, Camilla Carta, and Francois Fleuret.
#
# Copyright 2023 ams-OSRAM AG
#
# Author: Mohammad Mahdi Johari <mohammad.johari@idiap.ch>
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
#
# This file is a modified version of https://github.com/cvg/nice-slam/blob/master/src/utils/Visualizer.py
# which is covered by the following copyright and permission notice:
    #
    # Copyright 2022 Zihan Zhu, Songyou Peng, Viktor Larsson, Weiwei Xu, Hujun Bao, Zhaopeng Cui, Martin R. Oswald, Marc Pollefeys
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

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.common import cam_pose_to_matrix
import torch.nn.functional as F

class Frame_Visualizer(object):
    """
    Visualizes itermediate results, render out depth and color images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).
    Args:
        freq (int): frequency of visualization.
        inside_freq (int): frequency of visualization inside each iteration.
        vis_dir (str): directory to save the visualization results.
        renderer (Renderer): renderer.
        truncation (float): truncation distance.
        verbose (bool): whether to print out the visualization results.
        device (str): device.
    """

    def __init__(self, freq, inside_freq, vis_dir, renderer, truncation, verbose, device='cuda:0'):
        self.freq = freq
        self.device = device
        self.vis_dir = vis_dir
        self.verbose = verbose
        self.renderer = renderer
        self.inside_freq = inside_freq
        self.truncation = truncation
        os.makedirs(f'{vis_dir}', exist_ok=True)

    def save_nbv_res(self, idx, gt_depth, gt_color, render_depth, render_graspness, gt_graspness):
        gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
        gt_color_np = gt_color.squeeze(0).cpu().numpy()

        render_depth = (render_depth.squeeze(0))[0:720:4, 0:1280:4].view(1, 1, 180, 320)
        render_depth = F.interpolate(render_depth, scale_factor=4, mode='bilinear').squeeze()
        render_depth_np = render_depth.cpu().numpy()

        gt_graspness = (gt_graspness.squeeze(0))[0:720:4, 0:1280:4].view(1, 1, 180, 320)
        gt_graspness = F.interpolate(gt_graspness, scale_factor=4, mode='bilinear').squeeze()

        gt_graspness_np = gt_graspness.cpu().numpy()



        render_graspness = render_graspness[0:720:4, 0:1280:4].view(1,1,180,320)
        render_graspness = F.interpolate(render_graspness, scale_factor=4, mode='bilinear').squeeze()
        render_graspness_np = render_graspness.detach().cpu().numpy()
        render_graspness_residual = np.abs(gt_graspness_np - render_graspness_np)
        render_graspness_residual[gt_depth_np == 0.0] = 0.0


        fig, axs = plt.subplots(2, 3)
        fig.tight_layout()
        max_depth = np.max(gt_depth_np)
        axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
        axs[0, 0].set_title('Input Depth')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(render_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
        axs[0, 1].set_title('Render Depth')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(gt_color_np)
        axs[0, 2].set_title('Input Color')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])


        gt_graspness_np = np.clip(gt_graspness_np, 0, 1)
        render_graspness_np = np.clip(render_graspness_np, 0, 1)
        render_graspness_residual = np.clip(render_graspness_residual, 0, 1)
        axs[1, 0].imshow(gt_graspness_np, cmap="plasma", vmin=0, vmax=1)
        axs[1, 0].set_title('GT Net Graspness')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(render_graspness_np, cmap="plasma", vmin=0, vmax=1)
        axs[1, 1].set_title('Ren Graspness')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 2].imshow(render_graspness_residual, cmap="plasma", vmin=0, vmax=1)
        axs[1, 2].set_title('Graspness Residual')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'{self.vis_dir}/nbv_{idx:05d}.jpg', bbox_inches='tight', pad_inches=0.2, dpi=300)
        plt.cla()
        plt.clf()
        if self.verbose:
            print(f'Saved rendering visualization of nbv at {self.vis_dir}/nbv_{idx:05d}.jpg')

    def save_imgs(self, idx, iter, gt_depth, gt_color, c2w_or_camera_tensor, all_planes, decoders, gt_graspness = None):
        """
        Visualization of depth and color images and save to file.
        Args:
            idx (int): current frame index.
            iter (int): the iteration number.
            gt_depth (tensor): ground truth depth image of the current frame.
            gt_color (tensor): ground truth color image of the current frame.
            c2w_or_camera_tensor (tensor): camera pose, represented in 
                camera to world matrix or quaternion and translation tensor.
            all_planes (Tuple): feature planes.
            decoders (torch.nn.Module): decoders for TSDF and color.
        """
        with torch.no_grad():
            if (idx % self.freq == 0) and (iter % self.inside_freq == 0):
                gt_depth_np = gt_depth.squeeze(0).cpu().numpy()
                gt_color_np = gt_color.squeeze(0).cpu().numpy()

                gt_graspness_np = gt_graspness.squeeze(0).cpu().numpy()

                if c2w_or_camera_tensor.shape[-1] > 4: ## 6od
                    c2w = cam_pose_to_matrix(c2w_or_camera_tensor.clone().detach()).squeeze()
                else:
                    c2w = c2w_or_camera_tensor.squeeze().detach()

                depth, color, graspness = self.renderer.render_img(all_planes, decoders, c2w, self.truncation,
                                                        self.device, gt_depth=gt_depth)
                depth_np = depth.detach().cpu().numpy()
                color_np = color.detach().cpu().numpy()
                graspness_np = graspness.detach().cpu().numpy()

                depth_residual = np.abs(gt_depth_np - depth_np)
                depth_residual[gt_depth_np == 0.0] = 0.0
                color_residual = np.abs(gt_color_np - color_np)
                color_residual[gt_depth_np == 0.0] = 0.0
                # gt_graspness_np = apply_pca_colormap_return_proj(gt_graspness.squeeze(0)).cpu().numpy()
                # graspness_np = apply_pca_colormap_return_proj(graspness).cpu().numpy()
                graspness_residual= np.abs(gt_graspness_np - graspness_np)

                graspness_residual[gt_depth_np == 0.0] = 0.0

                fig, axs = plt.subplots(3, 3)
                fig.tight_layout()
                max_depth = np.max(gt_depth_np)
                max_graspness = np.max(gt_graspness_np)

                axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 0].set_title('Input Depth')
                axs[0, 0].set_xticks([])
                axs[0, 0].set_yticks([])
                axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 1].set_title('Generated Depth')
                axs[0, 1].set_xticks([])
                axs[0, 1].set_yticks([])
                axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0, vmax=max_depth)
                axs[0, 2].set_title('Depth Residual')
                axs[0, 2].set_xticks([])
                axs[0, 2].set_yticks([])
                gt_color_np = np.clip(gt_color_np, 0, 1)
                color_np = np.clip(color_np, 0, 1)
                color_residual = np.clip(color_residual, 0, 1)
                axs[1, 0].imshow(gt_color_np, cmap="plasma")
                axs[1, 0].set_title('Input RGB')
                axs[1, 0].set_xticks([])
                axs[1, 0].set_yticks([])
                axs[1, 1].imshow(color_np, cmap="plasma")
                axs[1, 1].set_title('Generated RGB')
                axs[1, 1].set_xticks([])
                axs[1, 1].set_yticks([])
                axs[1, 2].imshow(color_residual, cmap="plasma")
                axs[1, 2].set_title('RGB Residual')
                axs[1, 2].set_xticks([])
                axs[1, 2].set_yticks([])

                gt_graspness_np = np.clip(gt_graspness_np, 0, 1)
                graspness_np = np.clip(graspness_np, 0, 1)
                graspness_residual = np.clip(graspness_residual, 0, 1)
                axs[2, 0].imshow(gt_graspness_np, cmap="plasma")
                axs[2, 0].set_title('Input Graspness')
                axs[2, 0].set_xticks([])
                axs[2, 0].set_yticks([])
                axs[2, 1].imshow(graspness_np, cmap="plasma")
                axs[2, 1].set_title('Generated Graspness')
                axs[2, 1].set_xticks([])
                axs[2, 1].set_yticks([])
                axs[2, 2].imshow(graspness_residual, cmap="plasma")
                axs[2, 2].set_title('Graspness Residual')
                axs[2, 2].set_xticks([])
                axs[2, 2].set_yticks([])

                plt.subplots_adjust(wspace=0, hspace=0)
                plt.savefig(f'{self.vis_dir}/{idx:05d}_{iter:04d}.jpg', bbox_inches='tight', pad_inches=0.2, dpi=300)
                plt.cla()
                plt.clf()

                if self.verbose:
                    print(f'Saved rendering visualization of color/depth image at {self.vis_dir}/{idx:05d}_{iter:04d}.jpg')


def apply_pca_colormap_return_proj(
    image,
    proj_V = None,
    low_rank_min = None,
    low_rank_max = None,
    niter = 5,
):
    """Convert a multichannel image to color using PCA.

    Args:
        image: Multichannel image.
        proj_V: Projection matrix to use. If None, use torch low rank PCA.

    Returns:
        Colored PCA image of the multichannel input image.
    """
    image_flat = image.reshape(-1, image.shape[-1])

    # Modified from https://github.com/pfnet-research/distilled-feature-fields/blob/master/train.py
    if proj_V is None:
        mean = image_flat.mean(0)
        with torch.no_grad():
            U, S, V = torch.pca_lowrank(image_flat - mean, niter=niter)
        proj_V = V[:, :3]

    low_rank = image_flat @ proj_V
    if low_rank_min is None:
        low_rank_min = torch.quantile(low_rank, 0.01, dim=0)
    if low_rank_max is None:
        low_rank_max = torch.quantile(low_rank, 0.99, dim=0)

    low_rank = (low_rank - low_rank_min) / (low_rank_max - low_rank_min)
    low_rank = torch.clamp(low_rank, 0, 1)

    colored_image = low_rank.reshape(image.shape[:-1] + (3,))
    return colored_image
