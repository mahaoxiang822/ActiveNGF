import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open3d as o3d

import os
import time
import numpy as np
from colorama import Fore, Style

from src.common import (get_samples, random_select, matrix_to_cam_pose, cam_pose_to_matrix)
from src.utils.datasets import get_dataset, SeqSampler
from src.utils.Frame_Visualizer import Frame_Visualizer
from src.tools.cull_mesh import cull_mesh


class Mapper(object):
    """
    Mapping main class.
    Args:
        cfg (dict): config dict
        args (argparse.Namespace): arguments
        eslam (ESLAM): ESLAM object
    """

    def __init__(self, cfg, args, eslam):

        self.cfg = cfg
        self.args = args

        self.idx = eslam.idx
        self.truncation = eslam.truncation
        self.bound = eslam.bound
        self.logger = eslam.logger
        self.wandb_run = eslam.wandb_run
        self.mesher = eslam.mesher
        self.output = eslam.output
        self.verbose = eslam.verbose
        self.renderer = eslam.renderer
        self.mapping_idx = eslam.mapping_idx
        self.mapping_cnt = eslam.mapping_cnt
        self.decoders = eslam.shared_decoders
        self.grasper = eslam.grasper

        self.planes_xy = eslam.shared_planes_xy
        self.planes_xz = eslam.shared_planes_xz
        self.planes_yz = eslam.shared_planes_yz

        self.c_planes_xy = eslam.shared_c_planes_xy
        self.c_planes_xz = eslam.shared_c_planes_xz
        self.c_planes_yz = eslam.shared_c_planes_yz

        self.g_planes_xy = eslam.shared_g_planes_xy
        self.g_planes_xz = eslam.shared_g_planes_xz
        self.g_planes_yz = eslam.shared_g_planes_yz

        self.estimate_c2w_list = eslam.estimate_c2w_list
        self.mapping_first_frame = eslam.mapping_first_frame

        self.scale = cfg['scale']
        self.device = cfg['device']
        self.keyframe_device = cfg['keyframe_device']

        self.eval_rec = cfg['meshing']['eval_rec']
        self.joint_opt = False  # Even if joint_opt is enabled, it starts only when there are at least 4 keyframes
        self.joint_opt_cam_lr = cfg['mapping']['joint_opt_cam_lr']  # The learning rate for camera poses during mapping
        self.mesh_freq = cfg['mapping']['mesh_freq']
        self.ckpt_freq = cfg['mapping']['ckpt_freq']
        self.mapping_pixels = cfg['mapping']['pixels']
        self.every_frame = cfg['mapping']['every_frame']
        self.w_sdf_fs = cfg['mapping']['w_sdf_fs']
        self.w_sdf_center = cfg['mapping']['w_sdf_center']
        self.w_sdf_tail = cfg['mapping']['w_sdf_tail']
        self.w_depth = cfg['mapping']['w_depth']
        self.w_color = cfg['mapping']['w_color']
        self.w_graspness = cfg['mapping']['w_graspness']
        self.keyframe_every = cfg['mapping']['keyframe_every']
        self.mapping_window_size = cfg['mapping']['mapping_window_size']
        self.no_vis_on_first_frame = cfg['mapping']['no_vis_on_first_frame']
        self.no_log_on_first_frame = cfg['mapping']['no_log_on_first_frame']
        self.no_mesh_on_first_frame = cfg['mapping']['no_mesh_on_first_frame']
        self.keyframe_selection_method = cfg['mapping']['keyframe_selection_method']
        self.max_step = 10

        self.keyframe_dict = []
        self.keyframe_list = []
        self.frame_reader = get_dataset(cfg, args, self.scale, device=self.device)
        self.n_img = len(self.frame_reader)
        self.frame_loader = DataLoader(self.frame_reader, batch_size=1, num_workers=1, pin_memory=True,
                                       prefetch_factor=2, sampler=SeqSampler(self.n_img, self.every_frame))

        self.visualizer = Frame_Visualizer(freq=cfg['mapping']['vis_freq'],
                                           inside_freq=cfg['mapping']['vis_inside_freq'],
                                           vis_dir=os.path.join(self.output, 'mapping_vis'), renderer=self.renderer,
                                           truncation=self.truncation, verbose=self.verbose, device=self.device)

        self.H, self.W, self.fx, self.fy, self.cx, self.cy = eslam.H, eslam.W, eslam.fx, eslam.fy, eslam.cx, eslam.cy
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.W,
            height=self.H,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy
        )

        self.planning_time = 0
        self.mapping_time = 0

        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.005,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

    def sdf_losses(self, sdf, z_vals, gt_depth):
        """
        Computes the losses for a signed distance function (SDF) given its values, depth values and ground truth depth.

        Args:
        - self: instance of the class containing this method
        - sdf: a tensor of shape (R, N) representing the SDF values
        - z_vals: a tensor of shape (R, N) representing the depth values
        - gt_depth: a tensor of shape (R,) containing the ground truth depth values

        Returns:
        - sdf_losses: a scalar tensor representing the weighted sum of the free space, center, and tail losses of SDF
        """

        front_mask = torch.where(z_vals < (gt_depth[:, None] - self.truncation),
                                 torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        back_mask = torch.where(z_vals > (gt_depth[:, None] + self.truncation),
                                torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        center_mask = torch.where((z_vals > (gt_depth[:, None] - 0.4 * self.truncation)) *
                                  (z_vals < (gt_depth[:, None] + 0.4 * self.truncation)),
                                  torch.ones_like(z_vals), torch.zeros_like(z_vals)).bool()

        tail_mask = (~front_mask) * (~back_mask) * (~center_mask)

        fs_loss = torch.mean(torch.square(sdf[front_mask] - torch.ones_like(sdf[front_mask])))
        center_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[center_mask] - gt_depth[:, None].expand(z_vals.shape)[center_mask]))
        tail_loss = torch.mean(torch.square(
            (z_vals + sdf * self.truncation)[tail_mask] - gt_depth[:, None].expand(z_vals.shape)[tail_mask]))

        sdf_losses = self.w_sdf_fs * fs_loss + self.w_sdf_center * center_loss + self.w_sdf_tail * tail_loss

        return sdf_losses

    def keyframe_selection_overlap(self, gt_color, gt_depth, c2w, num_keyframes, num_samples=8, num_rays=50,
                                   gt_graspness=None):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_color: ground truth color image of the current frame.
            gt_depth: ground truth depth image of the current frame.
            c2w: camera to world matrix for target view (3x4 or 4x4 both fine).
            num_keyframes (int): number of overlapping keyframes to select.
            num_samples (int, optional): number of samples/points per ray. Defaults to 8.
            num_rays (int, optional): number of pixels to sparsely sample
                from each image. Defaults to 50.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        device = self.device
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        rays_o, rays_d, gt_depth, gt_color, gt_graspness = get_samples(
            0, H, 0, W, num_rays, H, W, fx, fy, cx, cy,
            c2w.unsqueeze(0), gt_depth.unsqueeze(0), gt_color.unsqueeze(0), device,
            graspnesses=gt_graspness.unsqueeze(0))

        gt_depth = gt_depth.reshape(-1, 1)
        nonzero_depth = gt_depth[:, 0] > 0
        rays_o = rays_o[nonzero_depth]
        rays_d = rays_d[nonzero_depth]
        gt_depth = gt_depth[nonzero_depth]
        gt_depth = gt_depth.repeat(1, num_samples)
        t_vals = torch.linspace(0., 1., steps=num_samples).to(device)
        near = gt_depth * 0.8
        far = gt_depth + 0.5
        z_vals = near * (1. - t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [num_rays, num_samples, 3]
        pts = pts.reshape(1, -1, 3)

        keyframes_c2ws = torch.stack([self.estimate_c2w_list[idx] for idx in self.keyframe_list], dim=0)
        w2cs = torch.inverse(keyframes_c2ws[:-2])  ## The last two keyframes are already included

        ones = torch.ones_like(pts[..., 0], device=device).reshape(1, -1, 1)
        homo_pts = torch.cat([pts, ones], dim=-1).reshape(1, -1, 4, 1).expand(w2cs.shape[0], -1, -1, -1)
        w2cs_exp = w2cs.unsqueeze(1).expand(-1, homo_pts.shape[1], -1, -1)
        cam_cords_homo = w2cs_exp @ homo_pts
        cam_cords = cam_cords_homo[:, :, :3]
        K = torch.tensor([[fx, .0, cx], [.0, fy, cy],
                          [.0, .0, 1.0]], device=device).reshape(3, 3)
        cam_cords[:, :, 0] *= -1
        uv = K @ cam_cords
        z = uv[:, :, -1:] + 1e-5
        uv = uv[:, :, :2] / z
        edge = 20
        mask = (uv[:, :, 0] < W - edge) * (uv[:, :, 0] > edge) * \
               (uv[:, :, 1] < H - edge) * (uv[:, :, 1] > edge)
        mask = mask & (z[:, :, 0] < 0)
        mask = mask.squeeze(-1)
        percent_inside = mask.sum(dim=1) / uv.shape[1]

        ## Considering only overlapped frames
        selected_keyframes = torch.nonzero(percent_inside).squeeze(-1)
        rnd_inds = torch.randperm(selected_keyframes.shape[0])
        selected_keyframes = selected_keyframes[rnd_inds[:num_keyframes]]

        selected_keyframes = list(selected_keyframes.cpu().numpy())

        return selected_keyframes

    def optimize_mapping(self, iters, lr_factor, idx, cur_gt_color, cur_gt_depth, gt_cur_c2w, keyframe_dict,
                                  keyframe_list, cur_c2w, cur_gt_graspness=None):
        """
        Mapping iterations. Sample pixels from selected keyframes,
        then optimize scene representation and camera poses(if joint_opt enables).

        Args:
            iters (int): number of mapping iterations.
            lr_factor (float): the factor to times on current lr.
            idx (int): the index of current frame
            cur_gt_color (tensor): gt_color image of the current camera.
            cur_gt_depth (tensor): gt_depth image of the current camera.
            gt_cur_c2w (tensor): groundtruth camera to world matrix corresponding to current frame.
            keyframe_dict (list): a list of dictionaries of keyframes info.
            keyframe_list (list): list of keyframes indices.
            cur_c2w (tensor): the estimated camera to world matrix of current frame.

        Returns:
            cur_c2w: return the updated cur_c2w, return the same input cur_c2w if no joint_opt
        """
        all_planes = (
        self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
        self.g_planes_xy, self.g_planes_xz, self.g_planes_yz)
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        cfg = self.cfg
        device = self.device

        if len(keyframe_dict) == 0:
            optimize_frame = []
        else:
            if self.keyframe_selection_method == 'global':
                optimize_frame = random_select(len(self.keyframe_dict) - 2, self.mapping_window_size - 1)
            elif self.keyframe_selection_method == 'overlap':
                optimize_frame = self.keyframe_selection_overlap(cur_gt_color, cur_gt_depth, cur_c2w,
                                                                 self.mapping_window_size - 1,
                                                                 gt_graspness=cur_gt_graspness)
        # add the last two keyframes and the current frame(use -1 to denote)
        if len(keyframe_list) > 1:
            optimize_frame = optimize_frame + [len(keyframe_list) - 1] + [len(keyframe_list) - 2]
            optimize_frame = sorted(optimize_frame)
        optimize_frame += [-1]  ## -1 represents the current frame

        pixs_per_image = self.mapping_pixels // len(optimize_frame)

        decoders_para_list = []
        decoders_para_list += list(self.decoders.linears.parameters())
        decoders_para_list += list(self.decoders.output_linear.parameters())
        decoders_para_list += list(self.decoders.c_linears.parameters())
        decoders_para_list += list(self.decoders.c_output_linear.parameters())
        decoders_para_list.append(self.decoders.beta)

        g_decoders_para_list = []
        g_decoders_para_list += list(self.decoders.g_linears.parameters())
        g_decoders_para_list += list(self.decoders.g_output_linear.parameters())

        planes_para = []
        for planes in [self.planes_xy, self.planes_xz, self.planes_yz]:
            for i, plane in enumerate(planes):
                plane = nn.Parameter(plane)
                planes_para.append(plane)
                planes[i] = plane

        c_planes_para = []
        for c_planes in [self.c_planes_xy, self.c_planes_xz, self.c_planes_yz]:
            for i, c_plane in enumerate(c_planes):
                c_plane = nn.Parameter(c_plane)
                c_planes_para.append(c_plane)
                c_planes[i] = c_plane
        g_planes_para = []
        for g_planes in [self.g_planes_xy, self.g_planes_xz, self.g_planes_yz]:
            for i, g_plane in enumerate(g_planes):
                g_plane = nn.Parameter(g_plane)
                g_planes_para.append(g_plane)
                g_planes[i] = g_plane

        gt_depths = []
        gt_colors = []
        gt_graspness_list = []
        c2ws = []
        gt_c2ws = []
        for frame in optimize_frame:
            # the oldest frame should be fixed to avoid drifting
            if frame != -1:
                gt_depths.append(keyframe_dict[frame]['depth'].to(device))
                gt_colors.append(keyframe_dict[frame]['color'].to(device))
                gt_graspness_list.append(keyframe_dict[frame]['graspness'].to(device))
                c2ws.append(keyframe_dict[frame]['est_c2w'])
                gt_c2ws.append(keyframe_dict[frame]['gt_c2w'])
            else:
                gt_depths.append(cur_gt_depth)
                gt_colors.append(cur_gt_color)
                gt_graspness_list.append(cur_gt_graspness)
                c2ws.append(cur_c2w)
                gt_c2ws.append(gt_cur_c2w)
        gt_depths = torch.stack(gt_depths, dim=0)
        gt_colors = torch.stack(gt_colors, dim=0)
        gt_graspness = torch.stack(gt_graspness_list, dim=0)
        c2ws = torch.stack(c2ws, dim=0)
        optimizer = torch.optim.Adam([{'params': decoders_para_list, 'lr': 0},
                                      {'params': g_decoders_para_list, 'lr': 0},
                                      {'params': planes_para, 'lr': 0},
                                      {'params': c_planes_para, 'lr': 0},
                                      {'params': g_planes_para, 'lr': 0}])

        optimizer.param_groups[0]['lr'] = cfg['mapping']['lr']['decoders_lr'] * lr_factor
        optimizer.param_groups[2]['lr'] = cfg['mapping']['lr']['planes_lr'] * lr_factor
        optimizer.param_groups[3]['lr'] = cfg['mapping']['lr']['c_planes_lr'] * lr_factor
        # optimizer.param_groups[3]['lr'] = cfg['mapping']['lr']['planes_lr'] * lr_factor

        for joint_iter in range(iters):
            if (not (idx == 0 and self.no_vis_on_first_frame)):
                self.visualizer.save_imgs(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes,
                                          self.decoders, gt_graspness=cur_gt_graspness)
            c2ws_ = c2ws

            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, batch_gt_graspness = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors, device, gt_graspness)

            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
                t = (self.bound.unsqueeze(0).to(
                    device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_color = batch_gt_color[inside_mask]

            depth, color, graspness, sdf, z_vals, uncertainty = self.renderer.render_batch_ray(all_planes,
                                                                                               self.decoders,
                                                                                               batch_rays_d,
                                                                                               batch_rays_o, device,
                                                                                               self.truncation,
                                                                                               gt_depth=batch_gt_depth)
            depth_mask = (batch_gt_depth > 0)

            ## SDF losses
            sdf_loss = self.sdf_losses(sdf[depth_mask], z_vals[depth_mask], batch_gt_depth[depth_mask])
            self.wandb_run.log({"sdf_loss": sdf_loss})
            loss = sdf_loss

            ## Color loss
            # uncertainty = uncertainty.unsqueeze(-1)
            color_loss_ = torch.square(batch_gt_color - color)  # .mean()

            color_loss = torch.mean(0.5 * torch.log(uncertainty)) + torch.mean(
                0.5 * color_loss_ / (uncertainty.unsqueeze(-1))) + 4
            self.wandb_run.log({"color_loss": color_loss})
            loss = loss + self.w_color * color_loss * 0.01

            ### Depth loss
            depth_loss = torch.square(batch_gt_depth[depth_mask] - depth[depth_mask]).mean()
            self.wandb_run.log({"depth_loss": depth_loss})
            loss = loss + self.w_depth * depth_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

        optimizer.param_groups[1]['lr'] = cfg['mapping']['lr']['decoders_lr'] * lr_factor
        optimizer.param_groups[4]['lr'] = cfg['mapping']['lr']['planes_lr'] * lr_factor
        optimizer.param_groups[0]['lr'] = 0
        optimizer.param_groups[2]['lr'] = 0
        optimizer.param_groups[3]['lr'] = 0

        render_depth, _, _ = self.renderer.render_img(all_planes, self.decoders, cur_c2w, self.truncation,
                                                      self.device, gt_depth=cur_gt_depth)
        cur_gt_graspness, objectness_mask = self.grasper.inference(render_depth)
        # cur_gt_graspness, objectness_mask = self.grasper.inference(cur_gt_depth)
        cur_gt_graspness = cur_gt_graspness.squeeze(0)
        gt_graspness_list.pop(-1)
        gt_graspness_list.append(cur_gt_graspness)
        gt_graspness = torch.stack(gt_graspness_list, dim=0)

        for joint_iter in range(iters):
            if (not (idx == 0 and self.no_vis_on_first_frame)):
                self.visualizer.save_imgs(idx, joint_iter, cur_gt_depth, cur_gt_color, cur_c2w, all_planes,
                                          self.decoders, gt_graspness=cur_gt_graspness)
            c2ws_ = c2ws
            batch_rays_o, batch_rays_d, batch_gt_depth, batch_gt_color, batch_gt_graspness = get_samples(
                0, H, 0, W, pixs_per_image, H, W, fx, fy, cx, cy, c2ws_, gt_depths, gt_colors, device, gt_graspness)

            # should pre-filter those out of bounding box depth value
            with torch.no_grad():
                det_rays_o = batch_rays_o.clone().detach().unsqueeze(-1)
                det_rays_d = batch_rays_d.clone().detach().unsqueeze(-1)
                t = (self.bound.unsqueeze(0).to(
                    device) - det_rays_o) / det_rays_d
                t, _ = torch.min(torch.max(t, dim=2)[0], dim=1)
                inside_mask = t >= batch_gt_depth
            batch_rays_d = batch_rays_d[inside_mask]
            batch_rays_o = batch_rays_o[inside_mask]
            batch_gt_depth = batch_gt_depth[inside_mask]
            batch_gt_graspness = batch_gt_graspness[inside_mask]
            depth, color, graspness, sdf, z_vals, uncertainty = self.renderer.render_batch_ray(all_planes,
                                                                                               self.decoders,
                                                                                               batch_rays_d,
                                                                                               batch_rays_o, device,
                                                                                               self.truncation,
                                                                                               gt_depth=batch_gt_depth)
            graspness_loss = torch.square(batch_gt_graspness - graspness).mean()
            self.wandb_run.log({"graspness_loss": graspness_loss})
            loss = self.w_graspness * graspness_loss
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

        return cur_c2w, cur_gt_graspness

    def uncertainty_estimation(self, c2w, thresh=0.1):
        all_planes = (
            self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
            self.g_planes_xy, self.g_planes_xz, self.g_planes_yz)
        with torch.no_grad():
            depth, graspness, _ = self.renderer.render_img_downsample(all_planes, self.decoders, c2w,
                                                                                      self.truncation, self.device,
                                                                                      downsample_rate=4)
            net_graspness, objectness_mask = self.grasper.inference(depth)
        net_graspness = net_graspness.squeeze(0)
        objectness_mask = objectness_mask.squeeze(0)
        net_graspness_ = (net_graspness > thresh).float()
        graspness_ = (graspness > thresh).float()
        uncertainty = torch.mean(torch.abs(graspness_ - net_graspness_)[objectness_mask])
        return uncertainty, graspness, net_graspness, depth, objectness_mask

    def run(self):
        def run(self):
            """
            Runs the mapping thread for the input RGB-D frames.

            Args:
                None

            Returns:
                None
        """

        cfg = self.cfg
        all_planes = (
        self.planes_xy, self.planes_xz, self.planes_yz, self.c_planes_xy, self.c_planes_xz, self.c_planes_yz,
        self.g_planes_xy, self.g_planes_xz, self.g_planes_yz)
        idx, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
        data_iterator = iter(self.frame_loader)
        # skip first
        # next(data_iterator)
        ## Fixing the first camera pose
        self.estimate_c2w_list[0] = gt_c2w
        error_list = []
        init_phase = True
        prev_idx = -1
        while True:
            while True:
                idx = self.idx[0].clone()
                if idx == self.n_img - 1:  ## Last input frame
                    break

                if idx % self.every_frame == 0 and idx != prev_idx:
                    break

                time.sleep(0.001)

            prev_idx = idx - 1

            if self.verbose:
                print(Fore.GREEN)
                print("Mapping Frame ", idx.item())
                print(Style.RESET_ALL)
            if idx == 0:
                if cfg['model']['grasp_output'] == "offline":
                    _, gt_color, gt_depth, gt_graspness, gt_c2w = self.frame_reader[0]
                    gt_graspness = gt_graspness.squeeze(0).to(self.device, non_blocking=True)
                elif cfg['model']['grasp_output'] == "online":
                    _, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
                    gt_graspness, _ = self.grasper.inference(gt_depth)
                    gt_graspness = gt_graspness.squeeze(0)

                else:
                    _, gt_color, gt_depth, gt_c2w = self.frame_reader[0]
                    gt_graspness = None
            else:
                last_c2w = self.estimate_c2w_list[prev_idx].cpu()
                largest_uncertainty = -1000
                render_graspness = None
                render_depth = None
                sampled_poses, indexs = self.frame_reader.sample_pose_distance(last_c2w, 0.1)
                for i, pose in enumerate(sampled_poses):
                    uncertainty, graspness, net_graspness, depth, objectness_mask = self.uncertainty_estimation(
                        pose.to(self.device))
                    if uncertainty > largest_uncertainty:
                        largest_uncertainty = uncertainty
                        nbv_idx = indexs[i]
                        render_graspness = graspness
                        render_depth = depth

                self.frame_reader.mapped_frames.append(nbv_idx)
                _, gt_color, gt_depth, gt_c2w = self.frame_reader[nbv_idx]
                gt_graspness, _ = self.grasper.inference(gt_depth)
                gt_graspness = gt_graspness.squeeze(0)

            gt_color = gt_color.squeeze(0).to(self.device, non_blocking=True)
            gt_depth = gt_depth.squeeze(0).to(self.device, non_blocking=True)

            gt_c2w = gt_c2w.squeeze(0).to(self.device, non_blocking=True)

            cur_c2w = gt_c2w

            if not init_phase:
                lr_factor = cfg['mapping']['lr_factor']
                iters = cfg['mapping']['iters']
            else:
                lr_factor = cfg['mapping']['lr_first_factor']
                iters = cfg['mapping']['iters_first']

            ## Deciding if camera poses should be jointly optimized
            self.joint_opt = (len(self.keyframe_list) > 4) and cfg['mapping']['joint_opt']

            start_time = time.time()
            cur_c2w, cur_gt_graspness = self.optimize_mapping(iters, lr_factor, idx, gt_color, gt_depth,
                                                                       gt_c2w,
                                                                       self.keyframe_dict, self.keyframe_list, cur_c2w,
                                                                       cur_gt_graspness=gt_graspness)

            if idx!=0:
                self.visualizer.save_nbv_res(idx, gt_depth, gt_color, render_depth, render_graspness, cur_gt_graspness)
            self.wandb_run.log({"mapping_time":time.time()-start_time})
            self.mapping_time = (time.time() - start_time)

            self.estimate_c2w_list[idx] = cur_c2w
            if self.joint_opt:
                self.estimate_c2w_list[idx] = cur_c2w

            # add new frame to keyframe set
            if idx % self.keyframe_every == 0:
                self.keyframe_list.append(idx)
                self.keyframe_dict.append({'gt_c2w': gt_c2w, 'idx': idx, 'color': gt_color.to(self.keyframe_device),
                                           'depth': gt_depth.to(self.keyframe_device), 'est_c2w': cur_c2w.clone(),
                                           'graspness': gt_graspness.to(self.keyframe_device)})

            init_phase = False
            self.mapping_first_frame[0] = 1  # mapping of first frame is done, can begin tracking

            if ((not (
                    idx == 0 and self.no_log_on_first_frame)) and idx % self.ckpt_freq == 0) or idx == self.n_img - 1 or idx == self.max_step:
                self.logger.log(idx, self.keyframe_list)

            self.mapping_idx[0] = idx
            self.mapping_cnt[0] += 1
            if self.cfg["model"]["grasp_output"]:
                self.idx[0] = idx + 1

            if ((idx % self.mesh_freq == 0) and (not (idx == 0 and self.no_mesh_on_first_frame))) or (
                    idx == self.max_step):
                mesh_out_file = f'{self.output}/mesh/{idx:05d}_mesh.ply'
                self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device,
                                     color=False, graspness=True)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device,
                          estimate_c2w_list=self.estimate_c2w_list[:idx + 1])
                os.remove(mesh_out_file)

            if idx == self.n_img - 1:
                if self.eval_rec:
                    mesh_out_file = f'{self.output}/mesh/final_mesh_eval_rec.ply'
                else:
                    mesh_out_file = f'{self.output}/mesh/final_mesh.ply'

                self.mesher.get_mesh(mesh_out_file, all_planes, self.decoders, self.keyframe_dict, self.device,
                                     color=False, graspness=True)
                cull_mesh(mesh_out_file, self.cfg, self.args, self.device, estimate_c2w_list=self.estimate_c2w_list)
                os.remove(mesh_out_file)
                break

            if idx == self.max_step:
                break
