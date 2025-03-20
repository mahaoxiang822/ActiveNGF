import copy
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from MinkUnet import MinkUNet18,MinkUNet50, MinkUNet34_openshape, MinkUNetAdaptor, MinkResNet34

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(BASE_DIR, 'pointnet2'))
import pytorch_utils as pt_utils
from pointnet2_utils import furthest_point_sample
from pointnet2_utils import CylinderQueryAndGroup
from GraspNet.utils import batch_viewpoint_params_to_matrix,generate_grasp_views

device = torch.device('cuda:0')


class ApproachNet_view_fps_objectness(nn.Module):
    def __init__(self, num_view, seed_feature_dim):

        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)
        self.conv3 = nn.Conv1d(self.num_view, self.num_view, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(self.num_view)
        self.graspable_head = nn.Sequential(
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.in_dim, 3, 1),
        )

    def forward(self, seed_xyz, seed_features, end_points, is_training=False):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                seed_features: [torch.FloatTensor, (batch_size,feature_dim,num_seed)
                    features of seed points
                end_points: [dict]

            Output:
                end_points: [dict]
        """

        B, _, _ = seed_xyz.size()
        end_points['fp2_xyz'] = seed_xyz
        graspable = self.graspable_head(seed_features)
        objectness_score = graspable[:, :2]
        graspness_score = graspable[:, 2]

        # end_points['objectness_score'] = objectness_score
        end_points['graspness_score'] = graspness_score

        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        end_points['objectness_mask'] = objectness_mask
        if is_training:
            graspness_mask = (end_points['graspness_label'] > 0.1) & objectness_mask
        else:
            graspness_mask = (graspness_score > 0.1) & objectness_mask

        graspable_inds_list = []
        for i in range(B):
            graspable_points = seed_xyz[i][graspness_mask[i] == 1]
            sample_inds = furthest_point_sample(graspable_points.unsqueeze(0), 1024).long()
            inds = torch.where(graspness_mask[i] == 1)[0].unsqueeze(0)
            graspable_inds = torch.gather(inds, 1, sample_inds)
            graspable_inds_list.append(graspable_inds)
        graspable_inds = torch.cat(graspable_inds_list, dim=0)
        graspable_xyz = torch.gather(seed_xyz, 1, graspable_inds.unsqueeze(2).repeat(1, 1, 3))
        graspable_features = torch.gather(seed_features.permute(0, 2, 1), 1,
                                          graspable_inds.unsqueeze(2).repeat(1, 1, 256))
        graspable_features = graspable_features.permute(0, 2, 1)
        _, num_seed, _ = graspable_xyz.size()

        end_points['fp2_xyz'] = graspable_xyz
        end_points['fp2_inds'] = graspable_inds
        end_points['fp2_features'] = graspable_features
        fp2_graspness = torch.gather(graspness_score, 1, graspable_inds)
        end_points['fp2_graspness'] = fp2_graspness

        # ###########
        # end_points = process_grasp_labels(end_points)
        # view_score = end_points['batch_grasp_view_label']
        # ###########

        features = F.relu(self.bn1(self.conv1(graspable_features)), inplace=True)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)
        features = self.conv3(features)
        view_score = features.transpose(1, 2).contiguous()  # (B, num_seed, num_view)
        end_points['view_score'] = view_score
        top_view_scores, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)

        top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
        # generate template approach on sphere
        template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)

        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1,
                                                                            -1).contiguous()  # (B, num_seed, num_view, 3)
        # select the class of best approach
        vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
        vp_xyz_ = vp_xyz.view(-1, 3)

        # no rotation here
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        # transfer approach to 3x3
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_top_view_score'] = top_view_scores
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = vp_rot
        return end_points


class ApproachNet_view_fps(nn.Module):
    def __init__(self, num_view, seed_feature_dim):

        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, self.num_view, 1)
        self.conv3 = nn.Conv1d(self.num_view, self.num_view, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.bn2 = nn.BatchNorm1d(self.num_view)
        self.graspable_head = nn.Sequential(
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.in_dim, 1, 1),
            # nn.Conv1d(self.in_dim, 3, 1),
        )

    def forward(self, seed_xyz, seed_features, end_points, is_training=False):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                seed_features: [torch.FloatTensor, (batch_size,feature_dim,num_seed)
                    features of seed points
                end_points: [dict]

            Output:
                end_points: [dict]
        """

        B, _, _ = seed_xyz.size()
        end_points['fp2_xyz'] = seed_xyz
        graspable = self.graspable_head(seed_features)
        # objectness_score = graspable[:, :2]
        # graspness_score = graspable[:, 2]
        graspness_score = graspable[:, 0]

        # end_points['objectness_score'] = objectness_score
        end_points['graspness_score'] = graspness_score

        graspness_mask = (graspness_score > 0.09) #& objectness_mask
        graspable_inds_list = []
        for i in range(B):
            graspable_points = seed_xyz[i][graspness_mask[i] == 1]
            sample_inds = furthest_point_sample(graspable_points.unsqueeze(0), 1024).long()
            inds = torch.where(graspness_mask[i] == 1)[0].unsqueeze(0)
            graspable_inds = torch.gather(inds, 1, sample_inds)
            graspable_inds_list.append(graspable_inds)
        graspable_inds = torch.cat(graspable_inds_list, dim=0)
        graspable_xyz = torch.gather(seed_xyz, 1, graspable_inds.unsqueeze(2).repeat(1, 1, 3))
        graspable_features = torch.gather(seed_features.permute(0, 2, 1), 1,
                                          graspable_inds.unsqueeze(2).repeat(1, 1, 256))
        graspable_features = graspable_features.permute(0, 2, 1)
        _, num_seed, _ = graspable_xyz.size()

        end_points['fp2_xyz'] = graspable_xyz
        end_points['fp2_inds'] = graspable_inds
        end_points['fp2_features'] = graspable_features
        fp2_graspness = torch.gather(graspness_score, 1, graspable_inds)
        end_points['fp2_graspness'] = fp2_graspness

        # ###########
        # end_points = process_grasp_labels(end_points)
        # view_score = end_points['batch_grasp_view_label']
        # ###########

        features = F.relu(self.bn1(self.conv1(graspable_features)), inplace=True)
        features = F.relu(self.bn2(self.conv2(features)), inplace=True)
        features = self.conv3(features)
        view_score = features.transpose(1, 2).contiguous()  # (B, num_seed, num_view)
        end_points['view_score'] = view_score
        top_view_scores, top_view_inds = torch.max(view_score, dim=2)  # (B, num_seed)

        top_view_inds_ = top_view_inds.view(B, num_seed, 1, 1).expand(-1, -1, -1, 3).contiguous()
        # generate template approach on sphere
        template_views = generate_grasp_views(self.num_view).to(features.device)  # (num_view, 3)

        template_views = template_views.view(1, 1, self.num_view, 3).expand(B, num_seed, -1,
                                                                            -1).contiguous()  # (B, num_seed, num_view, 3)

        # select the class of best approach
        vp_xyz = torch.gather(template_views, 2, top_view_inds_).squeeze(2)  # (B, num_seed, 3)
        vp_xyz_ = vp_xyz.view(-1, 3)

        # no rotation here
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        # transfer approach to 3x3
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)

        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_top_view_score'] = top_view_scores
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = vp_rot
        return end_points


class ApproachNet_regression_view_fps(nn.Module):
    def __init__(self, num_view, seed_feature_dim):

        super().__init__()
        self.num_view = num_view
        self.in_dim = seed_feature_dim
        self.conv1 = nn.Conv1d(self.in_dim, self.in_dim, 1)
        self.conv2 = nn.Conv1d(self.in_dim, 3, 1)
        self.bn1 = nn.BatchNorm1d(self.in_dim)
        self.graspable_head = nn.Sequential(
            nn.Conv1d(self.in_dim, self.in_dim, 1),
            nn.BatchNorm1d(self.in_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.in_dim, 3, 1),
        )

    def forward(self, seed_xyz, seed_features, end_points, nerf_graspness = None):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                seed_features: [torch.FloatTensor, (batch_size,feature_dim,num_seed)
                    features of seed points
                end_points: [dict]

            Output:
                end_points: [dict]
        """

        B, _, _ = seed_xyz.size()
        end_points['fp2_xyz'] = seed_xyz
        graspable = self.graspable_head(seed_features)
        objectness_score = graspable[:, :2]
        if nerf_graspness is None:
            graspness_score = graspable[:, 2]
        else:
            graspness_score = nerf_graspness
        end_points['objectness_score'] = objectness_score
        end_points['graspness_score'] = graspness_score

        objectness_pred = torch.argmax(objectness_score, 1)
        objectness_mask = (objectness_pred == 1)
        graspness_mask = (graspness_score > 0.1) & objectness_mask

        graspable_inds_list = []
        for i in range(B):
            graspable_points = seed_xyz[i][graspness_mask[i] == 1]
            sample_inds = furthest_point_sample(graspable_points.unsqueeze(0), 1024).long()
            inds = torch.where(graspness_mask[i] == 1)[0].unsqueeze(0)
            graspable_inds = torch.gather(inds, 1, sample_inds)
            graspable_inds_list.append(graspable_inds)
        graspable_inds = torch.cat(graspable_inds_list, dim=0)

        # graspable_inds = self.object_balance_sampling(end_points,samples_per_object=128)

        graspable_xyz = torch.gather(seed_xyz, 1, graspable_inds.unsqueeze(2).repeat(1, 1, 3))
        graspable_features = torch.gather(seed_features.permute(0, 2, 1), 1,
                                          graspable_inds.unsqueeze(2).repeat(1, 1, 256))
        graspable_features = graspable_features.permute(0, 2, 1)
        _, num_seed, _ = graspable_xyz.size()

        end_points['fp2_xyz'] = graspable_xyz
        end_points['fp2_inds'] = graspable_inds
        end_points['fp2_features'] = graspable_features
        fp2_graspness = torch.gather(graspness_score, 1, graspable_inds)
        end_points['fp2_graspness'] = fp2_graspness
        features = F.relu(self.bn1(self.conv1(graspable_features)), inplace=True)
        features = self.conv2(features)

        vp_xyz = features.transpose(1, 2).contiguous()  # (B, num_seed, 3)
        end_points['view_prediction'] = vp_xyz

        template_views = generate_grasp_views(300).to(seed_features.device)  # (num_view, 3)
        template_views = template_views.view(1, 1, 300, 3).expand(B, num_seed, -1, -1).contiguous()  # (B, num_seed, num_view, 3)

        top_view_inds = torch.argmax(torch.cosine_similarity(template_views, vp_xyz.unsqueeze(2), dim=-1), dim=2)
        vp_xyz_ = vp_xyz.view(-1, 3)

        # no rotation here
        batch_angle = torch.zeros(vp_xyz_.size(0), dtype=vp_xyz.dtype, device=vp_xyz.device)
        # transfer approach to 3x3
        vp_rot = batch_viewpoint_params_to_matrix(-vp_xyz_, batch_angle).view(B, num_seed, 3, 3)
        end_points['grasp_top_view_inds'] = top_view_inds
        end_points['grasp_top_view_xyz'] = vp_xyz
        end_points['grasp_top_view_rot'] = vp_rot
        return end_points


class GraspNetStage1(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300):
        super().__init__()
        self.backbone = MinkUNet18(in_channels=3,out_channels=256)
        # self.vpmodule = ApproachNet_reg_cls_view_fps(num_view, 256)
        self.vpmodule = ApproachNet_regression_view_fps(num_view,256)
        # self.vpmodule = ApproachNet_view_fps_objectness(num_view, 256)

    def forward(self, end_points, is_training= False):
        pointcloud = end_points['point_clouds']
        end_points['input_xyz'] = pointcloud
        B,num_points,_ = pointcloud.shape

        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        # mink_input_coor = ME.SparseTensor(features_batch[:,:3], coordinate_map_key=mink_input.coordinate_map_key, coordinate_manager=mink_input.coordinate_manager)

        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, num_points, -1).transpose(1, 2)
        end_points['seed_features'] = seed_features
        end_points = self.vpmodule(pointcloud, seed_features, end_points)
        return end_points


class GraspNetStage1_only(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300):
        super().__init__()
        self.backbone = MinkUNet18(in_channels=3,out_channels=256)
        self.vpmodule = ApproachNet_view_fps_objectness(num_view, 256)

    def forward(self, end_points, is_training= False):
        pointcloud = end_points['point_clouds']
        end_points['input_xyz'] = pointcloud
        B,num_points,_ = pointcloud.shape

        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, num_points, -1).transpose(1, 2)
        end_points['seed_features'] = seed_features
        end_points = self.vpmodule(pointcloud, seed_features, end_points)
        return end_points


class GraspNetStage1_w_nerf(nn.Module):
    def __init__(self,input_feature_dim=0, num_view=300,):
        super().__init__()
        self.backbone = MinkUNet18(in_channels=3,out_channels=256)
        self.vpmodule = ApproachNet_regression_view_fps(num_view,256)

    def forward(self, end_points,decoders, all_planes, is_training= False):
        pointcloud = end_points['point_clouds']
        end_points['input_xyz'] = pointcloud
        B, num_points, _ = pointcloud.shape

        coordinates_batch = end_points['coors']
        features_batch = end_points['feats']
        mink_input = ME.SparseTensor(features_batch, coordinates=coordinates_batch)
        seed_features = self.backbone(mink_input).F
        seed_features = seed_features[end_points['quantize2original']].view(B, num_points, -1).transpose(1, 2)
        end_points['seed_features'] = seed_features
        with torch.no_grad():
            ret = decoders(pointcloud, all_planes=all_planes)
            nerf_graspness = ret[..., -1]
        end_points = self.vpmodule(pointcloud, seed_features, end_points, nerf_graspness=nerf_graspness)
        return end_points


class CloudCrop(nn.Module):
    """ Cylinder group and align for grasp configure estimation. Return a list of grouped points with different cropping depths.

        Input:
            nsample: [int]
                sample number in a group
            seed_feature_dim: [int]
                number of channels of grouped points
            cylinder_radius: [float]
                radius of the cylinder space
            hmin: [float]
                height of the bottom surface
            hmax_list: [list of float]
                list of heights of the upper surface
    """

    def __init__(self, nsample, seed_feature_dim, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04]):
        super().__init__()
        self.nsample = nsample
        self.in_dim = seed_feature_dim
        self.cylinder_radius = cylinder_radius
        mlps = [self.in_dim, 64, 128, 256]

        self.groupers = []
        for hmax in hmax_list:
            self.groupers.append(CylinderQueryAndGroup(
                cylinder_radius, hmin, hmax, nsample, use_xyz=True
            ))
        self.mlps = pt_utils.SharedMLP(mlps, bn=True)

    def forward(self, seed_xyz, pointcloud, vp_rot, features=None, ret_points = False):
        """ Forward pass.

            Input:
                seed_xyz: [torch.FloatTensor, (batch_size,num_seed,3)]
                    coordinates of seed points
                pointcloud: [torch.FloatTensor, (batch_size,num_seed,3)]
                    the points to be cropped
                vp_rot: [torch.FloatTensor, (batch_size,num_seed,3,3)]
                    rotation matrices generated from approach vectors
                feature: [torch.FloatTensor, (batch_size,c, num_seed)]
                    feature such as normal

            Output:
                vp_features: [torch.FloatTensor, (batch_size,num_features,num_seed,num_depth)]
                    features of grouped points in different depths
        """
        B, num_seed, _, _ = vp_rot.size()
        num_depth = len(self.groupers)
        grouped_features = []
        for grouper in self.groupers:
            grouped_features.append(grouper(
                pointcloud, seed_xyz, vp_rot, features
            ))  # (batch_size, feature_dim, num_seed, nsample)
        grouped_features = torch.stack(grouped_features,
                                       dim=3)  # (batch_size, feature_dim, num_seed, num_depth, nsample)
        grouped_features = grouped_features.view(B, -1, num_seed * num_depth,
                                                 self.nsample)  # (batch_size, feature_dim, num_seed*num_depth, nsample)
        if ret_points:
            return  grouped_features.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.nsample, self.in_dim)
        vp_features = self.mlps(
            grouped_features
        )  # (batch_size, mlps[-1], num_seed*num_depth, nsample)
        vp_features = F.max_pool2d(
            vp_features, kernel_size=[1, vp_features.size(3)]
        )  # (batch_size, mlps[-1], num_seed*num_depth, 1)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        return vp_features


class GraspNetStage2_seed_features_multi_scale(nn.Module):
    def __init__(self, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                 is_training=True):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training

        self.crop1 = CloudCrop(64, 3, cylinder_radius * 0.25, hmin, hmax_list)
        self.crop2 = CloudCrop(64, 3, cylinder_radius * 0.5, hmin, hmax_list)
        self.crop3 = CloudCrop(64, 3, cylinder_radius * 0.75, hmin, hmax_list)
        self.crop4 = CloudCrop(64, 3, cylinder_radius, hmin, hmax_list)
        self.operation = OperationNet_regression(num_angle)
        self.tolerance = ToleranceNet_regression(num_angle)
        self.fuse_multi_scale = nn.Conv1d(256 * 4, 256, 1)
        self.gate_fusion = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.Sigmoid()
        )

    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        grasp_top_views_rot = end_points['grasp_top_view_rot']
        seed_xyz = end_points['fp2_xyz']
        grasp_top_views_rot_ = grasp_top_views_rot.detach()
        vp_features1 = self.crop1(seed_xyz, pointcloud, grasp_top_views_rot_)
        vp_features2 = self.crop2(seed_xyz, pointcloud, grasp_top_views_rot_)
        vp_features3 = self.crop3(seed_xyz, pointcloud, grasp_top_views_rot_)
        vp_features4 = self.crop4(seed_xyz, pointcloud, grasp_top_views_rot_)
        B, _, num_seed, num_depth = vp_features1.size()
        vp_features_concat = torch.cat([vp_features1, vp_features2, vp_features3, vp_features4], dim=1)
        vp_features_concat = vp_features_concat.view(B, -1, num_seed * num_depth)
        vp_features_concat = self.fuse_multi_scale(vp_features_concat)
        vp_features_concat = vp_features_concat.view(B, -1, num_seed, num_depth)
        seed_features = end_points['fp2_features']
        seed_features_gate = self.gate_fusion(seed_features) * seed_features
        seed_features_gate = seed_features_gate.unsqueeze(3).repeat(1, 1, 1, 4)
        vp_features = vp_features_concat + seed_features_gate
        end_points = self.operation(vp_features, end_points)
        end_points = self.tolerance(vp_features, end_points)
        return end_points


class ToleranceNet(nn.Module):
    """ Grasp tolerance prediction.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """

    def __init__(self, num_angle, num_depth):
        # Output:
        # tolerance (num_angle)
        super().__init__()
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed * num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        end_points['grasp_tolerance_pred'] = vp_features
        return end_points  # ,vp_features


class OperationNet(nn.Module):
    """ Grasp configure estimation.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """

    def __init__(self, num_angle, num_depth):
        # Output:
        # scores(num_angle)
        # angle class (num_angle)
        # width (num_angle)
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 3 * num_angle, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed * num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        end_points['grasp_score_pred'] = vp_features[:, 0:self.num_angle]
        end_points['grasp_angle_cls_pred'] = vp_features[:, self.num_angle:2 * self.num_angle]
        end_points['grasp_width_pred'] = vp_features[:, 2 * self.num_angle:3 * self.num_angle]
        return end_points


class ToleranceNet_regression(nn.Module):
    """ Grasp tolerance prediction.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """

    def __init__(self, num_depth):
        # Output:
        # tolerance (num_angle)
        super().__init__()
        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 1, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()
        vp_features = vp_features.view(B, -1, num_seed * num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, num_seed, num_depth)
        end_points['grasp_tolerance_pred'] = vp_features
        return end_points  # ,vp_features


class OperationNet_regression(nn.Module):
    """ Grasp configure estimation.

        Input:
            num_angle: [int]
                number of in-plane rotation angle classes
                the value of the i-th class --> i*PI/num_angle (i=0,...,num_angle-1)
            num_depth: [int]
                number of gripper depth classes
    """

    def __init__(self, num_depth):
        # Output:
        # scores(num_angle)
        # angle class (num_angle)
        # width (num_angle)
        super().__init__()
        self.num_depth = num_depth

        self.conv1 = nn.Conv1d(256, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 4, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, vp_features, end_points):
        """ Forward pass.

            Input:
                vp_features: [torch.FloatTensor, (batch_size,num_seed,3)]
                    features of grouped points in different depths
                end_points: [dict]

            Output:
                end_points: [dict]
        """
        B, _, num_seed, num_depth = vp_features.size()

        vp_features = vp_features.view(B, -1, num_seed * num_depth)
        vp_features = F.relu(self.bn1(self.conv1(vp_features)), inplace=True)
        vp_features = F.relu(self.bn2(self.conv2(vp_features)), inplace=True)
        vp_features = self.conv3(vp_features)
        vp_features = vp_features.view(B, -1, num_seed, num_depth)
        end_points['grasp_score_pred'] = vp_features[:, 0]
        # eps = 1e-8
        end_points['grasp_angle_pred'] = F.tanh(vp_features[:, 1:3])
        sin2theta = F.tanh(vp_features[:, 1])
        cos2theta = F.tanh(vp_features[:, 2])
        angle = 0.5 * torch.atan2(sin2theta, cos2theta)
        angle[angle < 0] += np.pi
        # sin2theta = torch.clamp(vp_features[:, 1], min=-1.0+eps, max=1.0-eps)
        # cos2theta = torch.clamp(vp_features[:, 1], min=-1.0 + eps, max=1.0 - eps)
        end_points['grasp_angle_value_pred'] = angle
        end_points['grasp_width_pred'] = F.sigmoid(vp_features[:, 3]) * 0.1
        return end_points


class GraspNet_MSCQ(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.08, hmin=-0.02,
                 hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=True):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = GraspNetStage1(input_feature_dim, num_view)
        self.grasp_generator = GraspNetStage2_seed_features_multi_scale(num_angle, num_depth, cylinder_radius, hmin,
                                                                            hmax_list, is_training)

    def forward(self, end_points):
        end_points = self.view_estimator(end_points,self.is_training)
        end_points = self.grasp_generator(end_points)
        return end_points


class GraspNet1(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.08, hmin=-0.02,
                 hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=True):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = GraspNetStage1_only(input_feature_dim, num_view)

    def forward(self, end_points):
        end_points = self.view_estimator(end_points,self.is_training)
        return end_points



class GraspNet_MSCQ_nerf(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.08, hmin=-0.02,
                 hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=True, ):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = GraspNetStage1_w_nerf(input_feature_dim, num_view)
        self.grasp_generator = GraspNetStage2_seed_features_multi_scale(num_angle, num_depth, cylinder_radius, hmin,
                                                                            hmax_list, is_training)

    def forward(self, end_points, decoders, planes):
        end_points = self.view_estimator(end_points, decoders, planes, self.is_training)
        end_points = self.grasp_generator(end_points)
        return end_points


def pred_decode(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        ## load predictions
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_center = end_points['fp2_xyz'][i].float()
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]
        grasp_width = torch.clamp(grasp_width, min=0, max=0.1)
        grasp_tolerance = end_points['grasp_tolerance_pred'][i]

        ## slice preds by angle
        # grasp angle
        grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
        grasp_angle = grasp_angle_class.float() / 12 * np.pi
        # grasp score & width & tolerance
        grasp_angle_class_ = grasp_angle_class.unsqueeze(0)
        grasp_score = torch.gather(grasp_score, 0, grasp_angle_class_).squeeze(0)
        grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)
        grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_angle_class_).squeeze(0)

        ## slice preds by score/depth
        # grasp depth
        grasp_depth_class = torch.argmax(grasp_score, 1, keepdims=True)
        grasp_depth = (grasp_depth_class.float() + 1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_score = torch.gather(grasp_score, 1, grasp_depth_class)
        grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)
        grasp_width = torch.gather(grasp_width, 1, grasp_depth_class)
        grasp_tolerance = torch.gather(grasp_tolerance, 1, grasp_depth_class)

        grasp_score = grasp_score * grasp_tolerance / 0.05
        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.view(Ns, 3)
        grasp_angle_ = grasp_angle.view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.view(Ns, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids],
                      axis=-1))
    return grasp_preds


def pred_decode_reg(end_points):
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        ## load predictions
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_center = end_points['fp2_xyz'][i].float()
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        # approaching = -end_points['approach_refined'][i].float()
        grasp_angle = end_points['grasp_angle_value_pred'][i]
        grasp_width = 1.2*end_points['grasp_width_pred'][i]
        grasp_width = torch.clamp(grasp_width, min=0, max=0.1)
        grasp_tolerance = end_points['grasp_tolerance_pred'][i]
        graspness = end_points['fp2_graspness'][i]

        ## slice preds by score/depth
        # grasp depth
        grasp_depth_class = torch.argmax(grasp_score, 1, keepdims=True)
        grasp_depth = (grasp_depth_class.float() + 1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_score = torch.gather(grasp_score, 1, grasp_depth_class)
        grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)
        grasp_width = torch.gather(grasp_width, 1, grasp_depth_class)
        grasp_tolerance = torch.gather(grasp_tolerance, 1, grasp_depth_class)
        # grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE

        grasp_score = grasp_score * graspness.unsqueeze(1)

        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.view(Ns, 3)
        grasp_angle_ = grasp_angle.view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.view(Ns, 9)
        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(
            torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids],
                      axis=-1))
    return grasp_preds