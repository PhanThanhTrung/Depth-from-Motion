# Copyright (c) OpenMMLab. All rights reserved.
import random
import warnings

import cv2
import mmcv
import numpy as np
import torch
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
from mmdet.datasets.pipelines import RandomCrop, RandomFlip, Resize
from ..builder import OBJECTSAMPLERS, PIPELINES
from ..utils import (boxes3d_kitti_camera_to_imageboxes,
                     boxes3d_kitti_camera_to_imagecenters,
                     boxes3d_lidar_to_kitti_camera, depth_map_in_boxes_cpu)
from .data_augment_utils import noise_per_object_v3_


@PIPELINES.register_module()
class GenerateAmodal2DBoxes(object):
    """Generate depth map by projecting LiDAR points."""

    def __init__(self):
        pass

    def __call__(self, input_dict):
        # the input boxes of boxes_3d_lidar_to_kitti_camera
        # should set the gravity center as the first 3 dimensions
        gt_centers_3d_numpy = input_dict['gt_bboxes_3d'].gravity_center.cpu(
        ).numpy()
        gt_bboxes_3d_numpy = input_dict['gt_bboxes_3d'].tensor.cpu().numpy()
        gt_bboxes_3d_numpy = np.concatenate(
            [gt_centers_3d_numpy, gt_bboxes_3d_numpy[:, 3:]], axis=-1)
        gt_bboxes_3d_cam = boxes3d_lidar_to_kitti_camera(
            gt_bboxes_3d_numpy, calib=None, pseudo_lidar=True)
        input_dict['gt_bboxes'] = boxes3d_kitti_camera_to_imageboxes(
            gt_bboxes_3d_cam,
            input_dict['calib'],
            input_dict['img_shape'],
            fix_neg_z_bug=True)
        input_dict['centers2d'] = boxes3d_kitti_camera_to_imagecenters(
            gt_bboxes_3d_cam, input_dict['calib'], input_dict['img_shape'])
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class GenerateDepthMap(object):
    """Generate depth map by projecting LiDAR points."""

    def __init__(self, generate_fgmask=True):
        self.generate_fgmask = generate_fgmask

    def __call__(self, input_dict):
        # Note: the input points should be not in the real lidar system
        # instead it is in the pseudo lidar system
        # which shares the same direction of xyz but different origin
        pseudo_points = (input_dict['points'].tensor)[:, :3].numpy()
        cam2img = input_dict['cam2img']
        pseudo2cam_T = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, 0]],
                                dtype=np.float32)
        homo_pseudo_points = np.concatenate(
            [pseudo_points,
             np.ones([pseudo_points.shape[0], 1])], axis=-1)
        rect_points = homo_pseudo_points @ pseudo2cam_T
        """
        # TODO: replace the LIGA projection with the accurate points_cam2img
        cam_points = points_cam2img(rect_points, cam2img, with_depth=True)
        img_coords = cam_points[:, :2]
        depths = cam_points[:, 2]
        # TODO: check img_shape
        depth_img = np.zeros(input_dict['pad_shape'][:2], dtype=np.float32)
        iy = np.round(img_coords[:, 1]).astype(np.int64)
        ix = np.round(img_coords[:, 0]).astype(np.int64)
        mask = (iy >= 0) & (ix >= 0) & (iy < depth_img.shape[0]) & (
            ix < depth_img.shape[1])
        iy, ix = iy[mask], ix[mask]
        depth_img[iy, ix] = depths[mask]
        input_dict['depth_img'] = depth_img
        """
        calib = input_dict['calib']
        img_coords, depths = calib.rect_to_img(rect_points[:, :3])
        # TODO: check img_shape
        depth_img = np.zeros(input_dict['pad_shape'][:2], dtype=np.float32)
        iy = np.round(img_coords[:, 1]).astype(np.int64)
        ix = np.round(img_coords[:, 0]).astype(np.int64)
        # depth > 0 mask in LIGA merged here
        img_shape = input_dict['img_shape'][:2]
        mask = (iy >= 0) & (ix >= 0) & (iy < img_shape[0]) & (ix <
                                                              img_shape[1]) & (
                                                                  depths > 0)
        iy, ix = iy[mask], ix[mask]
        depth_img[iy, ix] = depths[mask]
        input_dict['depth_img'] = depth_img

        if self.generate_fgmask:
            # TODO: add pipeline for generating amodal 2D gt
            input_dict['depth_fgmask_img'] = depth_map_in_boxes_cpu(
                input_dict['depth_img'],
                input_dict['gt_bboxes_3d'].tensor.cpu().numpy()[:, :7],
                cam2img,
                expand_distance=0.,
                expand_ratio=1.0)
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(generate_fgmask={self.generate_fgmask}.'
        return repr_str


@PIPELINES.register_module()
class RandomDropPointsColor(object):
    r"""Randomly set the color of points to all zeros.

    Once this transform is executed, all the points' color will be dropped.
    Refer to `PAConv <https://github.com/CVMI-Lab/PAConv/blob/main/scene_seg/
    util/transform.py#L223>`_ for more details.

    Args:
        drop_ratio (float, optional): The probability of dropping point colors.
            Defaults to 0.2.
    """

    def __init__(self, drop_ratio=0.2):
        assert isinstance(drop_ratio, (int, float)) and 0 <= drop_ratio <= 1, \
            f'invalid drop_ratio value {drop_ratio}'
        self.drop_ratio = drop_ratio

    def __call__(self, input_dict):
        """Call function to drop point colors.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after color dropping,
                'points' key is updated in the result dict.
        """
        points = input_dict['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims, \
            'Expect points have color attribute'

        # this if-expression is a bit strange
        # `RandomDropPointsColor` is used in training 3D segmentor PAConv
        # we discovered in our experiments that, using
        # `if np.random.rand() > 1.0 - self.drop_ratio` consistently leads to
        # better results than using `if np.random.rand() < self.drop_ratio`
        # so we keep this hack in our codebase
        if np.random.rand() > 1.0 - self.drop_ratio:
            points.color = points.color * 0.0
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(drop_ratio={self.drop_ratio})'
        return repr_str


@PIPELINES.register_module()
class RandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 with_baseline_offset=False,
                 **kwargs):
        super(RandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        self.with_baseline_offset = with_baseline_offset
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str, optional): Flip direction.
                Default: 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        # for semantic segmentation task, only points will be flipped.
        if 'bbox3d_fields' not in input_dict:
            input_dict['points'].flip(direction)
            return
        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))
        assert len(input_dict['bbox3d_fields']) == 1
        for key in input_dict['bbox3d_fields']:
            if 'points' in input_dict:
                input_dict['points'] = input_dict[key].flip(
                    direction, points=input_dict['points'])
            else:
                input_dict[key].flip(direction)
        # TODO: use img_shape or ori_shape according to the order of augs
        w = input_dict['img_shape'][1]
        # need to modify the horizontal position of camera center
        # along u-axis in the image (flip like centers2d)
        # ['cam2img'][0][2] = c_u
        # see more details and examples at
        # https://github.com/open-mmlab/mmdetection3d/pull/744
        if 'cam2img' in input_dict:
            if self.with_baseline_offset:
                K = input_dict['cam2img'][:3, :3].copy()
                inv_K = np.linalg.inv(K)
                T2 = np.matmul(inv_K, input_dict['cam2img'][:3, :])
                T2[0, 3] *= -1
                K[0, 2] = w - 1 - K[0, 2]
                input_dict['cam2img'][:3, :] = np.matmul(K, T2)
            else:
                input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]
        if 'calib' in input_dict:
            input_dict['calib'].flipl(input_dict['img_shape'][1])
        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction',
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added
                into result dict.
        """
        # flip 2D image and its annotations
        super(RandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@PIPELINES.register_module()
class RandomJitterPoints(object):
    """Randomly jitter point coordinates.

    Different from the global translation in ``GlobalRotScaleTrans``, here we
        apply different noises to each point in a scene.

    Args:
        jitter_std (list[float]): The standard deviation of jittering noise.
            This applies random noise to all points in a 3D scene, which is
            sampled from a gaussian distribution whose standard deviation is
            set by ``jitter_std``. Defaults to [0.01, 0.01, 0.01]
        clip_range (list[float]): Clip the randomly generated jitter
            noise into this range. If None is given, don't perform clipping.
            Defaults to [-0.05, 0.05]

    Note:
        This transform should only be used in point cloud segmentation tasks
            because we don't transform ground-truth bboxes accordingly.
        For similar transform in detection task, please refer to `ObjectNoise`.
    """

    def __init__(self,
                 jitter_std=[0.01, 0.01, 0.01],
                 clip_range=[-0.05, 0.05]):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(jitter_std, seq_types):
            assert isinstance(jitter_std, (int, float)), \
                f'unsupported jitter_std type {type(jitter_std)}'
            jitter_std = [jitter_std, jitter_std, jitter_std]
        self.jitter_std = jitter_std

        if clip_range is not None:
            if not isinstance(clip_range, seq_types):
                assert isinstance(clip_range, (int, float)), \
                    f'unsupported clip_range type {type(clip_range)}'
                clip_range = [-clip_range, clip_range]
        self.clip_range = clip_range

    def __call__(self, input_dict):
        """Call function to jitter all the points in the scene.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after adding noise to each point,
                'points' key is updated in the result dict.
        """
        points = input_dict['points']
        jitter_std = np.array(self.jitter_std, dtype=np.float32)
        jitter_noise = \
            np.random.randn(points.shape[0], 3) * jitter_std[None, :]
        if self.clip_range is not None:
            jitter_noise = np.clip(jitter_noise, self.clip_range[0],
                                   self.clip_range[1])

        points.translate(jitter_noise)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(jitter_std={self.jitter_std},'
        repr_str += f' clip_range={self.clip_range})'
        return repr_str


@PIPELINES.register_module()
class ObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
        use_ground_plane (bool): Whether to use gound plane to adjust the
            3D labels.
    """

    def __init__(self, db_sampler, sample_2d=False, use_ground_plane=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.use_ground_plane = use_ground_plane

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation,
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated
                in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']

        if self.use_ground_plane and 'plane' in input_dict['ann_info']:
            ground_plane = input_dict['ann_info']['plane']
            input_dict['plane'] = ground_plane
        else:
            ground_plane = None
        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                gt_bboxes_3d.tensor.numpy(),
                gt_labels_3d,
                img=None,
                ground_plane=ground_plane)

        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        input_dict['points'] = points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


@PIPELINES.register_module()
class ObjectNoise(object):
    """Apply noise to each GT objects in the scene.

    Args:
        translation_std (list[float], optional): Standard deviation of the
            distribution where translation noise are sampled from.
            Defaults to [0.25, 0.25, 0.25].
        global_rot_range (list[float], optional): Global rotation to the scene.
            Defaults to [0.0, 0.0].
        rot_range (list[float], optional): Object rotation range.
            Defaults to [-0.15707963267, 0.15707963267].
        num_try (int, optional): Number of times to try if the noise applied is
            invalid. Defaults to 100.
    """

    def __init__(self,
                 translation_std=[0.25, 0.25, 0.25],
                 global_rot_range=[0.0, 0.0],
                 rot_range=[-0.15707963267, 0.15707963267],
                 num_try=100):
        self.translation_std = translation_std
        self.global_rot_range = global_rot_range
        self.rot_range = rot_range
        self.num_try = num_try

    def __call__(self, input_dict):
        """Call function to apply noise to each ground truth in the scene.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after adding noise to each object,
                'points', 'gt_bboxes_3d' keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points = input_dict['points']

        # TODO: this is inplace operation
        numpy_box = gt_bboxes_3d.tensor.numpy()
        numpy_points = points.tensor.numpy()

        noise_per_object_v3_(
            numpy_box,
            numpy_points,
            rotation_perturb=self.rot_range,
            center_noise_std=self.translation_std,
            global_random_rot_range=self.global_rot_range,
            num_try=self.num_try)

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d.new_box(numpy_box)
        input_dict['points'] = points.new_point(numpy_points)
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_try={self.num_try},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' global_rot_range={self.global_rot_range},'
        repr_str += f' rot_range={self.rot_range})'
        return repr_str


@PIPELINES.register_module()
class GlobalAlignment(object):
    """Apply global alignment to 3D scene points by rotation and translation.

    Args:
        rotation_axis (int): Rotation axis for points and bboxes rotation.

    Note:
        We do not record the applied rotation and translation as in
            GlobalRotScaleTrans. Because usually, we do not need to reverse
            the alignment step.
        For example, ScanNet 3D detection task uses aligned ground-truth
            bounding boxes for evaluation.
    """

    def __init__(self, rotation_axis):
        self.rotation_axis = rotation_axis

    def _trans_points(self, input_dict, trans_factor):
        """Private function to translate points.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            trans_factor (np.ndarray): Translation vector to be applied.

        Returns:
            dict: Results after translation, 'points' is updated in the dict.
        """
        input_dict['points'].translate(trans_factor)

    def _rot_points(self, input_dict, rot_mat):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            rot_mat (np.ndarray): Rotation matrix to be applied.

        Returns:
            dict: Results after rotation, 'points' is updated in the dict.
        """
        # input should be rot_mat_T so I transpose it here
        input_dict['points'].rotate(rot_mat.T)

    def _check_rot_mat(self, rot_mat):
        """Check if rotation matrix is valid for self.rotation_axis.

        Args:
            rot_mat (np.ndarray): Rotation matrix to be applied.
        """
        is_valid = np.allclose(np.linalg.det(rot_mat), 1.0)
        valid_array = np.zeros(3)
        valid_array[self.rotation_axis] = 1.0
        is_valid &= (rot_mat[self.rotation_axis, :] == valid_array).all()
        is_valid &= (rot_mat[:, self.rotation_axis] == valid_array).all()
        assert is_valid, f'invalid rotation matrix {rot_mat}'

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after global alignment, 'points' and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        assert 'axis_align_matrix' in input_dict['ann_info'].keys(), \
            'axis_align_matrix is not provided in GlobalAlignment'

        axis_align_matrix = input_dict['ann_info']['axis_align_matrix']
        assert axis_align_matrix.shape == (4, 4), \
            f'invalid shape {axis_align_matrix.shape} for axis_align_matrix'
        rot_mat = axis_align_matrix[:3, :3]
        trans_vec = axis_align_matrix[:3, -1]

        self._check_rot_mat(rot_mat)
        self._rot_points(input_dict, rot_mat)
        self._trans_points(input_dict, trans_vec)

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(rotation_axis={self.rotation_axis})'
        return repr_str


@PIPELINES.register_module()
class GlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float], optional): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float], optional): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float], optional): The standard deviation of
            translation noise applied to a scene, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool, optional): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T

        input_dict['points'].translate(trans_factor)
        input_dict['pcd_trans'] = trans_factor
        for key in input_dict['bbox3d_fields']:
            input_dict[key].translate(trans_factor)

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation'
                and keys in input_dict['bbox3d_fields'] are updated
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        if len(input_dict['bbox3d_fields']) == 0:
            rot_mat_T = input_dict['points'].rotate(noise_rotation)
            input_dict['pcd_rotation'] = rot_mat_T
            input_dict['pcd_rotation_angle'] = noise_rotation
            return

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            if len(input_dict[key].tensor) != 0:
                points, rot_mat_T = input_dict[key].rotate(
                    noise_rotation, input_dict['points'])
                input_dict['points'] = points
                input_dict['pcd_rotation'] = rot_mat_T
                input_dict['pcd_rotation_angle'] = noise_rotation

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']
        points = input_dict['points']
        points.scale(scale)
        if self.shift_height:
            assert 'height' in points.attribute_dims.keys(), \
                'setting shift_height=True but points have no height attribute'
            points.tensor[:, points.attribute_dims['height']] *= scale
        input_dict['points'] = points

        for key in input_dict['bbox3d_fields']:
            input_dict[key].scale(scale)

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        self._rot_bbox_points(input_dict)

        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        self._trans_bbox_points(input_dict)

        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'
        return repr_str


@PIPELINES.register_module()
class PointShuffle(object):
    """Shuffle input points."""

    def __call__(self, input_dict):
        """Call function to shuffle points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        idx = input_dict['points'].shuffle()
        idx = idx.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[idx]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[idx]

        return input_dict

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class ObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self,
                 point_cloud_range,
                 filter_scheme='bev',
                 min_num_corners=1):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        assert filter_scheme in ['bev', 'corner']
        self.filter_scheme = filter_scheme
        if filter_scheme == 'corner':
            assert min_num_corners is not None
            self.min_num_corners = min_num_corners

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        if self.filter_scheme == 'bev':
            # Check points instance type and initialise bev_range
            if isinstance(input_dict['gt_bboxes_3d'],
                          (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
                bev_range = self.pcd_range[[0, 1, 3, 4]]
            elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
                bev_range = self.pcd_range[[0, 2, 3, 5]]
            mask = gt_bboxes_3d.in_range_bev(bev_range)
        elif self.filter_scheme == 'corner':
            limit_range = torch.from_numpy(self.pcd_range).to(
                gt_bboxes_3d.tensor.device)
            corners = gt_bboxes_3d.corners
            mask = ((corners >= limit_range[0:3]) &
                    (corners <= limit_range[3:6])).all(dim=2)
            mask = mask.sum(dim=1) >= self.min_num_corners  # (N)
        else:
            raise NotImplementedError
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        gt_labels_3d = gt_labels_3d[mask.numpy().astype(np.bool)]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class TruncatedObjectFilter(object):
    """Filter truncated objects from the image views.

    Args:
        truncated_threshold (float): Truncated ratio threshold.
        area_ratio_threshold (float): Area ratio threshold.
        area_2d_ratio_threshold (float): 2D area ratio threshold.
    """

    def __init__(self,
                 truncated_threshold,
                 area_ratio_threshold=None,
                 area_2d_ratio_threshold=None):
        self.truncated_threshold = truncated_threshold
        self.area_ratio_threshold = area_ratio_threshold
        self.area_2d_ratio_threshold = area_2d_ratio_threshold
        if (self.area_ratio_threshold
                is not None) or (self.area_2d_ratio_threshold is not None):
            warnings.warn(
                'We do not support truncated object filter with ' +
                'area_ratio_threshold and area_2d_ratio_threshold yet.' +
                'These two factors take no effect right now.')

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        valid_mask = input_dict['ann_info'][
            'truncated'] < self.truncated_threshold
        for key in input_dict['ann_info'].keys():
            # the only ann does correspond to each instance
            if key == 'plane':
                continue
            input_dict['ann_info'][key] = input_dict['ann_info'][key][
                valid_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(truncated_threshold={self.truncated_threshold})'
        repr_str += f'(area_ratio_threshold={self.area_ratio_threshold})'
        repr_str += \
            f'(area_2d_ratio_threshold={self.area_2d_ratio_threshold})'
        return repr_str


@PIPELINES.register_module()
class IgnoredObjectFilter(object):
    """Filter ignored objects from the image views."""

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        valid_mask = input_dict['ann_info']['labels'] > -1
        for key in input_dict['ann_info'].keys():
            # the only ann does correspond to each instance
            if key == 'plane':
                continue
            input_dict['ann_info'][key] = input_dict['ann_info'][key][
                valid_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class PointsRangeFilter(object):
    """Filter points by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        points_mask = points.in_range_3d(self.pcd_range)
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class ObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                  dtype=np.bool_)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][gt_bboxes_mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@PIPELINES.register_module()
class PointSample(object):
    """Point sample.

    Sampling data to a certain number.

    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, num_points, sample_range=None, replace=False):
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace

    def _points_random_sampling(self,
                                points,
                                num_samples,
                                sample_range=None,
                                replace=False,
                                return_choices=False):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool, optional): Sampling with or without replacement.
                Defaults to None.
            return_choices (bool, optional): Whether return choice.
                Defaults to False.
        Returns:
            tuple[np.ndarray] | np.ndarray:
                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            dist = np.linalg.norm(points.tensor, axis=1)
            far_inds = np.where(dist >= sample_range)[0]
            near_inds = np.where(dist < sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(
                    far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.
        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        points, choices = self._points_random_sampling(
            points,
            self.num_points,
            self.sample_range,
            self.replace,
            return_choices=True)
        results['points'] = points

        pts_instance_mask = results.get('pts_instance_mask', None)
        pts_semantic_mask = results.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            pts_instance_mask = pts_instance_mask[choices]
            results['pts_instance_mask'] = pts_instance_mask

        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[choices]
            results['pts_semantic_mask'] = pts_semantic_mask

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' sample_range={self.sample_range},'
        repr_str += f' replace={self.replace})'

        return repr_str


@PIPELINES.register_module()
class IndoorPointSample(PointSample):
    """Indoor point sample.

    Sampling data to a certain number.
    NOTE: IndoorPointSample is deprecated in favor of PointSample

    Args:
        num_points (int): Number of points to be sampled.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'IndoorPointSample is deprecated in favor of PointSample')
        super(IndoorPointSample, self).__init__(*args, **kwargs)


@PIPELINES.register_module()
class IndoorPatchPointSample(object):
    r"""Indoor point sample within a patch. Modified from `PointNet++ <https://
    github.com/charlesq34/pointnet2/blob/master/scannet/scannet_dataset.py>`_.

    Sampling data to a certain number for semantic segmentation.

    Args:
        num_points (int): Number of points to be sampled.
        block_size (float, optional): Size of a block to sample points from.
            Defaults to 1.5.
        sample_rate (float, optional): Stride used in sliding patch generation.
            This parameter is unused in `IndoorPatchPointSample` and thus has
            been deprecated. We plan to remove it in the future.
            Defaults to None.
        ignore_index (int, optional): Label index that won't be used for the
            segmentation task. This is set in PointSegClassMapping as neg_cls.
            If not None, will be used as a patch selection criterion.
            Defaults to None.
        use_normalized_coord (bool, optional): Whether to use normalized xyz as
            additional features. Defaults to False.
        num_try (int, optional): Number of times to try if the patch selected
            is invalid. Defaults to 10.
        enlarge_size (float, optional): Enlarge the sampled patch to
            [-block_size / 2 - enlarge_size, block_size / 2 + enlarge_size] as
            an augmentation. If None, set it as 0. Defaults to 0.2.
        min_unique_num (int, optional): Minimum number of unique points
            the sampled patch should contain. If None, use PointNet++'s method
            to judge uniqueness. Defaults to None.
        eps (float, optional): A value added to patch boundary to guarantee
            points coverage. Defaults to 1e-2.

    Note:
        This transform should only be used in the training process of point
            cloud segmentation tasks. For the sliding patch generation and
            inference process in testing, please refer to the `slide_inference`
            function of `EncoderDecoder3D` class.
    """

    def __init__(self,
                 num_points,
                 block_size=1.5,
                 sample_rate=None,
                 ignore_index=None,
                 use_normalized_coord=False,
                 num_try=10,
                 enlarge_size=0.2,
                 min_unique_num=None,
                 eps=1e-2):
        self.num_points = num_points
        self.block_size = block_size
        self.ignore_index = ignore_index
        self.use_normalized_coord = use_normalized_coord
        self.num_try = num_try
        self.enlarge_size = enlarge_size if enlarge_size is not None else 0.0
        self.min_unique_num = min_unique_num
        self.eps = eps

        if sample_rate is not None:
            warnings.warn(
                "'sample_rate' has been deprecated and will be removed in "
                'the future. Please remove them from your code.')

    def _input_generation(self, coords, patch_center, coord_max, attributes,
                          attribute_dims, point_type):
        """Generating model input.

        Generate input by subtracting patch center and adding additional
            features. Currently support colors and normalized xyz as features.

        Args:
            coords (np.ndarray): Sampled 3D Points.
            patch_center (np.ndarray): Center coordinate of the selected patch.
            coord_max (np.ndarray): Max coordinate of all 3D Points.
            attributes (np.ndarray): features of input points.
            attribute_dims (dict): Dictionary to indicate the meaning of extra
                dimension.
            point_type (type): class of input points inherited from BasePoints.

        Returns:
            :obj:`BasePoints`: The generated input data.
        """
        # subtract patch center, the z dimension is not centered
        centered_coords = coords.copy()
        centered_coords[:, 0] -= patch_center[0]
        centered_coords[:, 1] -= patch_center[1]

        if self.use_normalized_coord:
            normalized_coord = coords / coord_max
            attributes = np.concatenate([attributes, normalized_coord], axis=1)
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(normalized_coord=[
                    attributes.shape[1], attributes.shape[1] +
                    1, attributes.shape[1] + 2
                ]))

        points = np.concatenate([centered_coords, attributes], axis=1)
        points = point_type(
            points, points_dim=points.shape[1], attribute_dims=attribute_dims)

        return points

    def _patch_points_sampling(self, points, sem_mask):
        """Patch points sampling.

        First sample a valid patch.
        Then sample points within that patch to a certain number.

        Args:
            points (:obj:`BasePoints`): 3D Points.
            sem_mask (np.ndarray): semantic segmentation mask for input points.

        Returns:
            tuple[:obj:`BasePoints`, np.ndarray] | :obj:`BasePoints`:

                - points (:obj:`BasePoints`): 3D Points.
                - choices (np.ndarray): The generated random samples.
        """
        coords = points.coord.numpy()
        attributes = points.tensor[:, 3:].numpy()
        attribute_dims = points.attribute_dims
        point_type = type(points)

        coord_max = np.amax(coords, axis=0)
        coord_min = np.amin(coords, axis=0)

        for _ in range(self.num_try):
            # random sample a point as patch center
            cur_center = coords[np.random.choice(coords.shape[0])]

            # boundary of a patch, which would be enlarged by
            # `self.enlarge_size` as an augmentation
            cur_max = cur_center + np.array(
                [self.block_size / 2.0, self.block_size / 2.0, 0.0])
            cur_min = cur_center - np.array(
                [self.block_size / 2.0, self.block_size / 2.0, 0.0])
            cur_max[2] = coord_max[2]
            cur_min[2] = coord_min[2]
            cur_choice = np.sum(
                (coords >= (cur_min - self.enlarge_size)) *
                (coords <= (cur_max + self.enlarge_size)),
                axis=1) == 3

            if not cur_choice.any():  # no points in this patch
                continue

            cur_coords = coords[cur_choice, :]
            cur_sem_mask = sem_mask[cur_choice]
            point_idxs = np.where(cur_choice)[0]
            mask = np.sum(
                (cur_coords >= (cur_min - self.eps)) * (cur_coords <=
                                                        (cur_max + self.eps)),
                axis=1) == 3

            # two criteria for patch sampling, adopted from PointNet++
            # 1. selected patch should contain enough unique points
            if self.min_unique_num is None:
                # use PointNet++'s method as default
                # [31, 31, 62] are just some big values used to transform
                # coords from 3d array to 1d and then check their uniqueness
                # this is used in all the ScanNet code following PointNet++
                vidx = np.ceil(
                    (cur_coords[mask, :] - cur_min) / (cur_max - cur_min) *
                    np.array([31.0, 31.0, 62.0]))
                vidx = np.unique(vidx[:, 0] * 31.0 * 62.0 + vidx[:, 1] * 62.0 +
                                 vidx[:, 2])
                flag1 = len(vidx) / 31.0 / 31.0 / 62.0 >= 0.02
            else:
                # if `min_unique_num` is provided, directly compare with it
                flag1 = mask.sum() >= self.min_unique_num

            # 2. selected patch should contain enough annotated points
            if self.ignore_index is None:
                flag2 = True
            else:
                flag2 = np.sum(cur_sem_mask != self.ignore_index) / \
                               len(cur_sem_mask) >= 0.7

            if flag1 and flag2:
                break

        # sample idx to `self.num_points`
        if point_idxs.size >= self.num_points:
            # no duplicate in sub-sampling
            choices = np.random.choice(
                point_idxs, self.num_points, replace=False)
        else:
            # do not use random choice here to avoid some points not counted
            dup = np.random.choice(point_idxs.size,
                                   self.num_points - point_idxs.size)
            idx_dup = np.concatenate(
                [np.arange(point_idxs.size),
                 np.array(dup)], 0)
            choices = point_idxs[idx_dup]

        # construct model input
        points = self._input_generation(coords[choices], cur_center, coord_max,
                                        attributes[choices], attribute_dims,
                                        point_type)

        return points, choices

    def __call__(self, results):
        """Call function to sample points to in indoor scenes.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']

        assert 'pts_semantic_mask' in results.keys(), \
            'semantic mask should be provided in training and evaluation'
        pts_semantic_mask = results['pts_semantic_mask']

        points, choices = self._patch_points_sampling(points,
                                                      pts_semantic_mask)

        results['points'] = points
        results['pts_semantic_mask'] = pts_semantic_mask[choices]
        pts_instance_mask = results.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            results['pts_instance_mask'] = pts_instance_mask[choices]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(num_points={self.num_points},'
        repr_str += f' block_size={self.block_size},'
        repr_str += f' ignore_index={self.ignore_index},'
        repr_str += f' use_normalized_coord={self.use_normalized_coord},'
        repr_str += f' num_try={self.num_try},'
        repr_str += f' enlarge_size={self.enlarge_size},'
        repr_str += f' min_unique_num={self.min_unique_num},'
        repr_str += f' eps={self.eps})'
        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilter(object):
    """Filter background points near the bounding box.

    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (is_tuple_of(bbox_enlarge_range, float)
                and len(bbox_enlarge_range) == 3) \
            or isinstance(bbox_enlarge_range, float), \
            f'Invalid arguments bbox_enlarge_range {bbox_enlarge_range}'

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(
            bbox_enlarge_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = input_dict['points']
        gt_bboxes_3d = input_dict['gt_bboxes_3d']

        # avoid groundtruth being modified
        gt_bboxes_3d_np = gt_bboxes_3d.tensor.clone().numpy()
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.clone().numpy()

        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        points_numpy = points.tensor.clone().numpy()
        foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, gt_bboxes_3d_np, origin=(0.5, 0.5, 0.5))
        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d, origin=(0.5, 0.5, 0.5))
        foreground_masks = foreground_masks.max(1)
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        valid_masks = ~np.logical_and(~foreground_masks,
                                      enlarge_foreground_masks)

        input_dict['points'] = points[valid_masks]
        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[valid_masks]

        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[valid_masks]
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(bbox_enlarge_range={self.bbox_enlarge_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class VoxelBasedPointSampler(object):
    """Voxel based point sampler.

    Apply voxel sampling to multiple sweep points.

    Args:
        cur_sweep_cfg (dict): Config for sampling current points.
        prev_sweep_cfg (dict): Config for sampling previous points.
        time_dim (int): Index that indicate the time dimension
            for input points.
    """

    def __init__(self, cur_sweep_cfg, prev_sweep_cfg=None, time_dim=3):
        self.cur_voxel_generator = VoxelGenerator(**cur_sweep_cfg)
        self.cur_voxel_num = self.cur_voxel_generator._max_voxels
        self.time_dim = time_dim
        if prev_sweep_cfg is not None:
            assert prev_sweep_cfg['max_num_points'] == \
                cur_sweep_cfg['max_num_points']
            self.prev_voxel_generator = VoxelGenerator(**prev_sweep_cfg)
            self.prev_voxel_num = self.prev_voxel_generator._max_voxels
        else:
            self.prev_voxel_generator = None
            self.prev_voxel_num = 0

    def _sample_points(self, points, sampler, point_dim):
        """Sample points for each points subset.

        Args:
            points (np.ndarray): Points subset to be sampled.
            sampler (VoxelGenerator): Voxel based sampler for
                each points subset.
            point_dim (int): The dimension of each points

        Returns:
            np.ndarray: Sampled points.
        """
        voxels, coors, num_points_per_voxel = sampler.generate(points)
        if voxels.shape[0] < sampler._max_voxels:
            padding_points = np.zeros([
                sampler._max_voxels - voxels.shape[0], sampler._max_num_points,
                point_dim
            ],
                                      dtype=points.dtype)
            padding_points[:] = voxels[0]
            sample_points = np.concatenate([voxels, padding_points], axis=0)
        else:
            sample_points = voxels

        return sample_points

    def __call__(self, results):
        """Call function to sample points from multiple sweeps.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after sampling, 'points', 'pts_instance_mask'
                and 'pts_semantic_mask' keys are updated in the result dict.
        """
        points = results['points']
        original_dim = points.shape[1]

        # TODO: process instance and semantic mask while _max_num_points
        # is larger than 1
        # Extend points with seg and mask fields
        map_fields2dim = []
        start_dim = original_dim
        points_numpy = points.tensor.numpy()
        extra_channel = [points_numpy]
        for idx, key in enumerate(results['pts_mask_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        start_dim += len(results['pts_mask_fields'])
        for idx, key in enumerate(results['pts_seg_fields']):
            map_fields2dim.append((key, idx + start_dim))
            extra_channel.append(results[key][..., None])

        points_numpy = np.concatenate(extra_channel, axis=-1)

        # Split points into two part, current sweep points and
        # previous sweeps points.
        # TODO: support different sampling methods for next sweeps points
        # and previous sweeps points.
        cur_points_flag = (points_numpy[:, self.time_dim] == 0)
        cur_sweep_points = points_numpy[cur_points_flag]
        prev_sweeps_points = points_numpy[~cur_points_flag]
        if prev_sweeps_points.shape[0] == 0:
            prev_sweeps_points = cur_sweep_points

        # Shuffle points before sampling
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(prev_sweeps_points)

        cur_sweep_points = self._sample_points(cur_sweep_points,
                                               self.cur_voxel_generator,
                                               points_numpy.shape[1])
        if self.prev_voxel_generator is not None:
            prev_sweeps_points = self._sample_points(prev_sweeps_points,
                                                     self.prev_voxel_generator,
                                                     points_numpy.shape[1])

            points_numpy = np.concatenate(
                [cur_sweep_points, prev_sweeps_points], 0)
        else:
            points_numpy = cur_sweep_points

        if self.cur_voxel_generator._max_num_points == 1:
            points_numpy = points_numpy.squeeze(1)
        results['points'] = points.new_point(points_numpy[..., :original_dim])

        # Restore the corresponding seg and mask fields
        for key, dim_index in map_fields2dim:
            results[key] = points_numpy[..., dim_index]

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""

        def _auto_indent(repr_str, indent):
            repr_str = repr_str.split('\n')
            repr_str = [' ' * indent + t + '\n' for t in repr_str]
            repr_str = ''.join(repr_str)[:-1]
            return repr_str

        repr_str = self.__class__.__name__
        indent = 4
        repr_str += '(\n'
        repr_str += ' ' * indent + f'num_cur_sweep={self.cur_voxel_num},\n'
        repr_str += ' ' * indent + f'num_prev_sweep={self.prev_voxel_num},\n'
        repr_str += ' ' * indent + f'time_dim={self.time_dim},\n'
        repr_str += ' ' * indent + 'cur_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.cur_voxel_generator), 8)},\n'
        repr_str += ' ' * indent + 'prev_voxel_generator=\n'
        repr_str += f'{_auto_indent(repr(self.prev_voxel_generator), 8)})'
        return repr_str


@PIPELINES.register_module()
class AffineResize(object):
    """Get the affine transform matrices to the target size.

    Different from :class:`RandomAffine` in MMDetection, this class can
    calculate the affine transform matrices while resizing the input image
    to a fixed size. The affine transform matrices include: 1) matrix
    transforming original image to the network input image size. 2) matrix
    transforming original image to the network output feature map size.

    Args:
        img_scale (tuple): Images scales for resizing.
        down_ratio (int): The down ratio of feature map.
            Actually the arg should be >= 1.
        bbox_clip_border (bool, optional): Whether clip the objects
            outside the border of the image. Defaults to True.
    """

    def __init__(self, img_scale, down_ratio, bbox_clip_border=True):

        self.img_scale = img_scale
        self.down_ratio = down_ratio
        self.bbox_clip_border = bbox_clip_border

    def __call__(self, results):
        """Call function to do affine transform to input image and labels.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after affine resize, 'affine_aug', 'trans_mat'
                keys are added in the result dict.
        """
        # The results have gone through RandomShiftScale before AffineResize
        if 'center' not in results:
            img = results['img']
            height, width = img.shape[:2]
            center = np.array([width / 2, height / 2], dtype=np.float32)
            size = np.array([width, height], dtype=np.float32)
            results['affine_aug'] = False
        else:
            # The results did not go through RandomShiftScale before
            # AffineResize
            img = results['img']
            center = results['center']
            size = results['size']

        trans_affine = self._get_transform_matrix(center, size, self.img_scale)

        img = cv2.warpAffine(img, trans_affine[:2, :], self.img_scale)

        if isinstance(self.down_ratio, tuple):
            trans_mat = [
                self._get_transform_matrix(
                    center, size,
                    (self.img_scale[0] // ratio, self.img_scale[1] // ratio))
                for ratio in self.down_ratio
            ]  # (3, 3)
        else:
            trans_mat = self._get_transform_matrix(
                center, size, (self.img_scale[0] // self.down_ratio,
                               self.img_scale[1] // self.down_ratio))

        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['trans_mat'] = trans_mat

        self._affine_bboxes(results, trans_affine)

        if 'centers2d' in results:
            centers2d = self._affine_transform(results['centers2d'],
                                               trans_affine)
            valid_index = (centers2d[:, 0] >
                           0) & (centers2d[:, 0] <
                                 self.img_scale[0]) & (centers2d[:, 1] > 0) & (
                                     centers2d[:, 1] < self.img_scale[1])
            results['centers2d'] = centers2d[valid_index]

            for key in results.get('bbox_fields', []):
                if key in ['gt_bboxes']:
                    results[key] = results[key][valid_index]
                    if 'gt_labels' in results:
                        results['gt_labels'] = results['gt_labels'][
                            valid_index]
                    if 'gt_masks' in results:
                        raise NotImplementedError(
                            'AffineResize only supports bbox.')

            for key in results.get('bbox3d_fields', []):
                if key in ['gt_bboxes_3d']:
                    results[key].tensor = results[key].tensor[valid_index]
                    if 'gt_labels_3d' in results:
                        results['gt_labels_3d'] = results['gt_labels_3d'][
                            valid_index]

            results['depths'] = results['depths'][valid_index]

        return results

    def _affine_bboxes(self, results, matrix):
        """Affine transform bboxes to input image.

        Args:
            results (dict): Result dict from loading pipeline.
            matrix (np.ndarray): Matrix transforming original
                image to the network input image size.
                shape: (3, 3)
        """

        for key in results.get('bbox_fields', []):
            bboxes = results[key]
            bboxes[:, :2] = self._affine_transform(bboxes[:, :2], matrix)
            bboxes[:, 2:] = self._affine_transform(bboxes[:, 2:], matrix)
            if self.bbox_clip_border:
                bboxes[:,
                       [0, 2]] = bboxes[:,
                                        [0, 2]].clip(0, self.img_scale[0] - 1)
                bboxes[:,
                       [1, 3]] = bboxes[:,
                                        [1, 3]].clip(0, self.img_scale[1] - 1)
            results[key] = bboxes

    def _affine_transform(self, points, matrix):
        """Affine transform bbox points to input image.

        Args:
            points (np.ndarray): Points to be transformed.
                shape: (N, 2)
            matrix (np.ndarray): Affine transform matrix.
                shape: (3, 3)

        Returns:
            np.ndarray: Transformed points.
        """
        num_points = points.shape[0]
        hom_points_2d = np.concatenate((points, np.ones((num_points, 1))),
                                       axis=1)
        hom_points_2d = hom_points_2d.T
        affined_points = np.matmul(matrix, hom_points_2d).T
        return affined_points[:, :2]

    def _get_transform_matrix(self, center, scale, output_scale):
        """Get affine transform matrix.

        Args:
            center (tuple): Center of current image.
            scale (tuple): Scale of current image.
            output_scale (tuple[float]): The transform target image scales.

        Returns:
            np.ndarray: Affine transform matrix.
        """
        # TODO: further add rot and shift here.
        src_w = scale[0]
        dst_w = output_scale[0]
        dst_h = output_scale[1]

        src_dir = np.array([0, src_w * -0.5])
        dst_dir = np.array([0, dst_w * -0.5])

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2, :] = self._get_ref_point(src[0, :], src[1, :])
        dst[2, :] = self._get_ref_point(dst[0, :], dst[1, :])

        get_matrix = cv2.getAffineTransform(src, dst)

        matrix = np.concatenate((get_matrix, [[0., 0., 1.]]))

        return matrix.astype(np.float32)

    def _get_ref_point(self, ref_point1, ref_point2):
        """Get reference point to calculate affine transform matrix.

        While using opencv to calculate the affine matrix, we need at least
        three corresponding points separately on original image and target
        image. Here we use two points to get the the third reference point.
        """
        d = ref_point1 - ref_point2
        ref_point3 = ref_point2 + np.array([-d[1], d[0]])
        return ref_point3

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'down_ratio={self.down_ratio}) '
        return repr_str


@PIPELINES.register_module()
class RandomShiftScale(object):
    """Random shift scale.

    Different from the normal shift and scale function, it doesn't
    directly shift or scale image. It can record the shift and scale
    infos into loading pipelines. It's designed to be used with
    AffineResize together.

    Args:
        shift_scale (tuple[float]): Shift and scale range.
        aug_prob (float): The shifting and scaling probability.
    """

    def __init__(self, shift_scale, aug_prob):

        self.shift_scale = shift_scale
        self.aug_prob = aug_prob

    def __call__(self, results):
        """Call function to record random shift and scale infos.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after random shift and scale, 'center', 'size'
                and 'affine_aug' keys are added in the result dict.
        """
        img = results['img']

        height, width = img.shape[:2]

        center = np.array([width / 2, height / 2], dtype=np.float32)
        size = np.array([width, height], dtype=np.float32)

        if random.random() < self.aug_prob:
            shift, scale = self.shift_scale[0], self.shift_scale[1]
            shift_ranges = np.arange(-shift, shift + 0.1, 0.1)
            center[0] += size[0] * random.choice(shift_ranges)
            center[1] += size[1] * random.choice(shift_ranges)
            scale_ranges = np.arange(1 - scale, 1 + scale + 0.1, 0.1)
            size *= random.choice(scale_ranges)
            results['affine_aug'] = True
        else:
            results['affine_aug'] = False

        results['center'] = center
        results['size'] = size

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(shift_scale={self.shift_scale}, '
        repr_str += f'aug_prob={self.aug_prob}) '
        return repr_str


@PIPELINES.register_module()
class MultiViewImagePad(object):
    """Pad the multi-view image.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""

        # should be the max shape of multi-view images
        results['img_shape'] = [img.shape for img in results['img']]

        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results['img']
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(
                    img, self.size_divisor, pad_val=self.pad_val)
                for img in results['img']
            ]
        results['img'] = padded_img

        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg_mask(self, results):
        # TODO: set a custom value for seg_mask padding
        # temporarily use pad_val=self.pad_val
        for key in results.get('seg_fields', []):
            padded_mask = [
                mmcv.impad(
                    mask,
                    shape=results['pad_shape'][0][:2],
                    pad_val=self.pad_val) for mask in results[key]
            ]
            results[key] = padded_mask

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg_mask(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class MultiViewImageNormalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results['img']
        ]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += \
            f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class MultiViewImagePhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, \
                'PhotoMetricDistortion needs the input image of dtype '\
                "np.float32, please set 'to_float32=True' in "\
                "'LoadImageFromFile' pipeline"
            # random brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta,
                                          self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(self.saturation_lower,
                                                 self.saturation_upper)

            # random hue
            if np.random.randint(2):
                img[..., 0] += np.random.uniform(-self.hue_delta,
                                                 self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                              self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if np.random.randint(2):
                img = img[..., np.random.permutation(3)]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str


@PIPELINES.register_module()
class Resize3D(Resize):
    """Resize 3D labels.

    Different from 2D Resize, we accept img_scale=None and ratio_range is not
    None. In that case we will take the input img scale as the ori_scale for
    rescaling with ratio_range.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 interpolation='nearest',
                 override=False,
                 cam2img_keep_ratio=False):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is None:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.override = override
        self.bbox_clip_border = bbox_clip_border
        self.cam2img_keep_ratio = cam2img_keep_ratio

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``ori_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """
        # ori_scale = results['img'].shape[:2]
        # consider the ori_scale can be specified by self.img_scale
        if self.img_scale is not None:
            ori_scale = self.img_scale[0]
        else:
            ori_scale = results['img'].shape[:2]
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                ori_scale, self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_3d(self, results):
        """Resize centers2d and modify camera intrinisc with
        ``results['scale']``."""
        if 'centers2d' in results:
            results['centers2d'] *= results['scale_factor'][:2]
        # resize image equals to change focal length and
        # camera intrinsic
        results['cam2img'][0] *= results['scale_factor'][0].repeat(
            len(results['cam2img'][0]))
        if self.cam2img_keep_ratio:
            results['cam2img'][1] *= results['scale_factor'][0].repeat(
                len(results['cam2img'][1]))
        else:
            results['cam2img'][1] *= results['scale_factor'][1].repeat(
                len(results['cam2img'][1]))
        if 'calib' in results:
            assert self.keep_ratio, 'calib.scale only support ' \
                'keep_ratio=True for now.'
            results['calib'].scale(results['scale_factor'][0])

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        super(Resize3D, self).__call__(results)
        self._resize_3d(results)

        return results


@PIPELINES.register_module()
class MultiViewImageResize3D(Resize3D):
    """Random scale the image with modifying camera intrinsic. This function is
    still unstable, required further testing.

    Args:
        img_scale: resolution for the output image
        keep_ratio: keep the ratio between w and h
        resize_depth: consider simltaneously resize depth (for training aug)
        ratio_range:
        override: do the resize more than one time.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 override=False):
        super(MultiViewImageResize3D, self).__init__(
            img_scale=img_scale,
            multiscale_mode=multiscale_mode,
            ratio_range=ratio_range,
            keep_ratio=keep_ratio,
            bbox_clip_border=bbox_clip_border,
            backend=backend,
            override=override)
        # temporarily do not need these params
        # because resize_depth=False & resize_intrinsic=True
        # is an optimal setting surveyed before
        # self.resize_depth = resize_depth
        # self.rescale_intrinsic = rescale_intrinsic

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``ori_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into \
                ``results``, which would be used by subsequent pipelines.
        """
        # NOTE: we only support a single scale for multi-view for now
        # TODO: support different scales for multi-view images
        # ori_scale = results['img'].shape[:2]
        # consider the ori_scale can be specified by self.img_scale
        if self.img_scale is not None:
            ori_scale = self.img_scale[0]
        else:
            ori_scale = results['img'][0].shape[:2]
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                ori_scale, self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_3d(self, results):
        """Resize centers2d and modify camera intrinisc with
        ``results['scale']``."""
        if 'centers2d' in results:
            results['centers2d'] *= results['scale_factor'][:2]
        # resize image equals to change focal length and
        # camera intrinsic
        if 'cam2img' in results:
            for idx, cam2img in enumerate(results['cam2img']):
                cam2img[0] *= results['scale_factor'][0].repeat(
                    len(cam2img[0]))
                cam2img[1] *= results['scale_factor'][1].repeat(
                    len(cam2img[0]))
                results['cam2img'][idx] = cam2img
                results['lidar2img'][idx] = cam2img @ results['lidar2cam'][idx]

    def _resize_img(self, results):
        """Resize images with list of inputs ``results['scale']``."""

        # other with type of tensor
        for idx, img in enumerate(results['img']):
            if self.keep_ratio:
                resized_img, scale_factor = mmcv.imrescale(
                    img,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = resized_img.shape[:2]
                h, w = img.shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                resized_img, w_scale, h_scale = mmcv.imresize(
                    img,
                    results['scale'],
                    return_scale=True,
                    backend=self.backend)
            results['img'][idx] = resized_img

            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)

        results['img_shape'] = results['img'][0].shape
        # in case that there is no padding
        results['pad_shape'] = results['img'][0].shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            for idx, seg in enumerate(results[key]):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        seg,
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        seg,
                        results['scale'],
                        interpolation='nearest',
                        backend=self.backend)
                results[key][idx] = gt_seg

    def __call__(self, results):
        # Assume the multiview image is in the same shape
        # currently img is with the shape of [N, C, H, W, M]

        # NOTE: remove the check scale_factor part due to multi-view imgs
        if 'scale' not in results:
            if 'scale_factor' in results:
                self._random_scale(results)
        else:
            if not self.override:
                assert 'scale_factor' not in results, (
                    'scale and scale_factor cannot be both set.')
            else:
                results.pop('scale')
                if 'scale_factor' in results:
                    results.pop('scale_factor')
                self._random_scale(results)

        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._resize_3d(results)
        return results


@PIPELINES.register_module()
class RandomCrop3D(RandomCrop):
    """3D version of RandomCrop.

    RamdomCrop3D needs some customized settings and modifications for camera
    intrinsic matrix.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 recompute_bbox=False,
                 bbox_clip_border=True,
                 rel_offset_h=(0., 1.),
                 rel_offset_w=(0., 1.)):
        super().__init__(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border)
        # rel_offset specifies the relative offset range of cropping origin
        # [0., 1.] means starting from 0*margin to 1*margin + 1
        self.rel_offset_h = rel_offset_h
        self.rel_offset_w = rel_offset_w

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            img = results[key]
            margin_h = max(img.shape[0] - crop_size[0], 0)
            margin_w = max(img.shape[1] - crop_size[1], 0)
            # TOCHECK: a little different from LIGA implementation
            offset_h = np.random.randint(self.rel_offset_h[0] * margin_h,
                                         self.rel_offset_h[1] * margin_h + 1)
            offset_w = np.random.randint(self.rel_offset_w[0] * margin_w,
                                         self.rel_offset_w[1] * margin_w + 1)
            crop_h = min(crop_size[0], img.shape[0])
            crop_w = min(crop_size[1], img.shape[1])
            crop_y1, crop_y2 = offset_h, offset_h + crop_h
            crop_x1, crop_x2 = offset_w, offset_w + crop_w

            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape
            results[key] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # manipulate camera intrinsic matrix
        # needs to apply offset to K instead of P2 (on KITTI)
        K = results['cam2img'][:3, :3].copy()
        inv_K = np.linalg.inv(K)
        T = np.matmul(inv_K, results['cam2img'][:3])
        K[0, 2] -= crop_x1
        K[1, 2] -= crop_y1
        offset_cam2img = np.matmul(K, T)
        results['cam2img'][:offset_cam2img.shape[0], :offset_cam2img.
                           shape[1]] = offset_cam2img
        results['crop_offset'] = [crop_x1, crop_y1]
        if 'calib' in results:
            results['calib'].offset(crop_x1, crop_y1)

        return results

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}), '
        repr_str += f'rel_offset_h={self.rel_offset_h}), '
        repr_str += f'rel_offset_w={self.rel_offset_w})'
        return repr_str


@PIPELINES.register_module()
class MultiViewImageCrop3D(RandomCrop3D):
    """Random crop the image with recording the crop index.

    This function is still unstable, required further testing.
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 allow_negative_crop=False,
                 recompute_bbox=False,
                 bbox_clip_border=True,
                 rel_offset_h=(0., 1.),
                 rel_offset_w=(0., 1.)):
        super().__init__(
            crop_size=crop_size,
            crop_type=crop_type,
            allow_negative_crop=allow_negative_crop,
            recompute_bbox=recompute_bbox,
            bbox_clip_border=bbox_clip_border)
        # rel_offset specifies the relative offset range of cropping origin
        # [0., 1.] means starting from 0*margin to 1*margin + 1
        self.rel_offset_h = rel_offset_h
        self.rel_offset_w = rel_offset_w

    def _crop_data(self, results, crop_size, allow_negative_crop):
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Currently I did not modify the cam intrinsic because ImVoxelNet use
        the origin cam intrinsic; In the future,  if modify intrinsic, one
        may consider the conflict with flip augmentation.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (tuple): Expected absolute size after cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area. Default to False.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        for idx, img in enumerate(results['img']):
            if idx == 0:
                margin_h = max(img.shape[0] - crop_size[0], 0)
                margin_w = max(img.shape[1] - crop_size[1], 0)
                # TOCHECK: a little different from LIGA implementation
                offset_h = np.random.randint(
                    self.rel_offset_h[0] * margin_h,
                    self.rel_offset_h[1] * margin_h + 1)
                offset_w = np.random.randint(
                    self.rel_offset_w[0] * margin_w,
                    self.rel_offset_w[1] * margin_w + 1)
                crop_h = min(crop_size[0], img.shape[0])
                crop_w = min(crop_size[1], img.shape[1])
                crop_y1, crop_y2 = offset_h, offset_h + crop_h
                crop_x1, crop_x2 = offset_w, offset_w + crop_w
            # crop the image
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
            img_shape = img.shape

            results['img'][idx] = img
        results['img_shape'] = img_shape

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            # e.g. gt_bboxes and gt_bboxes_ignore
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            if self.bbox_clip_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1])
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (key == 'gt_bboxes' and not valid_inds.any()
                    and not allow_negative_crop):
                return None
            results[key] = bboxes[valid_inds, :]
            # label fields. e.g. gt_labels and gt_labels_ignore
            label_key = self.bbox2label.get(key)
            if label_key in results:
                results[label_key] = results[label_key][valid_inds]

            # mask fields, e.g. gt_masks and gt_masks_ignore
            mask_key = self.bbox2mask.get(key)
            if mask_key in results:
                results[mask_key] = results[mask_key][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results[key] = results[mask_key].get_bboxes()

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # manipulate camera intrinsic matrix
        # needs to apply offset to K instead of P2 (on KITTI)
        # K = results['cam2img'][:3, :3].copy()
        # inv_K = np.linalg.inv(K)
        # T = np.matmul(inv_K, results['cam2img'][:3])
        # K[0, 2] -= crop_x1
        # K[1, 2] -= crop_y1
        # offset_cam2img = np.matmul(K, T)
        # results['cam2img'][:offset_cam2img.shape[0], :offset_cam2img.
        #    shape[1]] = offset_cam2img
        results['img_crop_offset'] = [crop_x1, crop_y1]

        return results

    def __call__(self, results):
        """Call function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        if isinstance(results['img'], list):
            image_size = results['img'][0].shape[:2]
        else:
            image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border}), '
        repr_str += f'rel_offset_h={self.rel_offset_h}), '
        repr_str += f'rel_offset_w={self.rel_offset_w})'
        return repr_str


@PIPELINES.register_module()
class MultiViewRandomFlip3D(RandomFlip):
    """Multi-View version for RandomFlip3D.

    Current only flip the image and add the flip flag; do not convert the 3D
    bounding boxes and intrinsic;
    """

    def __init__(self, sync_2d=True, **kwargs):
        super(MultiViewRandomFlip3D, self).__init__(**kwargs)
        self.sync_2d = sync_2d

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added \
                into result dict.
        """

        if 'flip' not in results:
            if isinstance(self.direction, list):
                # None means non-flip
                direction_list = self.direction + [None]
            else:
                # None means non-flip
                direction_list = [self.direction, None]

            if isinstance(self.flip_ratio, list):
                non_flip_ratio = 1 - sum(self.flip_ratio)
                flip_ratio_list = self.flip_ratio + [non_flip_ratio]
            else:
                non_flip_ratio = 1 - self.flip_ratio
                # exclude non-flip
                single_ratio = self.flip_ratio / (len(direction_list) - 1)
                flip_ratio_list = [single_ratio] * (len(direction_list) -
                                                    1) + [non_flip_ratio]
            cur_dir = np.random.choice(direction_list, p=flip_ratio_list)
            results['flip'] = cur_dir is not None

        if 'flip_direction' not in results:
            results['flip_direction'] = cur_dir

        if results['flip']:
            # flip image
            results['img'] = [
                mmcv.imflip(img, results['flip_direction'])
                for img in results['img']
            ]
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = results[key].flip(results['flip_direction'])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])
            self.random_flip_data_3d(
                results, direction=results['flip_direction'])

        return results

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str, optional): Flip direction.
                Default: 'horizontal'.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']
        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['img_shape'][1]
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]
