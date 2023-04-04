from __future__ import print_function, division
import os
import numpy as np
import json
from functools import partial
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader # noqa: F401
from easydict import EasyDict
import torch
import torch.utils.data
from monodepth.data.datasets.utils import read_image, cam_relative_pose_nusc, get_transformation_matrix, read_vo_depth
from vision_base.utils.builder import build

class NusceneDepthMonoDataset(torch.utils.data.Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(NusceneDepthMonoDataset, self).__init__()
        self.nuscenes_version = getattr(data_cfg, 'nuscenes_version', 'v1.0-trainval')
        self.nuscenes_dir     = getattr(data_cfg, 'nuscenes_dir', '/data/nuscene')
        self.nusc_meta_file   = data_cfg.split_file

        
        with open(self.nusc_meta_file, 'r') as f:
            self.token_list = [line.strip().split(',') for line in f.readlines()]

        self.nusc = build('vision_base.data.datasets.nuscenes_utils.NuScenes', version=self.nuscenes_version, dataroot=self.nuscenes_dir, verbose=True)
        self.number_scenes = len(self.nusc.scene)
        print(f"Found {self.number_scenes} in the {self.nuscenes_version}")

        self.nusc_get_sample = partial(self.nusc.get, 'sample')
        self.nusc_get_sample_data = partial(self.nusc.get, 'sample_data')
        self.nusc_get_sensor     = partial(self.nusc.get, 'calibrated_sensor')
        self.nusc_get_ego_pose   = partial(self.nusc.get, 'ego_pose')

        self.cameras = getattr(data_cfg, 'channels', ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])
        self.vo_path = getattr(data_cfg, 'vo_path', None)
        self.is_read_vo_depth = self.vo_path is not None
        self.frame_ids = getattr(data_cfg, 'frame_ids', [0, -1, 1])

        self.is_motion_mask  = getattr(data_cfg, 'is_motion_mask', False)
        if self.is_motion_mask:
            self.precompute_path = data_cfg.precompute_path

        self.is_filter_static = getattr(data_cfg, 'is_filter_static', True)
        self.filter_threshold = getattr(data_cfg, 'filter_threshold', 0.03)

        self.transform = build(**data_cfg.augmentation)

    def __len__(self):
        return len(self.token_list) * len(self.cameras)
    
    def get_intrinsic(self, cs_record):
        return np.array(cs_record['camera_intrinsic']) #[3, 3]
    
    def get_extrinsic(self, cs_record):
        return get_transformation_matrix(cs_record['translation'], cs_record['rotation'])
    
    def get_ego_pose(self, ego_record):
        return get_transformation_matrix(ego_record['translation'], ego_record['rotation'])
    
    def __getitem__(self, index):
        token_index       = index // len(self.cameras)
        camera_type_index = index % len(self.cameras)
        camera_type       = self.cameras[camera_type_index]
        
        sample_tokens = self.token_list[token_index]
        samples        = list(map(self.nusc_get_sample, sample_tokens))
        camera_datas   = list(map(self.nusc_get_sample_data, [sample['data'][camera_type] for sample in samples]))
        cs_records     = list(map(self.nusc_get_sensor, [camera_data['calibrated_sensor_token'] for camera_data in camera_datas]))
        ego_records    = list(map(self.nusc_get_ego_pose, [camera_data['ego_pose_token'] for camera_data in camera_datas]))
        
        image_arrays   = list(map(
            read_image, [os.path.join(self.nuscenes_dir, camera_data['filename']) for camera_data in camera_datas]
        ))
        P2 = self.get_intrinsic(cs_records[0])
        extrinsics = list(map(self.get_extrinsic, cs_records)) #[T] 4 x 4 x 3
        poses      = list(map(self.get_ego_pose, ego_records)) #[T] 4 x 4 x 3
        
        data = dict()
        for i, idx in enumerate(self.frame_ids[1:]):
            data[('relative_pose', idx)] = cam_relative_pose_nusc(
               poses[0], poses[i+1], np.linalg.inv(extrinsics[0])
            ).astype(np.float32)
            if self.is_filter_static:
                translation = np.linalg.norm(data[('relative_pose'), idx][0:3, 3])
                if translation < self.filter_threshold or translation > 3:
                    return self[np.random.randint(len(self))]
        
        for i, frame_id in enumerate(self.frame_ids):
            data[('image', frame_id)] = image_arrays[i]
            data[('original_image', frame_id)] = data[('image', frame_id)].copy()

        if self.is_read_vo_depth:
            vo_path = camera_datas[0]['filename'].replace('samples', self.vo_path).replace('.jpg', '.png')
            if os.path.isfile(vo_path):
                vo_depth = read_vo_depth(
                    camera_datas[0]['filename'].replace('samples', self.vo_path).replace('.jpg', '.png')
                )
                data[('vo_depth', 0)] = vo_depth
            else:
                print(f'No VO Depth file found at {index}, {vo_path}')

        h, w, _ = data[("image", 0)].shape
        data["patched_mask"] = np.ones([h, w])
        
        data['P2'] = np.zeros((3, 4), dtype=np.float32)
        data['P2'][0:3, 0:3] = P2
        data['original_P2'] = data['P2'].copy()
        data['camera_type_index'] = camera_type_index
        data[('filename', 0)] = camera_datas[0]['filename']
        data['camera_type'] = camera_type

        data = self.transform(deepcopy(data))
        return data

class NusceneSweepDepthMonoDataset(NusceneDepthMonoDataset):
    """Use Sweep around key frames"""
    def __getitem__(self, index):
        token_index       = index // len(self.cameras)
        camera_type_index = index % len(self.cameras)
        camera_type       = self.cameras[camera_type_index]
        
        sample_tokens = self.token_list[token_index]
        main_token     = sample_tokens[0] # center sample data
        main_sample    = self.nusc_get_sample(main_token)
        main_camera_instance = self.nusc_get_sample_data(main_sample['data'][camera_type])
        camera_datas = [main_camera_instance]

        for frame_id in self.frame_ids[1:]:
            next_key = 'next' if frame_id > 0 else 'prev'
            tmp_camera_instance = main_camera_instance
            for _ in range(abs(frame_id)):
                tmp_camera_instance = self.nusc_get_sample_data(tmp_camera_instance[next_key])
            camera_datas.append(tmp_camera_instance)

        cs_records     = list(map(self.nusc_get_sensor, [camera_data['calibrated_sensor_token'] for camera_data in camera_datas]))
        ego_records    = list(map(self.nusc_get_ego_pose, [camera_data['ego_pose_token'] for camera_data in camera_datas]))
        
        image_arrays   = list(map(
            read_image, [os.path.join(self.nuscenes_dir, camera_data['filename']) for camera_data in camera_datas]
        ))
        P2 = self.get_intrinsic(cs_records[0])
        extrinsics = list(map(self.get_extrinsic, cs_records)) #[T] 4 x 4 x 3
        poses      = list(map(self.get_ego_pose, ego_records)) #[T] 4 x 4 x 3
        
        data = dict()
        for i, idx in enumerate(self.frame_ids[1:]):
            data[('relative_pose', idx)] = cam_relative_pose_nusc(
               poses[0], poses[i+1], np.linalg.inv(extrinsics[0])
            ).astype(np.float32)
            if self.is_filter_static:
                translation = np.linalg.norm(data[('relative_pose'), idx][0:3, 3])
                if translation < self.filter_threshold or translation > 3:
                    return self[np.random.randint(len(self))]
        
        for i, frame_id in enumerate(self.frame_ids):
            data[('image', frame_id)] = image_arrays[i]
            data[('original_image', frame_id)] = data[('image', frame_id)].copy()

        h, w, _ = data[("image", 0)].shape
        data["patched_mask"] = np.ones([h, w])
        
        data['P2'] = np.zeros((3, 4), dtype=np.float32)
        data['P2'][0:3, 0:3] = P2
        data['original_P2'] = data['P2'].copy()
        data['camera_type_index'] = camera_type_index

        data = self.transform(deepcopy(data))
        return data

class NusceneJsonDataset(torch.utils.data.Dataset):
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(NusceneJsonDataset, self).__init__()
        self.json_path = getattr(data_cfg, 'json_path',
                '/home/monodepth/meta_data/nusc_trainsub/json_nusc_front_train.json')

        self.json_dict = json.load(
            open(self.json_path, 'r')
        )

        self.image_keys = getattr(data_cfg, 'image_keys', ['frame0', 'frame1', 'frame-1'])
        self.pose_keys = getattr(data_cfg, 'pose_keys' , ['pose01', 'pose0-1'])
        self.intrinsic_key = getattr(data_cfg, 'intrinsic_key', 'P2')

        self.cameras = getattr(data_cfg, 'channels',
            ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'])
        self.frame_ids = getattr(data_cfg, 'frame_ids', [0, 1, -1])

        self.transform = build(**data_cfg.augmentation)

        self.vo_path = getattr(data_cfg, 'vo_path', None)
        self.is_read_vo_depth = self.vo_path is not None

    def __len__(self):
        return len(self.json_dict['samples'])
    
    def __getitem__(self, index):
        sample = self.json_dict['samples'][index]
        image_arrays   = list(map(
            read_image, [sample[key] for key in self.image_keys]
        ))
        P2 = np.array(sample[self.intrinsic_key]).reshape(3, 3).astype(np.float32)
        camera_type_index = sample['camera_type_indexes']
        camera_type = sample['camera_type']

        data = dict()
        data[('relative_pose', 1)] = np.array(sample['pose01']).reshape([4, 4]).astype(np.float32)
        data[('relative_pose', -1)] = np.array(sample['pose0-1']).reshape([4, 4]).astype(np.float32)
        
        for i, frame_id in enumerate(self.frame_ids):
            data[('image', frame_id)] = image_arrays[i]
            data[('original_image', frame_id)] = data[('image', frame_id)].copy()

        h, w, _ = data[("image", 0)].shape
        data["patched_mask"] = np.ones([h, w])
        if camera_type == 'CAM_BACK':
            data['patched_mask'][700:, :] = 0 # mask out the car parts in back image
        
        data['P2'] = np.zeros((3, 4), dtype=np.float32)
        data['P2'][0:3, 0:3] = P2
        data['original_P2'] = data['P2'].copy()
        data['camera_type_index'] = camera_type_index
        data[('filename', 0)] = os.path.join(*sample[self.image_keys[0]].split('/')[-3:])
        data['camera_type'] = camera_type

        if self.is_read_vo_depth:
            vo_path = data[('filename', 0)].replace('samples', self.vo_path).replace('.jpg', '.png')
            if os.path.isfile(vo_path):
                vo_depth = read_vo_depth(
                    data[('filename', 0)].replace('samples', self.vo_path).replace('.jpg', '.png')
                )
                data[('vo_depth', 0)] = vo_depth
            else:
                print(f'No VO Depth file found at {index}, {vo_path}')

        data = self.transform(deepcopy(data))
        return data
