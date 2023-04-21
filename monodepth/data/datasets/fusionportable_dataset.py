from __future__ import print_function, division
import os
import numpy as np
from typing import List
from easydict import EasyDict
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from copy import deepcopy
from multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader # noqa: F401

import torch
import torch.utils.data
from monodepth.data.datasets.utils import read_image, cam_relative_pose_nusc
from vision_base.utils.builder import build

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat


yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)
    
def read_opencv_yaml(file_path):
    # loading
    with open(file_path) as fin:
        c = fin.read()
        # some operator on raw conent of c may be needed
        c = "%YAML 1.1"+os.linesep+"---" + c[len("%YAML:1.0"):] if c.startswith("%YAML:1.0") else c
        result = yaml.full_load(c)
    return result

def read_pcd_file(file_name):
    pcd = o3d.io.read_point_cloud(file_name)
    point_array = np.asarray(pcd.points)
    return point_array

def read_camera_calib(file):
    camera_yaml = read_opencv_yaml(file)
    K = camera_yaml["camera_matrix"]
    distortion_model = camera_yaml["distortion_model"]
    R = camera_yaml["rectification_matrix"]
    D = camera_yaml["distortion_coefficients"]
    P = camera_yaml["projection_matrix"]
    height = camera_yaml["image_height"]
    width = camera_yaml["image_width"]
    q_imu2cam = camera_yaml["quaternion_sensor_bodyimu"][0] #qw, qx, qy, qz
    q_imu2cam = [q_imu2cam[1], q_imu2cam[2], q_imu2cam[3], q_imu2cam[0]] #[qx, qy, qz, qw]
    t_imu2cam = camera_yaml["translation_sensor_bodyimu"][0]
    result=dict(
        K=K, distortion_model=distortion_model, R=R, D=D, P=P, height=height, width=width, q_imu2cam=q_imu2cam, t_imu2cam=t_imu2cam, T_imu2cam=T_from_quaternion_translation(q_imu2cam, t_imu2cam)
    )
    return result

def read_ouster_calib(file):
    calib_yaml = read_opencv_yaml(file)
    q_imu2ouster = calib_yaml["quaternion_sensor_bodyimu"][0] #qw, qx, qy, qz
    q_imu2ouster = [q_imu2ouster[1], q_imu2ouster[2], q_imu2ouster[3], q_imu2ouster[0]] #[qx, qy, qz, qw]
    t_imu2ouster = calib_yaml["translation_sensor_bodyimu"][0]

    q_cam002ouster = calib_yaml["quaternion_sensor_frame_cam00"][0] #qw, qx, qy, qz
    q_cam002ouster = [q_cam002ouster[1], q_cam002ouster[2], q_cam002ouster[3], q_cam002ouster[0]] #[qx, qy, qz, qw]
    t_cam002ouster = calib_yaml["translation_sensor_frame_cam00"][0]
    result=dict(
        q_imu2ouster=q_imu2ouster, t_imu2ouster=t_imu2ouster, T_imu2ouster=T_from_quaternion_translation(q_imu2ouster, t_imu2ouster),
        q_cam002ouster=q_cam002ouster, t_cam002ouster=t_cam002ouster, T_cam002ouster=T_from_quaternion_translation(q_cam002ouster, t_cam002ouster)
    )
    return result


def read_odom(file):
    t_list = []
    q_list = []
    T_list = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            elements = line.split(" ")
            t_list.append(np.array([float(x) for x in elements[1:4]]))
            q_list.append(np.array([float(x) for x in elements[4:8]]))
            T_list.append(T_from_quaternion_translation(q_list[-1], t_list[-1]))
    return dict(t_list=np.array(t_list), q_list=np.array(q_list), T_list=np.array(T_list))

def T_from_quaternion_translation(q, t):
    rotation = R.from_quat(q)
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    T[:3, 3] = t
    return T

def read_split_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        return [int(line.strip()) for line in lines]

class FusionportableMonoDataset(torch.utils.data.Dataset):
    """Some Information about KittiDataset"""
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(FusionportableMonoDataset, self).__init__()
        self.base_path    = data_cfg.base_path

        self.use_right_image = getattr(data_cfg, 'use_right_image', True)
        self.frame_idxs  = data_cfg.frame_idxs

        split_file = data_cfg.split_file

        manager = Manager() # multithread manage wrapping for list objects

        self.imdb = read_split_file(split_file)

        self.meta_dict = {}
        self.meta_dict['calib'] = {}
        self.meta_dict['calib']['Cam00'] = read_camera_calib(os.path.join(self.base_path, 'calib', 'frame_cam00.yaml'))
        self.meta_dict['calib']['Cam01'] = read_camera_calib(os.path.join(self.base_path, 'calib', 'frame_cam01.yaml'))
        self.meta_dict['calib']['Ouster00'] = read_ouster_calib(os.path.join(self.base_path, 'calib', 'ouster00.yaml'))
        self.meta_dict['poses'] = read_odom(os.path.join(self.base_path, '20220226_campus_road_day.txt'))
        
        self.meta_dict = manager.dict(self.meta_dict)
        self.is_filter_static = getattr(data_cfg, 'is_filter_static', True)
        if self.is_filter_static:
            self.imdb = self._filter_static_indexes()
        self.transform = build(**data_cfg.augmentation)
        

    def _filter_static_indexes(self):
        imdb = []
        print(f"Start Filtering Static indexes, original length {len(self)}")
        for index in self.imdb:
            is_static = False
            imu2world_s = self.get_pose([index + idx for idx in self.frame_idxs])
            T_imu2cam = self.meta_dict['calib']['Cam00']['T_imu2cam']
            for i, idx in enumerate(self.frame_idxs[1:]):
                pose = cam_relative_pose_nusc(imu2world_s[0], imu2world_s[i + 1], T_imu2cam).astype(np.float32)
                if np.linalg.norm(pose[0:3, 3]) < 0.03:
                    is_static = True
            
            if not is_static:
                imdb.append(index)
        print(f"Finished filtering static indexes, find dynamic instances {len(imdb)}")
        return imdb
        
    def __getitem__(self, i):
        index = self.imdb[i]

        if (not self.use_right_image) or (np.random.rand() < 0.5):
            calib = self.meta_dict['calib']['Cam00']
            image_dir_name = 'frame_cam00'
        else:
            calib = self.meta_dict['calib']['Cam01']
            image_dir_name = 'frame_cam01'

        data = dict()
        for idx in self.frame_idxs:
            data[("image", idx)] = self.get_color(index + idx, image_dir_name)
            data[('original_image', idx)] = data[('image', idx)].copy()
        h, w, _ = data[("image", 0)].shape
        data["patched_mask"] = np.ones([h, w])
        
        imu2world_s = self.get_pose([index + idx for idx in self.frame_idxs])
        T_imu2cam = calib['T_imu2cam']
        for i, idx in enumerate(self.frame_idxs[1:]):
            data[('relative_pose', idx)] = cam_relative_pose_nusc(imu2world_s[0], imu2world_s[i + 1], T_imu2cam).astype(np.float32)

        data['P2'] = calib['P']

        data['original_P2'] = data['P2'].copy()
        
        data = self.transform(deepcopy(data))

        return data

    def __len__(self):
        return len(self.imdb)

    def get_color(self, frame_index, image_dir_name):
        image_dir = os.path.join(self.base_path, image_dir_name, 'image', 'data', '%06d.png' % frame_index)

        return read_image(image_dir)


    def get_pose(self, frame_indexes:List[int], *args, **kwargs):
        poses = self.meta_dict['poses']['T_list'][frame_indexes, :, :]
        return poses
