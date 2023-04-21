from __future__ import print_function, division
import os
import numpy as np
import cv2
from typing import List
from easydict import EasyDict

from copy import deepcopy
from multiprocessing import Manager
from torch.utils.data import Dataset, DataLoader # noqa: F401

import torch
import torch.utils.data
from monodepth.data.datasets.utils import read_image, read_depth, read_pose_mat, cam_relative_pose
from vision_base.utils.builder import build


def read_K_from_depth_prediction(file):
    with open(file, 'r') as f:
        line = f.readlines()[0]
        data = line.split(" ")
        K    = np.array([float(data[i]) for i in range(len(data[0:9]))])
        return np.reshape(K, (3, 3))

def read_P23_from_sequence(file):
    """ read P2 and P3 from a sequence file calib_cam_to_cam.txt
    """
    P2 = None
    P3 = None
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("P_rect_02"):
                data = line.split(" ")
                P2 = np.array([float(x) for x in data[1:13]])
                P2 = np.reshape(P2, [3, 4])
            if line.startswith("P_rect_03"):
                data = line.split(" ")
                P3 = np.array([float(x) for x in data[1:13]])
                P3 = np.reshape(P3, [3, 4])
    assert P2 is not None, f"can not find P2 in file {file}"
    assert P3 is not None, f"can not find P3 in file {file}"
    return P2, P3

def read_imu2velo(file):
    T = np.eye(4)
    R = None
    t = None
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("R"):
                data = line.split(" ")
                R = np.array([float(x) for x in data[1:10]])
                R = np.reshape(R, [3, 3])
            if line.startswith("T"):
                data = line.split(" ")
                t = np.array([float(x) for x in data[1:4]])
                t = np.reshape(t, [3, 1])
    assert R is not None, f"can not find R in file {file}"
    assert t is not None, f"can not find t in file {file}"
    T[0:3, 0:3] = R
    T[0:3, 3:4] = t
    return T

def read_T_from_sequence(file):
    """ read T from a sequence file calib_velo_to_cam.txt
    """
    R = None
    T = None
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("R:"):
                data = line.split(" ")
                R = np.array([float(x) for x in data[1:10]])
                R = np.reshape(R, [3, 3])
            if line.startswith("T:"):
                data = line.split(" ")
                T = np.array([float(x) for x in data[1:4]])
                T = np.reshape(T, [3, 1])
    assert R is not None, "can not find R in file {}".format(file)
    assert T is not None, "can not find T in file {}".format(file)

    T_velo2cam = np.eye(4)
    T_velo2cam[0:3, 0:3] = R
    T_velo2cam[0:3, 3:4] = T
    return T_velo2cam

def read_split_file(file:str):
    imdb:list = []
    with open(file, 'r') as f:
        lines = f.readlines() # 2011_09_26/2011_09_26_drive_0022_sync 473 r
        for i in range(len(lines)):
            line = lines[i].strip().split()

            folder = line[0]
            index = int(line[1])
            side = line[2]
            datetime = folder.split("/")[0]
            imdb.append(
                dict(
                    folder=folder,
                    index=index,
                    side=side,
                    datetime=datetime
                )
            )
    return imdb

class KittiDepthMonoDataset(torch.utils.data.Dataset):
    """Some Information about KittiDataset"""
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(KittiDepthMonoDataset, self).__init__()
        self.raw_path    = data_cfg.raw_path

        self.depth_path = getattr(data_cfg, 'depth_path', None)
        self.frame_idxs  = data_cfg.frame_idxs

        split_file = data_cfg.split_file

        manager = Manager() # multithread manage wrapping for list objects

        self.imdb = read_split_file(split_file)

        self.meta_dict = {}
        for date_time in os.listdir(self.raw_path):
            folder_path = os.path.join(self.raw_path, date_time)
            if not os.path.isdir(folder_path):
                print(f"{folder_path} is not a directory, skipping")
            P2, P3 = read_P23_from_sequence(os.path.join(self.raw_path, date_time, "calib_cam_to_cam.txt"))
            T      = read_T_from_sequence  (os.path.join(self.raw_path, date_time, "calib_velo_to_cam.txt"))
            T_imu2vel = read_imu2velo      (os.path.join(self.raw_path, date_time, "calib_imu_to_velo.txt"))
            self.meta_dict[date_time] = dict(
                P2 = P2,
                P3 = P3,
                T_vel2cam = T,
                T_imu2vel = T_imu2vel,
            )
        
        self.pose_dict = {}
        for key in set([obj['folder'] for obj in self.imdb]):
            self.pose_dict[key] = read_pose_mat(os.path.join(self.raw_path, key, 'oxts', 'pose.mat'))
        
        self.meta_dict = manager.dict(self.meta_dict)
        self.is_motion_mask  = getattr(data_cfg, 'is_motion_mask', False)
        self.is_precompute_flow = getattr(data_cfg, 'is_precompute_flow', False)
        if self.is_motion_mask:
            self.precompute_path = getattr(data_cfg, 'motion_mask_path', "")
        if self.is_precompute_flow:
            self.flow_path = getattr(data_cfg, 'flow_path', "")
        self.is_filter_static = getattr(data_cfg, 'is_filter_static', True)
        if self.is_filter_static:
            self.imdb = manager.list(self._filter_static_indexes())
        self.transform = build(**data_cfg.augmentation)

        # read pose for each sequence:
        

    def _filter_static_indexes(self):
        imdb = []
        print(f"Start Filtering Static indexes, original length {len(self)}")
        for obj in self.imdb:
            is_static = False
            folder  = obj['folder']
            index   = obj['index']
            datetime= obj['datetime']
            imu2world_s = self.get_pose(folder, [index + idx for idx in self.frame_idxs])
            T_imu2vel = self.meta_dict[datetime]['T_imu2vel']
            T_vel2cam = self.meta_dict[datetime]['T_vel2cam']
            for i, idx in enumerate(self.frame_idxs[1:]):
                pose = cam_relative_pose(imu2world_s[0], imu2world_s[i + 1], T_imu2vel, T_vel2cam).astype(np.float32)
                if np.linalg.norm(pose[0:3, 3]) < 0.03:
                    is_static = True
            
            if not is_static:
                imdb.append(obj)
        print(f"Finished filtering static indexes, find dynamic instances {len(imdb)}")
        return imdb
        
    def __getitem__(self, i):
        obj = self.imdb[i]

        folder  = obj['folder']
        index   = obj['index']
        side    = obj['side']
        datetime= obj['datetime']

        data = dict()
        for idx in self.frame_idxs:
            data[("image", idx)] = self.get_color(obj['folder'], index + idx, side)
            data[('original_image', idx)] = data[('image', idx)].copy()
        h, w, _ = data[("image", 0)].shape
        data["patched_mask"] = np.ones([h, w])
        
        if self.is_motion_mask:
            data['motion_mask'] = self.get_motion_mask(i)
            
        if self.is_precompute_flow:
            data['flow'] = self.get_flow(i)

        imu2world_s = self.get_pose(folder, [index + idx for idx in self.frame_idxs])
        T_imu2vel = self.meta_dict[datetime]['T_imu2vel']
        T_vel2cam = self.meta_dict[datetime]['T_vel2cam']
        for i, idx in enumerate(self.frame_idxs[1:]):
            data[('relative_pose', idx)] = cam_relative_pose(imu2world_s[0], imu2world_s[i + 1], T_imu2vel, T_vel2cam).astype(np.float32)

        selected_key = {"l":"P2", "r":"P3"}[side]
        data['P2'] = self.meta_dict[datetime][selected_key]

        data['original_P2'] = data['P2'].copy()

        if self.depth_path is not None:
            data[('sparse_depth', 0)] = self.get_depth(folder, index, side)

        
        
        data = self.transform(deepcopy(data))

        return data

    def __len__(self):
        return len(self.imdb)

    def get_color(self, folder, frame_index, side):
        camera_folder = {"l": "image_02", "r" : "image_03"}[side]
        image_dir = os.path.join(self.raw_path, folder, camera_folder, 'data', '%010d.png' % frame_index)

        return read_image(image_dir)

    def get_depth(self, folder, frame_index, side):

        camera_folder = {"l": "image_02", "r" : "image_03"}[side]
        image_dir = os.path.join(self.depth_path, folder.split('/')[1], 'proj_depth', 'groundtruth', camera_folder, "%010d.png" % frame_index)
        #image_dir = os.path.join(self.raw_path, folder, 'depth', '%010d.png'%frame_index)
        print(image_dir, folder.split('/')[1], camera_folder)
        return read_depth(image_dir)

    def get_pose(self, folder, frame_indexes:List[int], *args, **kwargs):
        poses = self.pose_dict[folder][frame_indexes, :, :]
        return poses

    def get_motion_mask(self, i):
        path = os.path.join(self.precompute_path, f"{i:08d}.png")
        motion_mask = cv2.imread(path, cv2.IMREAD_UNCHANGED) #[H, W]
        return motion_mask

    def get_flow(self, i):
        path = os.path.join(self.flow_path, f"{i:08d}.png")
        arflow_0 = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, 0:2] #[H, W]
        arflow_0 = (arflow_0.astype(np.float32) - 2 ** 15) / 64.0
        return arflow_0


class KittiDepthMonoEigenTestDataset(torch.utils.data.Dataset):
    """Some Information about KittiDataset"""
    def __init__(self, **data_cfg):
        data_cfg = EasyDict(data_cfg)
        super(KittiDepthMonoEigenTestDataset, self).__init__()
        self.raw_path    = data_cfg.raw_path

        split_file = data_cfg.split_file

        manager = Manager() # multithread manage wrapping for list objects

        if 'depth_path' in data_cfg:
            self.depth_path = data_cfg.depth_path
        else:
            self.depth_path = None

        self.imdb = read_split_file(split_file)

        self.meta_dict = {}
        for date_time in os.listdir(self.raw_path):
            P2, P3 = read_P23_from_sequence(os.path.join(self.raw_path, date_time, "calib_cam_to_cam.txt"))
            T      = read_T_from_sequence  (os.path.join(self.raw_path, date_time, "calib_velo_to_cam.txt"))
            T_imu2vel = read_imu2velo      (os.path.join(self.raw_path, date_time, "calib_imu_to_velo.txt"))
            self.meta_dict[date_time] = dict(
                P2 = P2,
                P3 = P3,
                T_vel2cam = T,
                T_imu2vel = T_imu2vel,
            )

        self.imdb = manager.list(self.imdb)
        self.meta_dict = manager.dict(self.meta_dict)

        self.transform = build(**data_cfg.augmentation)

    def __getitem__(self, index):
        obj = self.imdb[index]

        folder  = obj['folder']
        index   = obj['index']
        side    = obj['side']
        datetime= obj['datetime']


        data = dict()
        data[("image", 0)] = self.get_color(obj['folder'], index, side)
        if index > 0:
            data[("image", -1)] = self.get_color(obj['folder'], index - 1, side)
        else:
            data[("image", -1)] = self.get_color(obj['folder'], index, side)

        data[('original_image', 0)] = data[('image', 0)].copy()

        selected_key = {"l":"P2", "r":"P3"}[side]
        data['P2'] = self.meta_dict[datetime][selected_key]

        data['original_P2'] = data['P2'].copy()

        imu2world_s = self.get_pose(folder, [index + idx for idx in [0, -1]])
        T_imu2vel = self.meta_dict[datetime]['T_imu2vel']
        T_vel2cam = self.meta_dict[datetime]['T_vel2cam']
        data[('relative_pose', -1)] = cam_relative_pose(imu2world_s[0], imu2world_s[1], T_imu2vel, T_vel2cam).astype(np.float32)


        if self.depth_path is not None:
            data[('sparse_depth', 0)] = self.get_depth(folder, index, side)
        # generate_depth_from_velo(point_cloud, image.shape[0], image.shape[1], T_velo2cam, np.eye(4), P.copy(), base_depth=gt)
        
        data = self.transform(deepcopy(data))

        return data

    def __len__(self):
        return len(self.imdb)

    def get_color(self, folder, frame_index, side):
        camera_folder = {"l": "image_02", "r" : "image_03"}[side]
        image_dir = os.path.join(self.raw_path, folder, camera_folder, 'data', '%010d.png' % frame_index)

        return read_image(image_dir)

    def get_depth(self, folder, frame_index, side):

        #camera_folder = {"l": "image_02", "r" : "image_03"}[side]
        #image_dir = os.path.join(self.depth_path, folder, 'proj_depth', 'groundtruth', camera_folder, 'data', "%010d.png"%frame_index)
        image_dir = os.path.join(self.raw_path, folder, 'depth', '%010d.png' % frame_index)

        return read_depth(image_dir)
    
    def get_pose(self, folder, frame_indexes:List[int], *args, **kwargs):
        pose_path = os.path.join(self.raw_path, folder, 'oxts', 'pose.mat')
        pose_array = read_pose_mat(pose_path)
        return pose_array[frame_indexes, :, :]
