import numpy as np
import os
import tqdm

from monodepth.data.datasets.fusionportable_dataset import read_ouster_calib, read_pcd_file, read_camera_calib
from monodepth.networks.utils.monodepth_utils import project_depth_map
from .kitti_unsupervised_eval import KittiEigenEvaluator

class FusionPortableEvaluator(KittiEigenEvaluator):
    def _load_calib(self, calib_dir):
        self.ouster_calib = read_ouster_calib(os.path.join(calib_dir, 'ouster00.yaml'))
        self.cam00_calib  = read_camera_calib(os.path.join(calib_dir, 'frame_cam00.yaml'))

    def _precompute(self, data_path, split_file, gt_saved_file):
        calib_dir = os.path.join(data_path, 'calib')
        pc_dir    = os.path.join(data_path, 'ouster00', 'point', 'data')

        self._load_calib(calib_dir)

        with open(split_file, 'r' ) as f:
            lines = f.readlines()
        
        R_rect = np.eye(4)
        R_rect[0:3, 0:3] = self.cam00_calib['R']
        P_ouster2img = self.cam00_calib['P'] @ R_rect @ np.linalg.inv(self.ouster_calib['T_cam002ouster'])
        gt_depths = []
        for line in tqdm.tqdm(lines, dynamic_ncols=True):
            index = int(line.strip())

            lidar_filename = os.path.join(pc_dir, "{:06d}.pcd".format(index))
            lidar = read_pcd_file(lidar_filename)
            if lidar.shape[1] == 3:
                lidar = np.concatenate((lidar, np.ones((lidar.shape[0], 1))), axis=1) # homogeneous coordinates

            image_shape = np.array([self.cam00_calib['height'], self.cam00_calib['width']]) #

            gt_depth = project_depth_map(lidar, P_ouster2img, image_shape)

            gt_depths.append(gt_depth.astype(np.float32))
        
        np.savez_compressed(gt_saved_file, data=np.array(gt_depths))
        self.gt_depths = gt_depths
