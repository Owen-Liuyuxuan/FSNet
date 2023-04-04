
import os
import numpy as np
import cv2
import tqdm
from functools import reduce
from collections import Counter
from pyquaternion import Quaternion
from torch.utils.tensorboard import SummaryWriter
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from monodepth.evaluation.kitti_unsupervised_eval import KittiEigenEvaluator
from monodepth.data.datasets.utils import read_depth
from monodepth.networks.utils.monodepth_utils import compute_errors
from vision_base.data.datasets.nuscenes_utils import NuScenes

def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['LIDAR_TOP']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                       inverse=True)

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points

def pad_or_trim_to_np(x, shape, pad_val=0):
    shape = np.asarray(shape)
    pad = shape - np.minimum(np.shape(x), shape)
    zeros = np.zeros_like(pad)
    x = np.pad(x, np.stack([zeros, pad], axis=1), constant_values=pad_val)
    return x[:shape[0], :shape[1]]

def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def generate_depth_map(velo, extrinsics, intrinsics, cam=2, im_shape=[900, 1600]):
    """Generate a depth map from velodyne data
    """
    N = velo.shape[0]
    homo_velo = np.ones([N, 4])
    homo_velo[:, 0:3] = velo[:, 0:3]
    homo_intrinsics = np.eye(4)
    homo_intrinsics[0:3, 0:3] = intrinsics
    projection_matrix = np.dot(
        homo_intrinsics, np.linalg.inv(extrinsics)
    ) # [4, 4]
    

    # project the points to the camera
    velo_pts_im = np.dot(projection_matrix, homo_velo.T).T #[N, 4]
    velo_pts_im = velo_pts_im[velo_pts_im[:, 2] > 0]
    
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / (velo_pts_im[:, 2][..., np.newaxis])

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1] # type: ignore
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

def get_samples(nusc):
    samples = [samp for samp in nusc.sample]

    # sort by scene, timestamp (only to make chronological viz easier)
    samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

    return samples

def get_lidar(nusc, rec):
    lidar_data = get_lidar_data(nusc, rec, nsweeps=1, min_distance=2.2)
    lidar_data = lidar_data.transpose(1, 0)
    num_points = lidar_data.shape[0]
    lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
    lidar_mask = np.ones(81920).astype('float32')
    lidar_mask[num_points:] *= 0.0
    return lidar_data, lidar_mask

    

class NuscenesEvaluator(KittiEigenEvaluator):
    def __init__(self,
                 data_path, #path to nusc_base_dir
                 split_file,
                 gt_saved_dir,
                 nuscenes_version='v1.0-trainval',
                 is_evaluate_absolute=False,
                 is_force_recompute=False,
                 channels=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
                 ):
        self.is_evaluate_absolute = is_evaluate_absolute
        self.split_file = split_file
        with open(split_file, 'r') as f:
            self.token_list = [line.strip().split(',')[0] for line in f.readlines()]
        if (not os.path.isdir(gt_saved_dir)) or is_force_recompute:
            print(f"Start exporting ground truth depths specified by {nuscenes_version} to {gt_saved_dir}")
            self._precompute(data_path, gt_saved_dir, nuscenes_version)
        
        self.channels = channels
        self.gt_saved_dir = gt_saved_dir

    def _precompute(self, data_path, gt_saved_dir, nuscenes_version):
        nusc = NuScenes(version=nuscenes_version, dataroot = data_path,verbose=True)

        CAMs = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
        # Initialize
        for cam in CAMs:
            os.makedirs(
                os.path.join(gt_saved_dir, cam), exist_ok=True
            )
        
        for token in tqdm.tqdm(self.token_list, dynamic_ncols=True):
            rec = nusc.get('sample', token)
            lidar_data, lidar_mask = get_lidar(nusc, rec)
            lidar = lidar_data[lidar_mask==1, :]
            for cam in CAMs:
                samp = nusc.get('sample_data', rec['data'][cam])
                im_shape = [samp['height'], samp['width']]
                filename = samp['filename']
                depth_name = filename.replace('samples', gt_saved_dir).replace('.jpg', '.png')

                # extrinsics and intrinsics
                sens = nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
                trans = np.array(sens['translation'])
                rots = np.array(Quaternion(sens['rotation']).rotation_matrix)
                intrins = np.array(sens['camera_intrinsic'])
                T = np.eye(4)
                T[0:3,0:3] = rots
                T[0:3, 3] = trans

                depth = generate_depth_map(lidar, T, intrins, im_shape=im_shape)
                depth_uint16 = (depth*256).astype(np.uint16)
                cv2.imwrite(
                    depth_name, depth_uint16
                )
    
    def log(self, writer, channel, mean_errors, mean_abs_errors, global_step=0, epoch_num=0, is_print=True):
        log_str = f"Epoch {epoch_num} for channel {channel}"
        log_str += "\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
        log_str += "\n" + ("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\"

        log_str += f"\nEpoch {epoch_num} for channel {channel} | Abs Error without Scaled"
        log_str += "\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
        log_str += "\n" + ("&{: 8.3f}  " * 7).format(*mean_abs_errors.tolist()) + "\\\\"

        if writer is not None:
            writer.add_text(f"Evaluation logs/{channel}", log_str.replace(' ', '&nbsp;').replace('\n', '  \n'), global_step=epoch_num)
        
        if is_print:
            print(log_str)

    def _single_loss(self, depth_0, gt_depth):
        gt_height, gt_width = gt_depth.shape[:2]
        pred_depth = cv2.resize(depth_0, (gt_width, gt_height))
        mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80.0)

        crop = np.array([0.03594771 * gt_height, 0.99189189 * gt_height,
                        0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        if len(pred_depth) == 0 or len(gt_depth) == 0:
            raise ValueError
        
        ratio = np.median(gt_depth) / np.median(pred_depth)
        scaled_depth = pred_depth * ratio

        scaled_depth[scaled_depth < 1e-3] = 1e-3
        scaled_depth[scaled_depth > 80.0] = 80.0

        error = compute_errors(gt_depth, scaled_depth)

        pred_depth[pred_depth < 1e-3] = 1e-3
        pred_depth[pred_depth > 80.0] = 80.0

        abs_error = compute_errors(gt_depth, pred_depth)
        return dict(
            ratio = ratio,
            error = error,
            abs_error = abs_error
        )

    def single_call(self, depth_0, filename):
        gt_depth = read_depth(filename.replace('samples', self.gt_saved_dir).replace('.jpg', '.png'))
        return self._single_loss(depth_0, gt_depth)
    
    def __call__(self, result_path, writer:SummaryWriter=None, global_step=0, epoch_num=0):

        all_mean_errors = []
        all_mean_errors_abs = []
        for cam in self.channels:
            errors = []
            abs_errors = []
            ratios = []

            predict_dir = os.path.join(result_path, 'predict_depth', cam)
            filelist = os.listdir(predict_dir)

            gt_dir = os.path.join(self.gt_saved_dir, cam)
            print(f'Evaminating images at {predict_dir} against {gt_dir}')
            for i, image_file in enumerate(tqdm.tqdm(filelist, dynamic_ncols=True)):
                gt_depth = read_depth(os.path.join(gt_dir, image_file))
                gt_height, gt_width = gt_depth.shape[:2]

                pred_depth = read_depth(os.path.join(predict_dir, image_file))
                pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))
                    
                mask = np.logical_and(gt_depth > 1e-3, gt_depth < 80.0)

                crop = np.array([0.03594771 * gt_height, 0.99189189 * gt_height,
                                0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

                pred_depth = pred_depth[mask]
                gt_depth = gt_depth[mask]

                if len(pred_depth) == 0 or len(gt_depth) == 0:
                    import warnings
                    sample_token = image_file.split('.')[0]
                    warnings.warn(f"image at sample {sample_token} from camera {cam} as no usable points")
                    continue

                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                scaled_depth = pred_depth * ratio

                scaled_depth[scaled_depth < 1e-3] = 1e-3
                scaled_depth[scaled_depth > 80.0] = 80.0

                errors.append(compute_errors(gt_depth, scaled_depth))

                pred_depth[pred_depth < 1e-3] = 1e-3
                pred_depth[pred_depth > 80.0] = 80.0

                abs_errors.append(compute_errors(gt_depth, pred_depth))

            print(np.array(errors).shape, cam)
            mean_errors = np.array(errors).mean(0)
            mean_abs_errors = np.array(abs_errors).mean(0)
            
            self.log(writer, cam, mean_errors, mean_abs_errors, global_step=global_step, epoch_num=epoch_num)

            all_mean_errors.append(mean_errors)
            all_mean_errors_abs.append(mean_abs_errors)

        all_mean_errors = np.array(all_mean_errors).mean(0)
        all_mean_errors_abs = np.array(all_mean_errors_abs).mean(0)
        self.log(writer, 'all mean', all_mean_errors, all_mean_errors_abs, global_step=global_step, epoch_num=epoch_num)


if __name__ == '__main__':
    from fire import Fire
    Fire(NuscenesEvaluator._precompute)
