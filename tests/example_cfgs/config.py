from easydict import EasyDict as edict
import os
import numpy as np

cfg = edict()



## path
def build_path():
    path = edict()
    path.raw_path = "/data/kitti_raw"
    path.depth_path = "/data/data_depth_annotated"
    path.test_path = "/data/kitti_raw"
    path.base_path = "../.."
    #path.pretrained_checkpoint = "/home/yxliu/multi_cam/monodepth/workdirs/MonoDepth2/checkpoint/MonoDepthMeta_latest.pth"
    return path

cfg.path = build_path()

## trainer
trainer = edict(
    gpu = 0,
    max_epochs = 20,
    disp_iter = 50,
    save_iter = 5,
    test_iter = 5,
    training_hook = edict(
        name='lib.pipeline_hooks.train_val_hooks.base_training_hooks.BaseTrainingHook',
        clip_gradients=1.0,
    ),
    evaluate_hook = edict(
        name="lib.pipeline_hooks.evaluation_hooks.base_evaluation_hooks.BaseEvaluationHook",
        test_run_hook_cfg=edict(name='lib.pipeline_hooks.train_val_hooks.base_validation_hooks.BaseValidationHook'),
        result_write_cfg=edict(name='lib.data.data_writer.kitti_depth_writer.KittiDepthWriter'),
        dataset_eval_cfg=edict(
            name='lib.evaluation.kitti_unsupervised_eval.KittiEigenEvaluator',
            data_path=cfg.path.raw_path,
            split_file="../../meta_data/eigen/test_files.txt",
            gt_saved_file="../../meta_data/eigen/gt_depths.npz",
        ),

    )
)
cfg.trainer = trainer

## optimizer
optimizer = edict(
    name = 'adam',
    lr        = 1e-4,
    weight_decay = 0,
)
cfg.optimizer = optimizer
## scheduler
scheduler = edict(
    name = 'StepLR',
    step_size = 15
)
cfg.scheduler = scheduler

## data
data = edict(
    batch_size = 12,
    num_workers = 6,
    rgb_shape = (192, 640, 3),
    train_dataset = "lib.data.datasets.mono_dataset.KittiDepthMonoDataset",
    val_dataset   = "lib.data.datasets.mono_dataset.KittiDepthMonoEigenTestDataset",
    test_dataset  = "lib.data.datasets.mono_dataset.KittiDepthMonoEigenTestDataset",
    train_split_file = os.path.join(cfg.path.base_path, 'meta_data', 'eigen_zhou', 'train_files.txt'),
    val_split_file   = os.path.join(cfg.path.base_path, 'meta_data', 'eigen', 'test_files.txt'),
    frame_idxs  = [0, 1, -1],
)

resize_image_keys=[('image', idx) for idx in data.frame_idxs] + [('original_image', idx) for idx in data.frame_idxs]
color_augmented_image_keys = [('image', idx) for idx in data.frame_idxs]
data.augmentation = edict(
    rgb_mean = np.array([0.485, 0.456, 0.406]),
    rgb_std  = np.array([0.229, 0.224, 0.225]),
    cropSize = (data.rgb_shape[0], data.rgb_shape[1]),
    key_mappings=edict(
        image_keys=resize_image_keys,
        calib_keys=['P2'],
    )
)
data.train_augmentation = edict(
    name='lib.utils.builder.Sequential',
    cfg_list = [
        edict(name='lib.data.augmentations.augmentations.ConvertToFloat'),
        edict(name='lib.data.augmentations.augmentations.Resize', size=data.augmentation.cropSize, preserve_aspect_ratio=False # this should rewrite the keywords outside 
            ),
        edict(name="lib.data.augmentations.augmentations.Shuffle", 
              cfg_list=[
                    edict(name="lib.data.augmentations.augmentations.RandomBrightness", distort_prob=1.0),
                    edict(name="lib.data.augmentations.augmentations.RandomContrast", distort_prob=1.0, lower=0.6, upper=1.4),
                    edict(name="lib.utils.builder.Sequential",
                        cfg_list=[
                            edict(name="lib.data.augmentations.augmentations.ConvertColor", transform='HSV'),
                            edict(name="lib.data.augmentations.augmentations.RandomSaturation", distort_prob=1.0, lower=0.6, upper=1.4),
                            edict(name="lib.data.augmentations.augmentations.ConvertColor", current='HSV', transform='RGB'),
                        ] 
                    )
            ],
            image_keys=color_augmented_image_keys,
        ),
        edict(name='lib.data.augmentations.augmentations.RandomMirror', mirror_prob=0.5),
        edict(name='lib.data.augmentations.augmentations.Normalize', mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std, image_keys=color_augmented_image_keys),
        edict(name='lib.data.augmentations.augmentations.Normalize', mean=np.array([0, 0, 0]), stds=np.array([1, 1, 1]), image_keys=[('original_image', idx) for idx in data.frame_idxs]),
        edict(name='lib.data.augmentations.augmentations.ConvertToTensor'),
        
    ],
    **data.augmentation.key_mappings # common keywords
)


data.test_augmentation = edict(
    name='lib.utils.builder.Sequential',
    cfg_list=[
            edict(name='lib.data.augmentations.augmentations.ConvertToFloat'),
            edict(name='lib.data.augmentations.augmentations.Resize', size=data.augmentation.cropSize, preserve_aspect_ratio=False),
            edict(name='lib.data.augmentations.augmentations.Normalize', mean=data.augmentation.rgb_mean, stds=data.augmentation.rgb_std),
            edict(name='lib.data.augmentations.augmentations.ConvertToTensor'),
    ],
    image_keys=[('image', 0)], 
    calib_keys=['P2']
)
    
cfg.data = data

## networks
meta_arch = edict(
    name='lib.networks.models.meta_archs.monodepth2_model.MonoDepthMeta',

    depth_backbone_cfg = edict(
        name='lib.networks.models.backbone.resnet.resnet',
        depth=18,
        pretrained=True,
        frozen_stages=-1,
        num_stages=4,
        out_indices=(-1, 0, 1, 2, 3),
        norm_eval=False,
        dilations=(1, 1, 1, 1),
    ),

    pose_backbone_cfg = edict(
        name='lib.networks.models.backbone.resnet.resnet',
        depth=18,
        pretrained=True,
        frozen_stages=-1,
        num_stages=4,
        out_indices=(-1, 0, 1, 2, 3),
        norm_eval=False,
        dilations=(1, 1, 1, 1),
        num_input_images=2,
    ),

    head_cfg = edict(
        name='lib.networks.models.heads.monodepth2_decoder.MonoDepth2Decoder',
        scales=[0, 1, 2, 3],
        height=data.rgb_shape[0],
        width=data.rgb_shape[1],
        min_depth=0.1,
        max_depth=100.0,
        depth_decoder_cfg=edict(
            name='lib.networks.models.heads.depth_decoder.DepthDecoder',
            num_ch_enc=np.array([64, 64, 128, 256, 512]),
            num_output_channels=1,
            use_skips=True,
            scales=[0, 1, 2, 3],
        ),
        pose_decoder_cfg=edict(
            name='lib.networks.models.heads.pose_decoder.PoseDecoder',
            num_ch_enc=np.array([64, 64, 128, 256, 512]),
            num_input_features=1,
            num_frames_to_predict_for=2,
            stride=1
        )
    ),

    train_cfg = edict(
        frame_ids=[0, 1, -1],
    ),

    test_cfg = edict(

    ),
)


cfg.meta_arch = meta_arch
