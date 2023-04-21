# KITTI-360 Datasets

We test on the KITTI-360 sequences

[![kitti360_demo](https://i0.hdslb.com/bfs/archive/08cec9542e99c4982efea89d47b2ae2ab31e1bbd.jpg@320w_200h_1c_!web-space-index-myvideo.webp)](https://www.bilibili.com/video/BV1sa411d7FB/?spm_id_from=333.999.0.0)


## Dataset validation / visualization in ROS (Optional)
Check [kitti360_visualize](https://github.com/Owen-Liuyuxuan/kitti360_visualize) repo. 

## Training Schedule

Baseline:
```bash
## copy example config
cd config
cp kitti360_wpose_example kitti360_wpose.py

## Modify config path
nano kitti360_wpose.py
cd ..

## Train
./launcher/train.sh configs/kitti360_wpose.py 0 $experiment_name

## Evaluation
python3 scripts/test.py configs/kitti360_wpose.py 0 $CHECKPOINT_PATH
```

It's fine to just use the baseline model for projects. After training baseline, you can further re-train with self-distillation:
```bash
## export checkpoint
python3 monodepth/transform_teacher.py $Pretrained_checkpoint $output_compressed_checkpoint

## copy example config 
cd config
cp distill_kitti360_example distill_kitti360.py

## Modify config path and checkpoint path based on  $output_compressed_checkpoint
nano distill_kitti360.py
cd ..

## Train
./launcher/train.sh configs/distill_kitti360.py 0 $experiment_name
```

## Visualize with jupyter notebook

Check [demos/demo.ipynb](../demos/demo.ipynb) for visualizing datasets and simple demos.

## Onnx export

We support exporting pretrained model to onnx model, and you need to install onnx and onnxruntime.
```bash
python3 scripts/onnx_export.py $CONFIG_FILE $CHECKPOINT_PATH $ONNX_PATH 
```

## Online ROS full demo

1. Launch [kitti360_visualize](https://github.com/Owen-Liuyuxuan/kitti360_visualize) to stream image data topics and Rviz visualization.
2. Launch [monodepth_ros](https://github.com/Owen-Liuyuxuan/monodepth_ros) to infer on camera topics.
