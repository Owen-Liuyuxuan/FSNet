# KITTI Datasets

We test on the KITTI RAW sequences

## Exporting Sequence Poses for pythonic usage

Please refer to this [naive method](https://gist.github.com/Owen-Liuyuxuan/27f12e15788acba76053df84a28f2291) to produce poses for every sequence. 

## Dataset validation / visualization in ROS (Optional)
Check [kitti_visualize](https://github.com/Owen-Liuyuxuan/kitti_visualize) repo. 

## Training Schedule

Baseline:
```bash
## copy example config
cd config
cp kitti_wpose_example kitti_wpose.py

## Modify config path
nano kitti_wpose.py
cd ..

## Train
./launcher/train.sh configs/kitti_wpose.py 0 $experiment_name

## Evaluation
python3 scripts/test.py configs/kitti_wpose.py 0 $CHECKPOINT_PATH
```

It's fine to just use the baseline model for projects. After training baseline, you can further re-train with self-distillation:
```bash
## export checkpoint
python3 monodepth/transform_teacher.py $Pretrained_checkpoint $output_compressed_checkpoint

## copy example config 
cd config
cp distill_kitti_example distill_kitti.py

## Modify config path and checkpoint path based on  $output_compressed_checkpoint
nano distill_kitti.py
cd ..

## Train
./launcher/train.sh configs/distill_kitti.py 0 $experiment_name
```

## Visualize with jupyter notebook

Check [demos/demo.ipynb](../demos/demo.ipynb) for visualizing datasets and simple demos.

## Onnx export

We support exporting pretrained model to onnx model, and you need to install onnx and onnxruntime.
```bash
python3 scripts/onnx_export.py $CONFIG_FILE $CHECKPOINT_PATH $ONNX_PATH 
```

## Online ROS full demo

1. Launch [kitti_visualize](https://github.com/Owen-Liuyuxuan/kitti_visualize) to stream image data topics and Rviz visualization.
2. Launch [monodepth_ros](https://github.com/Owen-Liuyuxuan/monodepth_ros) to infer on camera topics.
