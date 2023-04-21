# KITTI-360 Fisheye Datasets

We offer to test on the fisheye sequences of KITTI-360 dataset.

[![fisheye_demo](https://i1.hdslb.com/bfs/archive/eee8af351fd745f8299f986d214fd30d9ebb3e78.jpg@320w_200h_1c_!web-space-index-myvideo.webp)](https://www.bilibili.com/video/BV1Qo4y1j7NL/?spm_id_from=333.999.0.0)

## Resize fisheye images

We notice that if we are loading $1400\times 1400$ images and resizing them to $384\times 384$ online, the training speed will be slower. It is possible to resize all fisheye images to $384\times 384$ before training.

## Dataset validation / visualization in ROS (Optional)
Check [kitti360_visualize](https://github.com/Owen-Liuyuxuan/kitti360_visualize) repo. 

## Training Schedule

Baseline:
```bash
## copy example config
cd config
cp kitti360_fisheye_example kitti360_fisheye.py

## Modify config path
nano kitti360_fisheye.py
cd ..

## Train
./launcher/train.sh configs/kitti360_fisheye.py 0 $experiment_name

## Evaluation
python3 scripts/test.py configs/kitti360_fisheye.py 0 $CHECKPOINT_PATH
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