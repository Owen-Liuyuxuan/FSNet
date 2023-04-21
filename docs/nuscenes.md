# NuScenes Datasets

We offer to test on the daytime sequences of Nuscenes dataset.

[![nusc_demo](https://i0.hdslb.com/bfs/archive/3766bf39218559a23e976ff745298ae174710889.jpg@320w_200h_1c_!web-space-index-myvideo.webp)](https://www.bilibili.com/video/BV1NR4y1z7jg/?spm_id_from=333.999.0.0)

## Jsonify NuScenes dataset

Check [meta_data/nusc_trainsub/json_from_cfg.ipynb](../meta_data/nusc_trainsub/json_from_cfg.ipynb) and modify the data path.

Run through the notebook to output jsonify nuscenes data. This will increase start-up speed and lower dataset memory usage. 

## Dataset validation / visualization in ROS (Optional)
Check [nuscenes_visualize](https://github.com/Owen-Liuyuxuan/nuscenes_visualize) repo. 

## Training Schedule

Baseline:
```bash
## copy example config
cd config
cp nuscenes_wpose_example nuscenes_wpose.py

## Modify config path
nano nuscenes_wpose.py
cd ..

## Train
./launcher/train.sh configs/nuscenes_wpose.py 0 $experiment_name

## Evaluation
python3 scripts/test.py configs/nuscenes_wpose.py 0 $CHECKPOINT_PATH
```

It's fine to just use the baseline model for projects. After training baseline, you can further re-train with self-distillation:
```bash
## export checkpoint
python3 monodepth/transform_teacher.py $Pretrained_checkpoint $output_compressed_checkpoint

## copy example config 
cd config
cp distill_nuscenes_example distill_nuscenes.py

## Modify config path and checkpoint path based on  $output_compressed_checkpoint
nano distill_nuscenes.py
cd ..

## Train
./launcher/train.sh configs/distill_nuscenes.py 0 $experiment_name
```

## Visualize with jupyter notebook

Check [demos/demo.ipynb](../demos/demo.ipynb) for visualizing datasets and simple demos.

## Onnx export

We support exporting pretrained model to onnx model, and you need to install onnx and onnxruntime.
```bash
python3 scripts/onnx_export.py $CONFIG_FILE $CHECKPOINT_PATH $ONNX_PATH 
```

## Online ROS full demo

1. Launch [nuscenes_visualize](https://github.com/Owen-Liuyuxuan/nuscenes_visualize) to stream image data topics and Rviz visualization.
2. Launch [monodepth_ros](https://github.com/Owen-Liuyuxuan/monodepth_ros) to infer on camera topics.

For nuscenes, we offer an additional node to inference six images in batches. Please make sure your computer is powerful enough to infer six images online.