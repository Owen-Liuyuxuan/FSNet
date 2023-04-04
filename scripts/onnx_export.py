"""
    Script for launching training process
"""
import fire
import torch
import onnx 
import onnxruntime as ort


from _path_init import manage_package_logging
from vision_base.utils.builder import build
from vision_base.utils.utils import cfg_from_file
from vision_base.networks.utils.utils import load_models

def main(config:str="config/config.py",
         checkpoint_path:str="monodepth.pth",
         onnx_file='metaarch.onnx',
         input_names=['input'],
         output_names=['output'],
         gpu:int=0,
        ):
    """_summary_

    Args:
        config (str, optional): Full path to the config file. Defaults to "config/config.py".
        checkpoint_path (str, optional): Full path to the checkpoint. Defaults to "retinanet_79.pth".
        onnx_file (str, optional): Full path to the output onnx model. Defaults to 'metaarch.onnx'.
        input_names (list, optional): input nodes name of onnx model. Defaults to ['input'].
        output_names (list, optional): output nodes name of onnx model. Defaults to ['output'].
        gpu (int, optional): the gpu we are using. Defaults to 0.
    """
    # Read Config
    cfg = cfg_from_file(config)
    
    # Force GPU selection in command line
    cfg.trainer.gpu = gpu
    torch.cuda.set_device(cfg.trainer.gpu)

    manage_package_logging()

    # Create the model
    meta_arch = build(**cfg.meta_arch)
    meta_arch = meta_arch.cuda()

    load_models(checkpoint_path, meta_arch, map_location=f'cuda:{gpu}', strict=False)
    meta_arch.eval()
    print(f"Loaded model from {checkpoint_path}.")

    meta_arch.forward = meta_arch.dummy_forward
    
    dummy_input = torch.zeros([1, cfg.data.rgb_shape[2], cfg.data.rgb_shape[0], cfg.data.rgb_shape[1]]).cuda()
    torch.onnx.export(meta_arch, dummy_input, onnx_file, input_names=input_names, output_names=output_names, opset_version=11)
    print(f"Finish export, start checking the exported file {onnx_file}.")
    
    # Load the ONNX model
    onnx_model = onnx.load(onnx_file)
    # Check that the IR is well formed
    onnx.checker.check_model(onnx_model)
    print(f"Finish onnx checker check.")

    # Print a human readable representation of the graph
    print("-----------------onnx helper print-----------------")
    print(onnx.helper.printable_graph(onnx_model.graph))
    print(f"Finish onnx helper print.")

    # Run the model using ONNX Runtime
    # Load the ONNX model to ort session
    ort_session = ort.InferenceSession(onnx_file, providers=[('CUDAExecutionProvider', {'device_id':gpu})],)
    outputs = ort_session.run(None, {'input': dummy_input.cpu().numpy()})
    print(f"The actual output of onnxruntime session: outputs[0].shape={outputs[0].shape}")

if __name__ == '__main__':
    fire.Fire(main)
