#!/bin/bash
set -e

TORCH_VER=$(python3 -c "import torch;print(torch.__version__)")
CUDA_VER=$(python3 -c "import torch;print(torch.version.cuda)")

if [[ $CUDA_VER < "10.0" || $TORCH_VER < '1.0' ]] ; then 
    echo "The current version of pytorch/cuda is $TORCH_VER/$CUDA_VER which is not compatible."
else
    echo "Start building"

    pushd vision_base/networks/ops/dcn
    sh make.sh
    rm -r build
    popd

fi