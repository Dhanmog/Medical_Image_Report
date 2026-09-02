#!/bin/bash
# Test GPU detection in WSL with proper CUDA library paths

PYTHON=~/miniconda3/envs/tf_gpu/bin/python

# Find all lib directories under nvidia packages in site-packages
NVIDIA_LIB_PATH=$(find ~/miniconda3/envs/tf_gpu/lib/python3.12/site-packages/nvidia/ -type d -name lib | paste -sd : -)

export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

$PYTHON -c "
import tensorflow as tf
print('TF version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('GPUs found:', gpus)
print('Num GPUs:', len(gpus))
if gpus:
    print('SUCCESS: GPU is available for training!')
else:
    print('WARNING: No GPU detected.')
"
