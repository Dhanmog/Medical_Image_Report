#!/bin/bash
# Script to run DenseNet121 v3 training with GPU enablement

PYTHON=~/miniconda3/envs/tf_gpu/bin/python

# Find all lib directories under nvidia packages in site-packages to set LD_LIBRARY_PATH
NVIDIA_LIB_PATH=$(find ~/miniconda3/envs/tf_gpu/lib/python3.12/site-packages/nvidia/ -type d -name lib | paste -sd : -)
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}:${LD_LIBRARY_PATH}"

echo "Starting training via GPU..."
$PYTHON medical_image_report/scripts/training/train_densenet121_mimic_v3.py "$@"
