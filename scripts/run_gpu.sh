#!/bin/bash
# Helper script to run any python command with GPU enabled under WSL
PYTHON=/home/dhanu/miniconda3/envs/tf_gpu/bin/python
NVIDIA_LIB_PATH=$(find /home/dhanu/miniconda3/envs/tf_gpu/lib/python3.12/site-packages/nvidia/ -type d -name lib | paste -sd : -)
export LD_LIBRARY_PATH="${NVIDIA_LIB_PATH}:${LD_LIBRARY_PATH}"

$PYTHON "$@"
