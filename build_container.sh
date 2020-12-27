#!/bin/bash
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Set these to help find cuda
CUDA_HOME=/usr/local/cuda
CUDA_BIN=${CUDA_HOME}/bin
CUDA_LIB=${CUDA_HOME}/lib64/stubs
export PATH=${CUDA_BIN}:${PATH}
export LD_LIBRARY_PATH=${CUDA_LIB}:${LD_LIBRARY_PATH}
export TORCH_CUDA_ARCH_LIST="5.2;6.0;6.1;6.2;7.0;7.5;8.0+PTX"

rm -rf build/ dist/ sparseconvnet.egg-info
python3 setup.py install && python examples/hello-world.py
