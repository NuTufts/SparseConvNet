#!/bin/bash
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Set these to help find cuda
CUDA_HOME=/usr/local/cuda
CUDA_BIN=${CUDA_HOME}/bin
CUDA_LIB=${CUDA_HOME}/lib64
export PATH=${CUDA_BIN}:${PATH}
export LD_LIBRARY_PATH=${CUDA_LIB}:${LD_LIBRARY_PATH}

rm -rf build/ dist/ sparseconvnet.egg-info
python setup.py install && python examples/hello-world.py
