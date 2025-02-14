#!/bin/bash
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

export TORCH_CUDA_ARCH_LIST="5.2;6.0;6.1;6.2;7.0;7.5;8.0+PTX"
sudo rm -rf build/ dist/ sparseconvnet.egg-info sparseconvnet_SCN*.so
sudo python3 setup.py develop
python3 examples/hello-world.py
