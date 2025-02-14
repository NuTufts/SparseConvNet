#!/bin/bash
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Set these to help find cuda
rm -rf build/ dist/ sparseconvnet.egg-info
python3 setup_cpu.py install && python3 examples/hello-world.py
