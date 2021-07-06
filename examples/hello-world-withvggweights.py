# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import print_function
import os,sys
import torch
import numpy as np
import sparseconvnet as scn

import torchvision
from torchvision.models import vgg as vgg

# Use the GPU if there is one and sparseconvnet can use it, otherwise CPU
use_cuda = torch.cuda.is_available() and scn.SCN.is_cuda_build()
use_cuda = False
device = 'cuda:0' if use_cuda else 'cpu'
if use_cuda:
    print("Using CUDA.")
else:
    print("Not using CUDA.")

vgg11file = 'vgg11-bbd30ac9.pth'
if not os.path.exists(vgg11file):
    os.system('wget %s'%(vgg.model_urls['vgg11']))
model_data = torch.load(vgg11file)
#print(model_data)
for name in model_data:
    print(name,model_data[name].shape)

# based on torchvision.vgg11 (configuration A)
vgg_layers = [ ['C',64,],
               ['MP',3,2],
               ['C',128],
               ['MP',3,2],
               ['C',256],
               ['C',32],
               ['MP',3,2] ]
               #['MP',3,2],               
               #['C',512],
               #['C',512],
               #['MP',3,2] ]
default_layers = [['C', 8],
                  ['C', 8],
                  ['MP', 3, 2],
                  ['C', 16],
                  ['C', 16],
                  ['MP', 3, 2],
                  ['C', 24],
                  ['C', 24],
                  ['MP', 3, 2]]
               
model = scn.Sequential().add(
    scn.SparseVggNet(2, 1, vgg_layers )
#).add(
#scn.SubmanifoldConvolution(2, 24, 32, 3, False)
#).add(
#    scn.BatchNormReLU(32)
).add(
    scn.SparseToDense(2, 32)
).to(device)

print(model)
par_dict = {}
for par in model.named_parameters():
    print("--------------------------------------------")
    print("setting [",par[0],par[1].shape,"]")
    if "weight" in par[0] and len(par[0].split("."))==3 and len(par[1].shape)==4:
        layerid1 = int(par[0].split(".")[0])
        layerid2 = int(par[0].split(".")[1])
        vgglayername = "features.%d.weight"%(layerid2)
        try:
            np_vggweight = model_data[vgglayername].numpy()
        except:
            continue
        vggweight = np.transpose( np_vggweight, (2,3,1,0) )
        vggweight = torch.from_numpy( vggweight.reshape( (9,1,vggweight.shape[2],vggweight.shape[3]) ) ).to(device)
        print(par[0],par[1].shape,'-->',vgglayername,vggweight.shape)
        par_dict[par[0]] = vggweight[:,:,:par[1].shape[2],:par[1].shape[3]]
    else:
        par_dict[par[0]] = par[1]
        #print(par[1])

    if "weight" in par[0] and len(par[0].split("."))==3 and len(par[1].shape)==1:
        par_dict[par[0].replace("weight","running_var")] = torch.ones( (par[1].shape[0]) )
    if "bias" in par[0] and len(par[0].split("."))==3 and len(par[1].shape)==1:
        par_dict[par[0].replace("bias","running_mean")] = torch.zeros( (par[1].shape[0]) )
#print(par_dict)
#par_dict["2.running_mean"] = torch.zeros( (32) )
#par_dict["2.running_var"]  = torch.ones( (32) )
model.load_state_dict(par_dict)
    
# output will be 10x10
inputSpatialSize = model.input_spatial_size(torch.LongTensor([10, 10]))
input_layer = scn.InputLayer(2, inputSpatialSize)
bl_input_layer = scn.BLInputLayer(2, inputSpatialSize)

msgs = [[" X   X  XXX  X    X    XX     X       X   XX   XXX   X    XXX   ",
         " X   X  X    X    X   X  X    X       X  X  X  X  X  X    X  X  ",
         " XXXXX  XX   X    X   X  X    X   X   X  X  X  XXX   X    X   X ",
         " X   X  X    X    X   X  X     X X X X   X  X  X  X  X    X  X  ",
         " X   X  XXX  XXX  XXX  XX       X   X     XX   X  X  XXX  XXX   "],

        [" XXX              XXXXX      x   x     x  xxxxx  xxx ",
         " X  X  X   XXX       X       x   x x   x  x     x  x ",
         " XXX                X        x   xxxx  x  xxxx   xxx ",
         " X     X   XXX       X       x     x   x      x    x ",
         " X     X          XXXX   x   x     x   x  xxxx     x ",]]


# Create Nx3 and Nx1 vectors to encode the messages above using InputLayer:
locations = []
features = []
for batchIdx, msg in enumerate(msgs):
    for y, line in enumerate(msg):
        for x, c in enumerate(line):
            if c == 'X':
                locations.append([y, x, batchIdx])
                features.append([1])
locations = torch.LongTensor(locations)
features = torch.FloatTensor(features).to(device)

input = input_layer([locations,features])
print('Input SparseConvNetTensor:', input.features.shape)
output = model(input)

# Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
# feature planes, and 10x10 is the spatial size of the output.
print('Output SparseConvNetTensor:', output)

# Alternatively:
# Create Nx3 and Nx1 vectors to encode the messages above using BLInputLayer:
batch=[]
for batchIdx, msg in enumerate(msgs):
    l,f=[],[]
    for y, line in enumerate(msg):
        for x, c in enumerate(line):
            if c == 'X':
                l.append([y, x])  #Locations
                f.append([1])     #Features
    batch.append([torch.LongTensor(l),torch.FloatTensor(f)])
batch=scn.prepare_BLInput(batch)
batch[1]=batch[1].to(device)
    
input = bl_input_layer(batch)
print('Input SparseConvNetTensor:', input.features.shape)
output = model(input)

# Output is 2x32x10x10: our minibatch has 2 samples, the network has 32 output
# feature planes, and 10x10 is the spatial size of the output.
print('Output SparseConvNetTensor:', output)


