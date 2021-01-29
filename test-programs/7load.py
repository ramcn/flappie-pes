import torch
from torch import nn
import torch.nn.utils.prune as prune

model = torch.load('mGru_flipflop_remapping_model_r9_DNA.checkpoint')
zeroes = 0
for name, param in model.named_parameters():
    if name in ['sublayers.1.layer.cudnn_gru.weight_hh_l0']:
        rows = param.shape[0]
        cols = param.shape[1]
        for x in range(0, rows):
            for y in range(0, cols):
              if(param.data[x,y] < 0.14 and param.data[x,y] > -0.08):
                  param.data[x,y] = 0
                  zeroes = zeroes+1
print((zeroes*100)/(rows*cols))
zeroes = 0
for name, param in model.named_parameters():
    if name in ['sublayers.5.layer.cudnn_gru.weight_hh_l0']:
        rows = param.shape[0]
        cols = param.shape[1]
        for x in range(0, rows):
            for y in range(0, cols):
              if(param.data[x,y] < 0.18 and param.data[x,y] > -0.14):
                  param.data[x,y] = 0
                  zeroes = zeroes+1
print((zeroes*100)/(rows*cols))

torch.save(model, './mGru_flipflop_remapping_model_r9_DNA_pruned.checkpoint')

