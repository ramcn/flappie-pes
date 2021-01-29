import torch
from torch import nn
import torch.nn.utils.prune as prune

model = torch.load('mGru_flipflop_remapping_model_r9_DNA.checkpoint')
#print(list(model.named_parameters()))
for name, param in model.named_parameters():
    if name in ['sublayers.5.layer.cudnn_gru.weight_hh_l0']:
      weight = param.tolist()
      for i in range(len(weight)):
        print(*weight[i],sep="\n")


