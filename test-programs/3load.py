import torch
from torch import nn
import torch.nn.utils.prune as prune

model = torch.load('mGru_flipflop_remapping_model_r9_DNA.checkpoint')
module = model.sublayers
#print(list(module.named_parameters()))
gru = module[1].cudnn_gru
#print(gru.weight_hh_l0)
#print(list(gru.named_parameters()))
#prune.random_unstructured(gru, name="weight_hh_l0", amount=0.3)
#print(list(gru.named_parameters()))

sd = gru.state_dict()
#print(list(gru.named_parameters()))
for k in sd.keys():
   if 'weight_hh_l0' in k:
     w = sd[k]
     #sd[k] = w * (w > 0.3) 
     #w_pruned = sd[k]
     #print(w)
     #print(list(w_pruned))

#for k in sd.keys():
#   if 'weight_hh_l0' in k:
#     w = sd[k]
     #print(list(w))

weight = w.tolist()
for i in range(len(weight)):
   print(*weight[i],sep="\n")

#print(gru.weight_hh_l0) # after pruning the modified weight is an attribute
#w = gru.weight_hh_l0
#weight = w.tolist()
#for i in range(len(weight)):
#   print(*weight[i],sep="\n" 
#torch.save(model.state_dict(), './model_checkpoint_00000_pruned.checkpoint')
torch.save(model, './mGru_flipflop_remapping_model_r9_DNA_pruned.checkpoint')
model = torch.load('mGru_flipflop_remapping_model_r9_DNA_pruned.checkpoint')
module = model.sublayers
gru = module[2].cudnn_gru
sd = gru.state_dict()
for k in sd.keys():
   if 'weight_hh_l0' in k:
     w = sd[k]
     #print(list(w))
