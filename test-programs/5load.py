import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.quantization

model = torch.load('mGru_flipflop_remapping_model_r9_DNA.checkpoint')
#print(list(model.named_parameters()))

qconfig = torch.quantization.get_default_qconfig('fbgemm')
#print(torch.backends.quantized.supported_engines) # Prints the quantized backends that are supported
# Set the backend to what is needed. This needs to be consistent with the option you used to select the qconfig
torch.backends.quantized.engine='fbgemm'
qmodel = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

torch.save(qmodel, './mGru_flipflop_remapping_model_r9_DNA_quantized.checkpoint')
model = torch.load('mGru_flipflop_remapping_model_r9_DNA_quantized.checkpoint')
#print(list(model.named_parameters()))
for name, param in model.named_parameters():
    if name in ['sublayers.1.layer.cudnn_gru.weight_hh_l0']:
        print (param)
        weight = param.tolist()
        #for i in range(len(weight)):
           #print(*weight[i],sep="\n") 

