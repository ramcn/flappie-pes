import torch
model = torch.load('mLstm_flipflop_model_r103_DNA.checkpoint', map_location=lambda storage, loc: storage)

print (model)



