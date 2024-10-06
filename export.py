import torch
from modeling_minicpmv import MiniCPMV


model = MiniCPMV.from_pretrained("./", torch_dtype=torch.float32)
model.export()
