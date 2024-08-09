import torch
print("torch version: ", torch.__version__)
print("cuda availability: ", torch.cuda.is_available())
print("device count: ", torch.cuda.device_count())
print("cuda version: ",torch.version.cuda)