import torch
import torch.utils.data as Data
# import matplotlib.pyplot as plt
BATCH_SIZE = 8
m = torch.Tensor(BATCH_SIZE,2,3)
print(m)
m = m.reshape([-1,6])
print(m)