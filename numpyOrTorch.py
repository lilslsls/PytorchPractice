import torch
import numpy as np

numpy_data = np.arange(6).reshape(2,3)
torch_data=torch.from_numpy(numpy_data)
tensor2numpy=torch_data.numpy()
print(numpy_data)
print(torch_data)
print(tensor2numpy)