import torch

shape= torch.Size((3,3))
x = torch.cuda.FloatTensor(shape)
error_rand = (-2) * torch.rand(shape, out=x) + 1
error_mask = error_rand * 0.5

print(error_rand)
print(error_mask)