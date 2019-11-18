import torch

from bindsnet.shared_preference import SharedPreference

a = torch.ones(784, 1600)
b = SharedPreference.get_copy(SharedPreference)
print(b[:3, :3])
for i in range(2):
    SharedPreference.set_copy(SharedPreference, target=a, col=i)

b = SharedPreference.get_copy(SharedPreference)
print(b[:3, :3])

