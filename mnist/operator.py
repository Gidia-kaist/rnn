
import os
import torch

from bindsnet.network import network
from mnist.shared_preference import SharedPreference


#os.system("python3.6 batch_eth_mnist.py --nu_single 1e-3 --nu_pair 1e-2 --time 500")
#os.system("python3.6 batch_eth_mnist.py --nu_single 1e-3 --nu_pair 1e-2 --time 600")
#os.system("python3.6 batch_eth_mnist.py --nu_single 1e-3 --nu_pair 1e-2 --time 700")
#print(network.connections[("X", "Ae")].w)
'''
instance = SharedPreference.get_filter_mask(SharedPreference)
print(instance)
SharedPreference.set_filter_mask(SharedPreference, False)
instance = SharedPreference.get_filter_mask(SharedPreference)
print(instance)'''