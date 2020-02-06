import os

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


#os.system("python3.6 batch_eth_mnist.py --error_range 0.10")
#os.system("python3.6 batch_eth_mnist.py --error_range 0.15")
#os.system("python3.6 batch_eth_mnist.py --error_range 0.20")
#os.system("python3.6 batch_eth_mnist.py --error_range 0.25")
os.system("python3.6 batch_eth_mnist.py --error_range 0.30")
os.system("python3.6 batch_eth_mnist.py --error_range 0.35")
os.system("python3.6 batch_eth_mnist.py --error_range 0.40")
# os.system("python3.6 batch_eth_mnist.py --connectivity 0.6 --prune 50")
# os.system("python3.6 batch_eth_mnist.py --connectivity 0.6 --prune 50")

# os.system("python3.6 batch_eth_mnist.py --connectivity 0.5 --prune 10")
# os.system("python3.6 batch_eth_mnist.py --connectivity 0.5 --prune 25")