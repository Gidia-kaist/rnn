import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms
from tqdm import tqdm

import pandas as pd
from datetime import datetime
from time import time as t

from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights
from bindsnet.analysis.plotting import plot_spikes, plot_weights
from bindsnet.shared_preference import SharedPreference
global seed_weight
global seed_weight_bf
global seed_weight_now

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=1600)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--n_epochs", type=int, default=5)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--update_steps", type=int, default=100)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=200)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--nu_single", type=float, default=1e-3)
parser.add_argument("--nu_pair", type=float, default=1e-2)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True , gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_workers = args.n_workers
update_steps = args.update_steps
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot
gpu = args.gpu
nu_single = args.nu_single
nu_pair = args.nu_pair

update_interval = update_steps * batch_size
now = datetime.now()


filter_mask = True



# Sets up Gpu use
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    print('Using GPU!')
else:
    torch.manual_seed(seed)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    nu=(nu_single, nu_pair),
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

# Directs network to GPU
if gpu:
    network.to("cuda")
    print('Connected "Network" to CUDA')

# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons)
proportions = torch.zeros(n_neurons, n_classes)
rates = torch.zeros(n_neurons, n_classes)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

temp = []
count = 0
mark_true = torch.ones(1600)
mark_false = torch.zeros(1600)
check_list = []
mark_save = False

spike_record = torch.zeros(update_interval, time, n_neurons)

# Train the network.
print("\nBegin training.\n")
start = t()
csv_datas = []
csv_datas_conn = []
for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    for step, batch in enumerate(tqdm(dataloader)):
        # Get next input sample.
        inpts = {"X": batch["encoded_image"]}
        if gpu:
            inpts = {k: v.cuda() for k, v in inpts.items()}
        #print(network.connections[("X", "Ae")].w.size())
        #network.connections[("Ai", "Ae")].w[:, :] = 0
        #print(network.connections[("Ai", "Ae")].w)

        if step % update_steps == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )
            #print("\n")
            #print(assignments.size())
            #print(label_tensor.long().size())
            #print(all_activity_pred.size())
            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )
            if accuracy["all"][-1] > 80:
               SharedPreference.set_filter_mask(True)

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )
            csv_datas.append([step, '%.2f' % (accuracy["all"][-1]), '%.2f' % (np.mean(accuracy["all"])),
                              '%.2f' % (np.max(accuracy["all"]))])
            df = pd.DataFrame(csv_datas, columns=['step', 'LAST', 'AVG', 'BEST'])
            df.to_csv('/home/gidia/anaconda3/envs/myspace/examples/mnist/outputs/test_' + str(now.year) + '_' + str(
                now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) + '.csv',
                      index=False)

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )
            #print(all_activity_pred.size())
            if count == 1:
                temp = assignments
                #print(temp.size())
            elif count > 1:
                check_list = torch.where(temp == assignments, mark_true, mark_false)
                temp = assignments

                for i in range(1600):
                    if check_list[i] == 1:
                        if abs(max(network.connections[("X", "Ae")].w[:, i]) - min(network.connections[("X", "Ae")].w[:, i])) > 0.35:
                            # set connectivity of inpts to exc as ZERO(FALSE)
                            SharedPreference.set_boolean_mask(SharedPreference, i, 0)
                            # set connectivity of inh to exc as ZERO(FALSE)
                            network.connections[("Ai", "Ae")].w[:, i] = 0
                            mark_save = True
                            #print(network.connections[("X", "Ae")].w)
                            #print("Some connectivities have changed!")
                        else:
                            SharedPreference.set_boolean_mask(SharedPreference, i, 1)


                #if mark_save == True:

                    #df = pd.DataFrame(csv_datas_conn, columns=['Connectivity'])
                    #df.to_csv('/home/gidia/anaconda3/envs/myspace/examples/mnist/outputs/conn_' + str(count) + "_" + str(now.year) + '_'
                    #          + str( now.month) + '_' + str(now.day) + '_' + str(now.hour) + '_' + str(now.minute) + '.csv', index=False)
                    #mark_save = False
            count += 1
            print(assignments[0:80])
            #print(SharedPreference.get_boolean_mask(SharedPreference))
            labels = []
        labels.extend(batch["label"].tolist())




        # Run the network on the input.
        network.run(inpts=inpts, time=time, input_time_dim=1)

        # Add to spikes recording.
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        spike_record[
        (step * batch_size)
        % update_interval: (step * batch_size % update_interval)
                           + s.size(0)
        ] = s

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # for l in spikes:
        # print(l, spikes[l].get("s").sum((0, 2)))

        # Optionally plot various simulation information.
        if plot:
            if step % update_steps == 0 and step > 0:
                # image = batch["image"].view(28, 28)
                # inpt = inpts["X"].view(time, 784).sum(0).view(28, 28)
                #count += 1
                input_exc_weights = network.connections[("X", "Ae")].w
                square_weights = get_square_weights(
                    input_exc_weights.view(784, n_neurons), n_sqrt, 28
                )
                # square_assignments = get_square_assignments(assignments, n_sqrt)
                # spikes_ = {
                #    layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
                # }
                # voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
                #
                # inpt_axes, inpt_ims = plot_input(
                #     image, inpt, label=labels[step], axes=inpt_axes, ims=inpt_ims
                # )
                # spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
                weights_im = plot_weights(square_weights, im=weights_im, count=count)
                # assigns_im = plot_assignments(square_assignments, im=assigns_im)
                # perf_ax = plot_performance(accuracy, ax=perf_ax)
                # voltage_ims, voltage_axes = plot_voltages(
                #     voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
                # )

                plt.pause(1e-8)

        '''When weight is large enough to disconnect.
        if accuracy["all"][-1] > 80:
            for i in range(784):
                for j in range(1600):
                    if network.connections[("X", "Ae")].w[i, j] > 0.25:
                        
    
'''

        network.reset_()  # Reset state variables.
    print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))


print("Training complete.\n")





