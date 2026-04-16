import argparse
import json
from os import path, makedirs

parser = argparse.ArgumentParser()
parser.add_argument('--cloud_config_filename', type=str, help='File name for cloud configs')
parser.add_argument('--dev_config_filename', type=str, help='File name for device configs')

parser.add_argument('--cloud_ip', type=str, default="127.0.0.1", help='Cloud IP')
parser.add_argument('--cloud_port', type=int, default=22, help='Cloud port')
parser.add_argument('--cloud_cuda', type=str, default="cuda", help='Cloud cuda_name')
parser.add_argument('--model_name', type=str, default="conv5",help='Model name')
parser.add_argument('--loss_func_name', type=str, default="cross_entropy", help='Loss function name')
parser.add_argument('--loss_type', type=str, default="fedavg", help='fedavg, fedmax, fedprox')
parser.add_argument('--mu', type=float, default=1.0, help='fedavg, fedmax, fedprox')
parser.add_argument('--beta', type=float, default=10.0, help='fedavg, fedmax, fedprox')
parser.add_argument('--comm_rounds', type=int, default=30, help='Communication rounds')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--verbose', type=str, default='false', help='Verbosity [true, false]')

parser.add_argument('--experiment', type=int, default=1, help='Experiment number')
parser.add_argument('--run', type=int, default=1, help='Run number')
parser.add_argument('--seed', type=int, default=1, help='Seed for experiment')
parser.add_argument('--laptop_number', type=str, default="laptop_", help='Laptop number')
parser.add_argument('--data_iid', type=str, help='IID [true] or non-IID [false]')

parser.add_argument('--num_devices', type=int, default=2, help='Number of devices')
parser.add_argument('--dev_hw_types', nargs='+', default=["mc1", "rpi"], help='Hardware type for each device')
parser.add_argument('--hosts', nargs='+', default=["127.0.0.1", "127.0.0.1"], help='IPs for devices')
parser.add_argument('--ports', nargs='+', default=[22, 22], help='Ports for devices')
parser.add_argument('--cuda_names', nargs='+', default=['cpu', 'cpu'], help='Cuda_name for each device [cuda, cpu]')
parser.add_argument('--model_names', nargs='+', default=['conv5', 'conv5'], help='Model name')
parser.add_argument('--dev_local_epochs', nargs='+', default=[5, 5], help='Local epochs for each device')


args = parser.parse_args()

cloud_config_filename = args.cloud_config_filename
dev_config_filename = args.dev_config_filename

cloud_ip = args.cloud_ip
cloud_port = args.cloud_port
cloud_cuda = args.cloud_cuda
model_name = args.model_name
loss_func_name = args.loss_func_name
loss_type = args.loss_type
mu = args.mu
beta = args.beta
comm_rounds = args.comm_rounds
learning_rate = args.learning_rate
verbose = args.verbose
if verbose == "true":
    verbose = True
else:
    verbose = False

experiment = args.experiment
run = args.run
seed = args.seed
laptop_number = args.laptop_number
data_iid = args.data_iid
if data_iid == "true":
    data_iid = True
else:
    data_iid = False

num_devices = args.num_devices
dev_hw_types = args.dev_hw_types
hosts = args.hosts
ports = args.ports
cuda_names = args.cuda_names
model_names = args.model_names
dev_local_epochs = args.dev_local_epochs

config_dict = {}
makedirs("configs", exist_ok=True)

with open(path.join("configs", cloud_config_filename), 'w') as file:
    config_dict["cloud_ip"] = cloud_ip
    config_dict["cloud_port"] = cloud_port
    config_dict["cloud_cuda_name"] = cloud_cuda
    config_dict["model_name"] = model_name
    config_dict["loss_func_name"] = loss_func_name
    config_dict["loss_type"] = loss_type
    config_dict["mu"] = mu
    config_dict["beta"] = beta
    config_dict["comm_rounds"] = comm_rounds
    config_dict["learning_rate"] = learning_rate
    config_dict["verbose"] = verbose

    config_dict["experiment"] = cloud_config_filename.split("_")[2]
    config_dict["run"] = cloud_config_filename.split("_")[3].split('.')[0]
    config_dict["seed"] = seed
    config_dict["laptop_number"] = laptop_number
    config_dict["data_iid"] = data_iid

    config_dict["num_devices"] = num_devices

    json.dump(config_dict, file, indent=2)

aux = []
for k in dev_hw_types[0].split(" "):
    aux.append(k)
dev_hw_types = aux

aux = []
for k in hosts[0].split(" "):
    aux.append(k)
hosts = aux

aux = []
for k in ports[0].split(" "):
    aux.append(int(k))
ports = aux

aux = []
for k in cuda_names[0].split(" "):
    aux.append(k)
cuda_names = aux

aux = []
for k in model_names[0].split(" "):
    aux.append(k)
model_names = aux

aux = []
for k in dev_local_epochs[0].split(" "):
    aux.append(int(k))
dev_local_epochs = aux

config_dict = {}
with open(path.join("configs", dev_config_filename), 'w') as file:
    config_dict["num_devices"] = num_devices
    for dev_idx in range(num_devices):
        dev_dict = {}
        dev_dict["hw_type"] = dev_hw_types[dev_idx]
        dev_dict["host"] = hosts[dev_idx]
        dev_dict["port"] = ports[dev_idx]
        dev_dict["cuda_name"] = cuda_names[dev_idx]
        dev_dict["model_name"] = model_names[dev_idx]
        dev_dict["local_epochs"] = dev_local_epochs[dev_idx]
        config_dict[f"dev{dev_idx + 1}"] = dev_dict
    json.dump(config_dict, file, indent=2)
