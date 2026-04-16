# Introduction

Read carefully this README file, starting from the introduction and then continuing with **Parts 1 - 7** in that order. 

A few notes:
* We do not provide any way to run multiple experiments in one go. The `config_1.bash` file is able to create multiple
experiments, but you have to run them sequentially. To be more concise, `config_1.bash` creates configurations for two 
experiments. You can only run one experiment at a time. After you run `config_1.bash` you obtain one cloud and one dev 
config for each defined experiment. When you run the devices, you start by putting the `--exp` argument and in the 
cloud you put the `cloud_cfg` and `dev_cfg`. Hence, you only run one experiment at a time. But, you can generate 
multiple experiments from the same `config_1.bash` file.
* In the FL framework, `dev_idx=0` is always the RPi and `dev_idx=1` is always the MC1. As such, `dev_idx` has nothing 
to do with your actual RPi and MC1 numbers, for example `4` and `23` that correspond to `sld-rpi-04` and `sld-mc1-23`,
respectively. The `dev_idx` is the device index when training in FL and is always the same, `0` for RPi and `1` for MC1. 
The only place you put the `4` and `23` is in the `config_1.bash`. Just make sure even there to keep the same order of 
the RPi and MC1. In other words it matters which is first and which is second in the `config_1.bash` file as well.

## Rename the project folder
From `ece_361e_fl-v1` rename the project folder to `ece_361e_fl`

Whenever you get a new version of the project, make sure to:
1. Rename on your laptop the project folder
2. Transfer files to the edge devices as in **Part 4.1** shown below.

## Configurations and experiments
The experiment configuration generation file `config_1.bash` is meant to generate as many experiments as you would like to
define. Inside this file, you have to update the device numbers to correspond to your assigned devices.
```bash
# hw_type | host | port | cuda_name | local_epochs
"rpi sld-rpi-13.ece.utexas.edu 9090 cpu 1" # Change number of device (Keep RPi the first one)
"mc1 sld-mc1-13.ece.utexas.edu 9090 cpu 1" # Change number of device (Keep the MC1 the first one)
```
**Example:** If you execute the `config_1.bash` as is, you will generate configurations for two experiments, namely
experiment 1 and experiment 2. For each experiment you generate two configurations: one for the cloud and one for the 
edge devices. You run each experiment individually, one at a time. For example, to run experiment 1, 
you have to run for each device with the argument `--exp 1` (check **Part 5** below) and for the cloud with the 
`--cloud_cfg` and `--dev_cfg` set as the experiment 1 cloud and device configurations as shown in **Part 6** below.

**IMPORTANT:** For all results you show in the **M1**. **M2** and **M3** presentations, you have to run all 30 
communication rounds for each experiment. However, for your faster exploration you can prematurely end some of the experiments
using `CTRL+C`. 

**Example:** You already made the **M1** presentation and want to explore more. If you are looking to 
consume the least average energy per communication round, you can end the experiment after 5 communication rounds, 
since that already gives you an idea of the average energy consumption per communication round. 

**Note:** Each experiment runs around 2 hours so make sure to start early.

## 1. Prepare environment on your laptop
```bash
conda create -n ece_361e_fl python==3.8
conda activate ece_361e_fl
pip3 install torch torchvision torchaudio
pip3 install paramiko scp tqdm matplotlib pandas
```
**Note:** The RPi and MC1 devices have everything already installed.

**IMPORTANT:** Make sure you are connected to VPN and **do the following test with both RPi and MC1 devices**:
1. Connect to UT VPN, find your laptop's IP address.
2. SSH into the edge device, exit, then SSH again. You will see something like
```bash
$ ssh student@sld-rpi-13.ece.utexas.edu
student@sld-rpi-13.ece.utexas.edu password:
Linux sld-rpi-13 6.1.21-v8+ #1642 SMP PREEMPT Mon Apr  3 17:24:16 BST 2023 aarch64

The programs included with the Debian GNU/Linux system are free software;
the exact distribution terms for each program are described in the
individual files in /usr/share/doc/*/copyright.

Debian GNU/Linux comes with ABSOLUTELY NO WARRANTY, to the extent
permitted by applicable law.
Last login: Sun Mar 31 13:35:36 2024 from 172.29.235.62 
                                               ^------------------This is your laptop IP address
```
3. From the edge device, try to SSH back into your laptop using `ssh <your_laptop_username>@<laptop_ip>`
   a. For Windows run `whoami` to see your username
   b. For MacBook run `who am i` to see your username
4. If **you cannot SSH back into your laptop** you have to enable incoming SSH connections.
   a. For Windows use [this link](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=gui).
   b. For MacBook use [this link](https://support.apple.com/lt-lt/guide/mac-help/mchlp1066/mac).
   c. You can also find other online solutions to enable incoming SSH connections on your laptop.
5. Run again steps 2 and 3.

## 2. Change paths
In `utils/general_utils.py` the function `get_hw_info` contains passwords for devices, password and username for your 
laptop and path for the project files on your laptop. Change them accordingly (check the TODOs).
```bash
def get_hw_info(hw_type, device_number=None):
    if hw_type == 'rpi':
        password = "password"  # TODO change here password
    elif hw_type == 'mc1':
        password = "password"  # TODO change here password
    elif hw_type.split('_')[0] == 'laptop': 
        password = "password" # TODO change here password
        username = "username" # TODO change here username
        local_path = "/home/username/ece_361e_fl" # TODO change here your path; make sure it ends with ece_361e_fl
```
## 3. Connect to the devices in two separate terminals
**Note:** Change the RPi and MC1 numbers accordingly
```bash
ssh student@sld-rpi-13.ece.utexas.edu
ssh student@sld-mc1-13.ece.utexas.edu
```

## 4. Transfer files
### 4.1. First time transfer everything
```bash
scp -r /your_path/ece_361e_fl student@sld-rpi-13.ece.utexas.edu:/home/student
scp -r /your_path/ece_361e_fl student@sld-mc1-13.ece.utexas.edu:/home/student
```
### 4.2. Update code parts as you need
Transfer python files from project root
```bash
scp -r /your_path/ece_361e_fl/*.py student@sld-rpi-13.ece.utexas.edu:/home/student/ece_361e_fl
scp -r /your_path/ece_361e_fl/*.py student@sld-mc1-13.ece.utexas.edu:/home/student/ece_361e_fl
```
Transfer python files from `utils` folder
```bash
scp -r /your_path/ece_361e_fl/utils/*.py student@sld-rpi-13.ece.utexas.edu:/home/student/ece_361e_fl/utils/
scp -r /your_path/ece_361e_fl/utils/*.py student@sld-mc1-13.ece.utexas.edu:/home/student/ece_361e_fl/utils/
```
Transfer python files from `models` folder
```bash
scp -r /your_path/ece_361e_fl/models/*.py student@sld-rpi-13.ece.utexas.edu:/home/student/ece_361e_fl/models/
scp -r /your_path/ece_361e_fl/models/*.py student@sld-mc1-13.ece.utexas.edu:/home/student/ece_361e_fl/models/
```

## 5. Run on Devices
Raspberry Pi 3B+
```bash
python device.py --host sld-rpi-13.ece.utexas.edu --port 9090 --device_type rpi --dev_idx 0 --r 1 --seed 2 --exp 1 
```
Odroid MC1
```bash
python device.py --host sld-mc1-13.ece.utexas.edu --port 9090 --device_type mc1 --dev_idx 1 --r 1 --seed 2 --exp 1  
```
Do go to the next step until you see for each device the following two lines:
```bash
[+] Device 0 is listening on host:port sld-rpi-13.ece.utexas.edu:9090
[+] Recording power and temperature...
```
**Note:** If you see warnings when executing on the device you can safely ignore them. The code executes just fine.

For **M1** and **M2** **do not change** the **run** `r` or the **seed** `seed`.

**Only for M3 you have to run your final solution three times, each run with a different preset seed as follows:**

| **Run** | Seed |
|---------|------|
| **1**   | 2    |
| **2**   | 14   |
| **3**   | 26   |


## 6. Run on the Cloud
Generate the configs for experiments 1 and 2.
```bash
bash config_1.bash
```
Then run the Cloud and start doing Federated Learning!
```bash
python cloud.py --cloud_cfg configs/cloud_cfg_exp1_run1.json --dev_cfg configs/dev_cfg_exp1_run1.json
```

## 7. Create plots and performance metrics after experiments finish
Use the `generate_figs.py` to create plots under a new folder `figs/` and print performance metrics.

**Example 1:** Show only experiment 1 with run 1 for accuracy and loss curves named "IID" (as the experiment label 
indicates below). For the same experiment also print time, power, energy and communication related performance metrics.

**Note 1:** For power, energy and communication, at least one of them needs to be enabled to create a new figure under `figs/`
called `exp1_run1_pow_energy_comm.png`.

**Note 2:** The `plot_title` is the plot title for both the accuracy and the loss plots.
```bash
python generate_figs.py --exps 1 --exp_labels IID --runs 1 --plot_title FedAvg --accuracy --time --power --energy --communication
```

**Example 2:** In case you ran two or more experiments and want to compare their performance in terms of accuracy and 
loss on the same plot, you can run as below and create an accuracy plot with two curves: one for experiment 1 with the 
name "IID" and the other curve for experiment 2 with the name "Non-IID".
```bash
python generate_figs.py --exps 1 2 --exp_labels IID Non-IID --runs 1 1 --plot_title FedAvg --accuracy --time --power --energy --communication
```

**Example 3 (for M3):** In case you have multiple runs of the same experiment, you can just increase the `runs` number and all 
plots and performance metrics will be averaged for the same experiment over all the runs.
```bash
python generate_figs.py --exps 1 --exp_labels IID Non-IID --runs 3 --plot_title FedAvg --accuracy --time --power --energy --communication
```

# How to change the framework?
## A. Change the model
In `models/get_model.py` add to the function `get_model(model_name)` your new model name string and return the model 
class instance (don't forget to also import the model). Add the new model class definition file under the `models` 
folder. Finally, change the config_1.bash with the new model name, and you're good to go!

## B. Change the local epoch number
```bash
# hw_type | host | port | cuda_name | local_epochs
"rpi sld-rpi-13.ece.utexas.edu 9090 cpu 1" # Change number of the device and the number of local epochs
"mc1 sld-mc1-13.ece.utexas.edu 9090 cpu 1" # Change number of the device and the number of local epochs
```

## C. Change the learning rate or loss type
In the `config_1.bash` file change the following lines according to your exploration plans:
```bash
loss_type="fedavg" # change to fedmax or fedprox
mu=1.0 # For FedProx
beta=10.0 # For FedMAX
learning_rate=0.01 # change to any value you want to explore
```
Make sure to change accordingly the `mu` and `beta` parameters for FedProx and FedMAX, respectively. For the implementation
of FedMAX and FedProx, check the `utils/train_test.py` code.

**Note:** For the implementation of FedMAX note differences in the `test` function from `utils/train_test.py`, in the
`get_model` function from `models/get_model.py` and in the model architecture definitions from `models/conv5.py`. If you
want to use FedMAX make sure to add the activations to be returned besides the usual prediction outputs of the model.

## D. Change the learning rate based on device
Hint if you want to have different learning rates on different devices. You can follow the same code that sends 
the local epochs on the devices. You start from the config, then you add properly the learning rates in the 
`generate_configs.py`, then you follow up in the `cloud.py` to get for each device its own learning rate (similar 
to the local epoch) and compose the message you send to the device properly. Finally, you make sure on the device 
that you get the proper learning rate from the cloud as a message and that should be it. For sanity check, in 
`utils/train_test.py` you can print in the `local_training` function the learning rate for each device to make sure 
the learning rates for each device are correctly generated in the configs and transferred to the devices.