#!/bin/bash

# This script is used to generate commands to run cloud.py and device.py on edge devices
# Change the parameters below and run "bash run.bash" on terminal
# It will also run "generate_configs.py" based on the given parameters

############### TODO CHANGE
cloud_ip="172.29.204.128" #Change every time when new VPN is connected

model_name="conv5small"
loss_func_name="cross_entropy"
loss_type="fedmax" # change to fedmax or fedprox
mu=1.1 # For FedProx
beta=10.0 # For FedMAX
learning_rate=0.01

declare -a experiment_configs=(
# experiment | run | data_iid
  "18 1 false " # Experiment 6, Run 1, Non-IID
)

declare -a devices_configs=(
# hw_type | host | port | cuda_name | local_epochs
  "rpi sld-rpi-13.ece.utexas.edu 9090 cpu 2"
  "mc1 sld-mc1-13.ece.utexas.edu 9090 cpu 2"
)
############### TODO END CHANGE

verbose='false'
laptop_number='laptop_1'
cloud_port="9090"
cloud_cuda="cpu"
comm_rounds=30
num_devices=2
for experiment_config in "${experiment_configs[@]}"
do
  read -a exp_config <<< "$experiment_config"
  experiment="${exp_config[0]}"
  run="${exp_config[1]}"
  # Seeds for all runs are predetermined.
  if [ "$run" -eq 1 ]; then
    seed=2
  elif [ "$run" -eq 2 ]; then
    seed=14
  elif [ "$run" -eq 3 ]; then
    seed=26
  else
    echo "Invalid run value. Please specify 1, 2, or 3."
    exit 1
  fi
  data_iid="${exp_config[2]}"

  declare -a dev_hw_types
  declare -a ips
  declare -a ports
  declare -a model_names
  declare -a dev_local_epochs
  for devs_configs in "${devices_configs[@]}"
  do
    read -a dev_config <<< "$devs_configs"
    hw_type="${dev_config[0]}"
    dev_hw_types+=("$hw_type")

    host="${dev_config[1]}"
    hosts+=("$host")

    port="${dev_config[2]}"
    ports+=("$port")

    cuda_name="${dev_config[3]}"
    cuda_names+=("$cuda_name")

    model_names+=("$model_name")
    local_epochs="${dev_config[4]}"
    dev_local_epochs+=("$local_epochs")
  done

  cloud_config_filename="cloud_cfg_exp${experiment}_run${run}.json"
  dev_config_filename="dev_cfg_exp${experiment}_run${run}.json"

  python generate_configs.py \
  --cloud_config_filename "${cloud_config_filename}" \
  --dev_config_filename "${dev_config_filename}" \
  --cloud_ip "${cloud_ip}" \
  --cloud_port "${cloud_port}" \
  --cloud_cuda "${cloud_cuda}" \
  --model_name "${model_name}" \
  --loss_func_name "${loss_func_name}" \
  --loss_type "${loss_type}" \
  --mu "${mu}" \
  --beta "${beta}" \
  --comm_rounds "${comm_rounds}" \
  --learning_rate "${learning_rate}" \
  --verbose "${verbose}" \
  --experiment "${experiment}" \
  --run "${run}" \
  --seed "${seed}" \
  --laptop_number "${laptop_number}" \
  --data_iid "${data_iid}" \
  --num_devices "${num_devices}" \
  --dev_hw_types "${dev_hw_types[*]}" \
  --hosts "${hosts[*]}" \
  --ports "${ports[*]}" \
  --cuda_names "${cuda_names[*]}" \
  --model_names "${model_names[*]}" \
  --dev_local_epochs "${dev_local_epochs[*]}"
done
