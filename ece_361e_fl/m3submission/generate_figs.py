import os
import pandas as pd
import matplotlib.pyplot as plt
import paramiko
import socket
from scp import SCPClient
from utils.general_utils import get_hw_info
import json
import argparse
import numpy as np
import csv

def fetch_file(hw_type, device_number, filename, local_path):
    password, username, remote_path, host = get_hw_info(hw_type, device_number)

    # Ensure the local directory exists
    os.makedirs(local_path, exist_ok=True)
    remote_file = f"{remote_path}/{filename}"
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        print(f"Connecting to {host}...")
        ssh.connect(hostname=host, username=username, password=password)
    except socket.gaierror:
        print(f"Unable to connect to {host}. Skipping file fetch.")
        return -1
    scp = SCPClient(ssh.get_transport())
    scp.get(remote_file, local_path)
    scp.close()
    print(f"Successfully fetched {filename}.")
    return 1

def plot_acc_loss(experiments, runs, exp_labels, title, time=True):
    # Accuracy
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.yaxis.set_ticks([10, 30, 50, 70, 90, 100])
    plt.ylim(0, 100)
    plt.title(title)
    plt.xlabel("Communication round")
    plt.ylabel("Accuracy [%]")
    plt.grid()

    print(f"\n{title}")
    for exp, rs, label in zip(experiments, runs, exp_labels):
        mydf = []
        for r in range(1, rs+1):
            file_path = f"logs/log_exp{exp}_run{r}.csv"
            df = pd.read_csv(file_path)
            mydf.append(df)

        mylst = []
        mydict = {}
        for i in range(rs):
            for j in range(len(mydf[i]['Acc'])):
                mylst.append(mydf[i]['Acc'].iloc[j])
            mydict[i] = np.expand_dims(np.array(mylst), axis=0)
            mylst = []

        mylst = np.concatenate(tuple(mydict.values()), axis=0)
        mystd = np.std(mylst, axis=0)
        mylst = np.mean(mylst, axis=0)
        max_index = np.argmax(mylst)
        if rs > 1:
            print(f"Experiment {exp}: \n\tGlobal test accuracy [%]: {np.round(mylst[max_index], 2)} \u00B1 "
                  f"{np.round(mystd[max_index], 2)} %")
        else:
            print(f"Experiment {exp}: \n\tGlobal test accuracy [%]: {np.round(mylst[max_index], 2)} %")

        found = False
        for idx in range(len(mylst)):
            if mylst[idx] >= 90.0:
                # Check if all subsequent rounds meet the criteria
                if all(i >= 90.0 for i in mylst[idx:]):
                    print(f"\tConvergence time [#round]: {idx} communication rounds")
                    found = True
                    break

        if time and found:
            total_time = 0.0
            with open(file_path, mode='r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                # Iterate over each row in the CSV file
                for row in csvreader:
                    comm_round = int(row['CommRound'])
                    # Check if the current round is within the target range
                    if comm_round <= idx:
                        time = float(row['Time'])
                        total_time += time
                    else:
                        # Break the loop if the comm_round exceeds the target
                        break
            # Print the total time
            print(f"\tTotal wall clock time to reach 90.00% [s]: {total_time:,.2f} seconds")

        if not found:
            print("\tNo round found where all subsequent rounds have accuracy \u2265 90.00%.")
        plt.plot(range(len(mylst)), mylst, label=label, linewidth=4)
        mydf = []
        for r in range(1, rs + 1):
            try:
                file_path = f"logs/log_exp{exp}_run{r}.csv"
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist. Skipping...")
                    continue
                df = pd.read_csv(file_path)
                mydf.append(df)
            except BaseException as e:
                print(f"Missing log_exp{exp}_run{r}.csv and avoiding it with label {label}")
                continue
        mylst = []
        mydict = {}
        for i in range(rs):
            for j in range(len(mydf[i]['Time'])):
                mylst.append(mydf[i]['Time'].iloc[j])
            mydict[i] = np.expand_dims(np.array(mylst), axis=0)
            mylst = []
        mylst = np.concatenate(tuple(mydict.values()), axis=0)
        mystd = np.std(mylst, axis=0)
        mylst = np.mean(mylst, axis=0)
        if rs > 1:
            print(f"\tAverage time per communication round [s]: {np.round(mylst[max_index], 2)} \u00B1 "
                  f"{np.round(mystd[max_index], 2)} seconds")
        else:
            print(f"\tAverage time per communication round [s]: {np.round(mylst[max_index], 2)} seconds")
        # Total Wall Clock time
        with open('logs/time.csv', mode='r') as csvfile:
            csvreader = csv.reader(csvfile)
            experiment_data = {}
            # Iterate over each row in the CSV
            for row in csvreader:
                # Extract experiment name and time
                experiment, run, time = row
                time = float(time)  # Convert time from string to float

                # Check if the experiment is already in the dictionary
                if experiment in experiment_data:
                    experiment_data[experiment]['total_time'] += time
                    experiment_data[experiment]['run_count'] += 1
                else:
                    experiment_data[experiment] = {'total_time': time, 'run_count': 1}

        # Calculate and print the average time for each experiment
        for experiment, data in experiment_data.items():
            if experiment == f"exp{exp}":
                average_time = data['total_time'] / data['run_count']
                print(f"\tTotal wall clock time [s]: {average_time:,.2f} seconds\n")
    plt.legend()
    plt.savefig(f'figs/acc_{title}.png', bbox_inches='tight', dpi=300)

    # Loss
    plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.xlabel("Communication round")
    plt.ylabel("Loss")
    plt.grid()
    for exp, rs, label in zip(experiments, runs, exp_labels):
        mydf = []
        for r in range(1, rs + 1):
            try:
                file_path = f"logs/log_exp{exp}_run{r}.csv"
                if not os.path.exists(file_path):
                    print(f"File {file_path} does not exist. Skipping...")
                    continue
                df = pd.read_csv(file_path)
                mydf.append(df)
            except BaseException as e:
                print(f"Missing log_exp{exp}_run{r}.csv and avoiding it with label {label}")
                continue
        mylst = []
        mydict = {}
        for i in range(rs):
            for j in range(len(mydf[i]['Loss'])):
                mylst.append(mydf[i]['Loss'].iloc[j])
            mydict[i] = np.expand_dims(np.array(mylst), axis=0)
            mylst = []
        mylst = np.concatenate(tuple(mydict.values()), axis=0)
        mylst = np.mean(mylst, axis=0)
        plt.semilogy(range(len(mylst)), mylst, label=label, linewidth=4)
    plt.legend()
    plt.savefig(f'figs/loss_{title}.png', bbox_inches='tight', dpi=300)


def plot_power_energy_communication(exp, run, power=True, energy=True, communication=True):
    devices = [0, 1]  # Device IDs: 0 for RPi, 1 for MC1
    avg_powers = []
    avg_energies = []
    avg_data_communicated = []

    for device in devices:
        # Construct file paths
        comm_log_path = f"logs/device_logs/dev{device}_communication_log_exp{exp}_run{run}.csv"
        pow_temp_log_path = f"logs/device_logs/dev{device}_pow_temp_log_exp{exp}_run{run}.csv"

        # Read the data
        comm_df = pd.read_csv(comm_log_path)
        pow_temp_df = pd.read_csv(pow_temp_log_path)

        # Filter for actual communication rounds and calculate average communicated data
        comm_rounds = comm_df[comm_df['CommRound'] > 0]
        avg_data = comm_rounds['CommAmount [B]'].mean() / (1024 * 1024)  # Convert bytes to MB
        avg_data_communicated.append(avg_data)

        # Calculate average power and total energy for communication rounds
        power_list = []
        energy_list = []
        for round_start, round_end in zip(comm_df[comm_df['CommRound'] == -1]['Timestamp'], comm_rounds['Timestamp']):
            time_frame = pow_temp_df[(pow_temp_df['Timestamp'] >= round_start) & (pow_temp_df['Timestamp'] <= round_end)]
            avg_power = time_frame['Power'].mean()
            total_energy = time_frame['Power'].sum()  # Considering each row as 1 second
            power_list.append(avg_power)
            energy_list.append(total_energy)

        avg_powers.append(np.mean(power_list))
        avg_energies.append(np.mean(energy_list))

    fig, axs = plt.subplots(1, sum([power, energy, communication]), figsize=(18, 6))
    if sum([power, energy, communication]) == 1:
        axs = [axs]
    plot_index = 0
    colors = ['blue', 'orange']  # Colors for the devices
    title = ""
    return_list = []
    if power:
        # Average Power Consumption per Device [W]
        axs[plot_index].bar(np.arange(len(devices)), avg_powers, width=0.35, color=colors, label=['RPi', 'MC1'])
        axs[plot_index].set_title('Average Power Consumption per Device')
        axs[plot_index].set_xticks(np.arange(len(devices)))
        axs[plot_index].set_xticklabels(['RPi', 'MC1'])
        axs[plot_index].set_ylabel('Power [W]')
        axs[plot_index].grid(True)
        axs[plot_index].set_ylim(0, 12)
        plot_index += 1
        title+='pow_'
        return_list.append(avg_powers)

    if energy:
        # Average Energy Consumption per Communication Round [J]
        axs[plot_index].bar(np.arange(len(devices)), avg_energies, width=0.35, color=colors, label=['RPi', 'MC1'])
        axs[plot_index].set_title('Average Energy Consumption per Communication Round [J]')
        axs[plot_index].set_xticks(np.arange(len(devices)))
        axs[plot_index].set_xticklabels(['RPi', 'MC1'])
        axs[plot_index].set_ylabel('Energy [J]')
        axs[plot_index].grid(True)
        axs[plot_index].set_ylim(0, 2000)
        plot_index += 1
        title += 'energy_'
        return_list.append(avg_energies)

    if communication:
        # Average Communicated Data per Communication Round [MB]
        axs[plot_index].bar(np.arange(len(devices)), avg_data_communicated, width=0.35, color=colors, label=['RPi', 'MC1'])
        axs[plot_index].set_title('Avg. Amount of Communicated Data per Communication Round [MB]')
        axs[plot_index].set_xticks(np.arange(len(devices)))
        axs[plot_index].set_xticklabels(['RPi', 'MC1'])
        axs[plot_index].set_ylabel('Data [MB]')
        axs[plot_index].grid(True)
        axs[plot_index].set_ylim(0, 2)
        title += 'comm'
        return_list.append(avg_data_communicated)

    plt.tight_layout()
    plt.savefig(f'figs/exp{exp}_run{run}_{title}.png', bbox_inches='tight', dpi=300)
    return return_list



parser = argparse.ArgumentParser(description="Results arguments")
parser.add_argument('--accuracy', action='store_true', help='')
parser.add_argument('--time', action='store_true', help='')
parser.add_argument('--power', action='store_true', help='')
parser.add_argument('--energy', action='store_true', help='')
parser.add_argument('--communication', action='store_true', help='')
parser.add_argument('--exps', nargs='+', default=[1, 2], help='Experiment numbers')
parser.add_argument('--exp_labels', nargs='+', default=["iid", "niid"], help='Experiment labels to be used as plot titles')
parser.add_argument('--runs', nargs='+', default=[2, 1], help='Total number of runs for each experiments')
parser.add_argument('--plot_title', type=str, default="Testing", help='The main title of the plots')
args = parser.parse_args()

exps = []
for k in args.exps:
    exps.append(int(k))

runs = []
for k in args.runs:
    runs.append(int(k))

exp_labels = []
for k in args.exp_labels:
    exp_labels.append(k)

args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs("figs", exist_ok=True)
    device_dict = {0: 'rpi', 1: 'mc1'}
    for exp, r, title in zip(exps, runs, exp_labels):
        cloud_cfg_file = f"configs/cloud_cfg_exp{exp}_run{r}.json"
        dev_cfg_file = f"configs/dev_cfg_exp{exp}_run{r}.json"
        with open(dev_cfg_file, "r") as f:
            dev_cfg = json.load(f)
            device_numbers = []
            for device in dev_cfg.keys():
                if device.startswith('dev'):
                    device_number = int(dev_cfg[device]['host'].split('-')[-1].split('.')[0])
                    device_numbers.append(device_number)
        for device, dev_idx in zip([0, 1], device_numbers):
            for rs in range(1, r+1):
                file_name_pow = f"dev{device}_pow_temp_log_exp{exp}_run{rs}.csv"
                result = fetch_file(device_dict[device], dev_idx, file_name_pow, 'logs/device_logs/')
                if result == -1:
                    print(f"[!] File name: {file_name_pow} does not exist for device {device}")
                    exit(-1)
                file_name_comm = f"dev{device}_communication_log_exp{exp}_run{rs}.csv"
                result = fetch_file(device_dict[device], dev_idx, file_name_comm, 'logs/device_logs/')
                if result == -1:
                    print(f"[!] File name: {file_name_comm} does not exist for device {device}")
                    exit(-1)
    if args.accuracy:
        plot_acc_loss(exps, runs, exp_labels, args.plot_title, args.time)

    for exp, myrun in zip(exps,runs):
        all_runs = []
        for r in range(1,myrun+1):
            all_runs.append(plot_power_energy_communication(exp=exp, run=r, power=args.power,
                                                            energy=args.energy, communication=args.communication))
        # Initialize lists to accumulate metrics across all runs
        accumulated_metrics = [[0, 0], [0, 0], [0, 0]]  # For power, energy, and comm data

        for run in all_runs:
            for i, metrics in enumerate(run):
                accumulated_metrics[i] = [sum(x) for x in zip(accumulated_metrics[i], metrics)]

        # Calculate the average for each metric across all runs
        avg_metrics = [[metric / len(all_runs) for metric in metrics] for metrics in accumulated_metrics]

        # Extracting the averaged metrics
        avg_powers, avg_energies, avg_data_communicated = avg_metrics

        devices = 2  # Number of devices
        # Printing the results
        if myrun > 1:
            print(f"\nExperiment {exp} average over {myrun} runs:")
        else:
            print(f"\nExperiment {exp}:")

        # Energy
        if args.energy:
            print(f"\tRPi avg. energy consumption per round [J]: {avg_energies[0]:,.2f} Joules")
            print(f"\tMC1 avg. energy consumption per round [J]: {avg_energies[1]:,.2f} Joules")
            total_avg_energy = sum(avg_energies)
            print(f"\tTotal avg. energy consumption per communication round [J]: {total_avg_energy:,.2f} Joules")

        # Communication
        if args.communication:
            avg_data_comm = sum(avg_data_communicated) / devices
            print(f"\tAvg. amount of communicated data per communication round [MB]: {avg_data_comm:.2f} MB")

        # Power
        if args.power:
            print(f"\tRPi avg. power consumption per round [W]: {avg_powers[0]:,.2f} Watts")
            print(f"\tMC1 avg. power consumption per round [W]: {avg_powers[1]:,.2f} Watts")


