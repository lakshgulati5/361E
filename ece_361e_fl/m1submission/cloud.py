from os import path
import os
import copy
import torch
import argparse
import json
from models.get_model import get_model
from utils.train_test import test
from utils.general_utils import get_loss_func, get_hw_info, seed_everything
from utils.fl_utils import aggregate_avg
import time
from device_handler import DeviceHandler


class Cloud:
    def __init__(self, cloud_cfg, dev_cfg):
        self.cloud_cfg = cloud_cfg
        self.dev_cfg = dev_cfg
        os.makedirs("logs", exist_ok=True)
        with open(self.cloud_cfg, 'r') as cfg:
            dat = json.load(cfg)
        seed_everything(dat["seed"])

    def federated_learning(self):
        total_time_start = time.time()
        with open(self.cloud_cfg, 'r') as cfg:
            dat = json.load(cfg)

            exp = dat["experiment"]
            r = dat["run"]
            log_file = f"log_{exp}_{r}.csv"

            laptop_number = dat["laptop_number"]

            cloud_ip = dat["cloud_ip"]
            cloud_port = dat["cloud_port"]
            cloud_pwd, cloud_usr, cloud_path = get_hw_info(hw_type=laptop_number)
            os.makedirs(cloud_path, exist_ok=True)
            cloud_cuda_name = dat["cloud_cuda_name"]

            model_name = dat["model_name"]
            comm_rounds = dat["comm_rounds"]
            loss_name = dat["loss_func_name"]
            loss_type = dat["loss_type"]
            mu = dat["mu"]
            beta = dat["beta"]
            learning_rate = dat["learning_rate"]
            data_iid = dat["data_iid"]
            verbose = dat["verbose"]
            seed = dat["seed"]

        net_glob = get_model(model_name=f"{model_name}", loss_type=loss_type)
        torch.save(net_glob.state_dict(), path.join(cloud_path, f"global_weights.pth"))
        loss_func = get_loss_func(loss_name=loss_name)

        loss_test, acc_test = test(model=net_glob, loss_func=loss_func, cuda_name=cloud_cuda_name, verbose=verbose,
                                   seed=seed, loss_type=loss_type)
        print(f"Initial Accuracy: {acc_test:.2f}%; Initial Loss: {loss_test:.4f}")

        with open(path.join("logs", log_file), 'w') as logger:
            logger.write(f"CommRound,Acc,Loss,Time\n0,{acc_test},{loss_test},{time.time()-total_time_start}\n")

        for comm_round in range(1,comm_rounds+1):
            comm_round_time_start = time.time()
            global_weights = torch.load(path.join(cloud_path, f"global_weights.pth"))
            net_glob.load_state_dict(global_weights)

            with open(self.dev_cfg, 'r') as f:
                dt = json.load(f)
            num_devices = dt["num_devices"]
            if verbose:
                print(f"Number of devices: {num_devices}")

            device_handler_list = []
            for idx, i in enumerate(range(num_devices)):
                dev = dt[f"dev{i+1}"]
                dev_type = dev["hw_type"]
                local_epochs = dev["local_epochs"]
                dev_host = dev["host"]
                dev_port = dev["port"]
                dev_cuda_name = dev["cuda_name"]
                dev_model_name = dev["model_name"]

                net_local = copy.deepcopy(net_glob).to(torch.device(dev_cuda_name))
                dev_model_filename = f"dev_{i}.pth"
                torch.save(net_local.state_dict(), path.join(cloud_path, dev_model_filename))
                dev_pwd, dev_usr, dev_path = get_hw_info(dev_type)

                """
                setup_message = {cloud_info},{device_info},{data}

                where

                {cloud_info}  = {cloud_ip};{cloud_port};{cloud_path};{cloud_usr};{cloud_pwd}
                {device_info} = {dev_type};{data_iid};{cuda_name};{verbose};{seed}
                {data}        = {comm_round};{model_name};{filename};{local_epochs};{learning_rate};{loss_name};{loss_type};{mu};{beta}
                """
                cloud_info = f"{cloud_ip};{cloud_port};{cloud_path};{cloud_usr};{cloud_pwd}"
                device_info = f"{dev_type};{data_iid};{dev_cuda_name};{verbose};{seed}"
                data = f"{comm_round};{dev_model_name};{dev_model_filename};{local_epochs};{learning_rate};{loss_name};{loss_type};{mu};{beta}"

                setup_message = f"{cloud_info},{device_info},{data}"
                device_handler_list.append(
                    DeviceHandler(cloud_path=cloud_path, dev_idx=i, dev_host=dev_host, dev_port=dev_port,
                                  dev_usr=dev_usr, dev_pwd=dev_pwd, dev_path=dev_path,
                                  dev_model_filename=dev_model_filename, setup_message=setup_message, verbose=verbose)
                )

            if verbose:
                print(f"[+] Started all clients")
            for idx, i in enumerate(range(num_devices)):
                device_handler_list[idx].start()

            if verbose:
                print(f"\n[+] Wait until clients to finish their job")
            value = []
            for idx, i in enumerate(range(num_devices)):
                value.append(device_handler_list[idx].join())

            if verbose:
                print("[+] Joined all clients")

            local_weights = []
            for i in range(num_devices):
                dev_model_filename = f"dev_{i}.pth"
                net_local = get_model(model_name=f"{model_name}", loss_type=loss_type)
                net_local.load_state_dict(torch.load(path.join(cloud_path, dev_model_filename)))
                w_local = net_local.state_dict()
                local_weights.append(w_local)

            w_glob = aggregate_avg(local_weights=local_weights)
            torch.save(w_glob, path.join(cloud_path, f"global_weights.pth"))
            net_glob.load_state_dict(torch.load(path.join(cloud_path, f"global_weights.pth")))
            loss_test, acc_test = test(model=net_glob, loss_func=loss_func, cuda_name=cloud_cuda_name, verbose=verbose,
                                       seed=seed, loss_type=loss_type)

            with open(path.join("logs", log_file), 'a+') as logger:
                logger.write(f"{comm_round},{acc_test},{loss_test},{time.time()-comm_round_time_start}\n")
            print(f"CommRound: {comm_round}; Accuracy: {acc_test:.2f}%; Loss: {loss_test:.4f}")

        print(f"Total time for experiment: {time.time() - total_time_start} seconds")

        with open(path.join("logs", "time.csv"), 'a+') as logger:
            logger.write(f"{exp},{r},{time.time() - total_time_start}\n")

        self.end_experiment(verbose=verbose)

    def end_experiment(self, verbose):
        if verbose:
            print("[+] Closing everything")
        device_handler_list = []
        with open(self.dev_cfg, 'r') as f:
            dt = json.load(f)
        for i in range(dt["num_devices"]):
            dev = dt[f"dev{i + 1}"]
            device_handler_list.append(
                DeviceHandler(dev_host=dev["host"], dev_port=dev["port"], setup_message="end", verbose=verbose)
            )

        if verbose:
            print("[+] Closing all clients...")

        for i in range(dt["num_devices"]):
            device_handler_list[i].start()

        if verbose:
            print("[+] Wait until clients close")

        for i in range(dt["num_devices"]):
            device_handler_list[i].join()

        if verbose:
            print("[+] Closed all clients")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloud_cfg', type=str, default="configs/cloud_cfg_exp1.json",
                        help='Cloud configuration file name')
    parser.add_argument('--dev_cfg', type=str, default="configs/dev_cfg_exp1.json",
                        help='Device configuration file name')
    args = parser.parse_args()

    cloud = Cloud(cloud_cfg=args.cloud_cfg, dev_cfg=args.dev_cfg)
    cloud.federated_learning()
