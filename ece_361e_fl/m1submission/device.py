from os import path
import os
import argparse
import time
from sys import getsizeof
from utils.communication import receive_msg, send_msg, close_connection, receive_file, send_file
from utils.train_test import local_training
from utils.general_utils import get_loss_func, get_hw_info, seed_everything
import socket
import threading
from utils.device_utils import measure_pow_temp


class Device:
    def __init__(self, connection, exp, r, dev_idx, verbose=False, seed=42):
        seed_everything(seed=seed)
        self.seed = seed
        self.connection = connection
        self.dev_idx = dev_idx
        self.comm_log_file = f"dev{self.dev_idx}_communication_log_exp{exp}_run{r}.csv"
        self.verbose = verbose

    def run(self):
        """
        Main method for the Device. The Cloud sends a message to the Device in the following format:

        setup_message = {cloud_info},{device_info},{data}

        where

        {cloud_info}  = {target_ip};{target_port};{target_path};{target_usr};{target_pwd}
        {device_info} = {hw_type};{data_iid};{cuda_name};{verbose};{seed}
        {data}        = {comm_round};{model_name};{filename};{local_epochs};{learning_rate};{loss_name};{loss_type};{mu};{beta}
        """
        with open(path.join("logs", self.comm_log_file), 'a+') as logger:
            logger.write(f'{time.time()},-1,,\n')

        # Step 1. Receive setup message from Cloud.
        setup_message = receive_msg(self.connection, verbose=self.verbose)

        # Exit if there is no message.
        if setup_message is None:
            print(f"[!] No message received from the Cloud on device {self.dev_idx}.")
            return False

        if setup_message == "end":
            close_connection(connection=self.connection, verbose=self.verbose)
            return True

        filesize = getsizeof(setup_message)

        cloud_info, device_info, data = setup_message.split(',')

        target_host, target_port, target_path, target_usr, target_pwd = cloud_info.split(';')

        hw_type, data_iid, cuda_name, self.verbose, seed = device_info.split(';')
        assert seed != self.seed, "Configuration seed different from device seed"
        dev_pwd, dev_usr, dev_path = get_hw_info(hw_type)
        os.makedirs(dev_path, exist_ok=True)
        data_iid = True if data_iid.lower() == "true" else False
        self.verbose = True if self.verbose.lower() == "true" else False

        comm_round, model_name, model_filename, local_epochs, learning_rate, loss_name, loss_type, mu, beta = \
            data.split(';')
        comm_round = int(comm_round)
        local_epochs = int(local_epochs)
        learning_rate = float(learning_rate)
        mu = float(mu)
        beta = float(beta)
        loss_func = get_loss_func(loss_name)

        model_path = path.join(dev_path, model_filename)

        # Step 2. Synch with Cloud with msg "done_setup"
        comm_time = 0.
        start_time = time.time()

        send_msg(connection=self.connection, msg="done_setup", verbose=self.verbose)

        filesize += getsizeof("done_setup")
        comm_time += time.time() - start_time

        # Step 3. Receive local model weight file
        start_time = time.time()
        filesize += receive_file(connection=self.connection, target="Cloud", target_host=target_host,
                                 local_path=dev_path, verbose=self.verbose, msg="model weight zip")
        comm_time += time.time() - start_time

        # Step 4. Train the model
        train_loss, train_acc, train_time = \
            local_training(model_name=model_name, loss_func=loss_func, loss_type=loss_type, model_path=model_path,
                           cuda_name=cuda_name, learning_rate=learning_rate, local_epochs=local_epochs,
                           dev_idx=self.dev_idx, verbose=self.verbose, data_iid=data_iid, seed=self.seed, mu=mu,
                           beta=beta
                           )

        # Step 5. Sync with Cloud with "done_training" message
        start_time = time.time()
        send_msg(connection=self.connection, msg=f"done_training;{train_time}", verbose=self.verbose)
        filesize += getsizeof(f"done_training;{train_time}")
        comm_time += time.time() - start_time

        # Step 6. Send updated model weights back to Cloud
        start_time = time.time()
        filesize += send_file(connection=self.connection, target="Cloud", target_host=target_host,
                              target_port=target_port, target_usr=target_usr, target_pwd=target_pwd,
                              target_path=target_path, filename=model_filename, source_path=dev_path,
                              verbose=self.verbose)
        comm_time += time.time() - start_time

        with open(path.join("logs", self.comm_log_file), 'a+') as logger:
            logger.write(f'{time.time()},{comm_round},{filesize},{comm_time}\n')

        close_connection(connection=self.connection, verbose=self.verbose)
        return False


def start_device(host, port, device_type, dev_idx, exp, r, verbose=False, seed=42):
    """
    Starts the device, handles incoming connections, and records power and temperature (optional).

    Parameters:
        host (str): Hostname or IP to bind the device to.
        port (int): Port to bind the device to.
        device_type (str): Type of the device ('rpi' or 'mc1').
        dev_idx (int): Index of the device.
        exp (int): Experiment number.
        r (int): Run number.
        seed (int): Seed for experiment.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.

    Returns:
        None
    """
    os.makedirs("logs", exist_ok=True)
    with open(path.join("logs", f"dev{dev_idx}_communication_log_exp{exp}_run{r}.csv"), 'w') as logger:
        logger.write('Timestamp,CommRound,CommAmount [B],CommTime [s]\n')

    # Create a socket object
    soc = socket.socket()
    # Set the socket to reuse the address
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bind the socket to the host and port
    soc.bind((host, port))

    if verbose:
        print(f"[+] Socket created on device {dev_idx}")

    # Start the thread for recording power and temperature
    stop_event = threading.Event()
    t1 = threading.Thread(target=measure_pow_temp, args=(stop_event, device_type, dev_idx, exp, r, verbose))
    t1.start()

    try:
        while True:
            # Listen for incoming connections
            soc.listen(5)
            print(f"\n[+] Device {dev_idx} is listening on host:port {host}:{port}")
            # Accept a connection
            connection, _ = soc.accept()
            # Create a Device instance with the accepted connection
            device = Device(connection=connection, exp=exp, r=r, dev_idx=dev_idx, verbose=verbose, seed=seed)
            # Run the device
            if device.run():
                break
    except BaseException as e:
        print(f'[!] ERROR: {e} Socket closed due to no incoming connections or error.')
    finally:
        # Stop the power and temperature recording thread if it's running
        if stop_event:
            stop_event.set()
            t1.join()
        # Close the socket
        soc.close()

parser = argparse.ArgumentParser(description="Device arguments")
parser.add_argument('--host', default="127.0.0.1", help='IP address of the device')
parser.add_argument('--port', type=int, default=10001, help='Port for the device to listen on')
parser.add_argument('--device_type', default='rpi', help='Hardware type of the device [rpi/mc1]')
parser.add_argument('--dev_idx', type=int, default=0, help='Index of the device')
parser.add_argument('--exp', type=int, default=0,  help='Experiment number')
parser.add_argument('--r', type=int, default=0, help='Run number')
parser.add_argument('--seed', type=int, default=42, help='Seed for experiment')
parser.add_argument('--verbose', action='store_true', help='If verbose or not')
args = parser.parse_args()

start_device(host=args.host, port=args.port, device_type=args.device_type, dev_idx=args.dev_idx, exp=args.exp, r=args.r,
             verbose=args.verbose, seed=args.seed)
