import threading
from utils.communication import receive_msg, send_msg, close_connection, receive_file, send_file
from utils.general_utils import seed_everything
from utils.communication import connect


class DeviceHandler(threading.Thread):
    def __init__(self, dev_host, dev_port, cloud_path=None, dev_idx=None, dev_usr=None, dev_pwd=None, dev_path=None,
                 dev_model_filename=None, setup_message=None, verbose=False):
        threading.Thread.__init__(self)

        seed_everything()
        self.connection = None
        self.cloud_path = cloud_path
        self.dev_model_filename = dev_model_filename
        self.setup_message = setup_message
        self.dev_host = dev_host
        self.dev_port = dev_port
        self.dev_usr = dev_usr
        self.dev_pwd = dev_pwd
        self.dev_path = dev_path
        self.dev_idx = dev_idx
        self.verbose = verbose
        self.train_time = None

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self.train_time

    def run(self):
        # Step 1. Connect to device
        self.connection = connect(host=self.dev_host, port=self.dev_port, verbose=self.verbose)

        # Step 2. Send configs and wait for setup to be done
        send_msg(connection=self.connection, msg=self.setup_message, verbose=self.verbose)
        if self.setup_message == "end":
            close_connection(connection=self.connection, verbose=self.verbose)
            exit()

        # Step 3. Synch with device with "done_setup"
        done_setup = receive_msg(connection=self.connection, verbose=self.verbose)
        assert done_setup is not "done_setup", f"[!] Received no input from Device{self.dev_idx}"

        # Step 4. Send the device model weight
        send_file(connection=self.connection, target=f"Device{self.dev_idx}", target_host=self.dev_host,
                  target_port=self.dev_port, target_usr=self.dev_usr, target_pwd=self.dev_pwd,
                  target_path=self.dev_path, filename=self.dev_model_filename, source_path=self.cloud_path,
                  verbose=self.verbose)
        # Step 5. Synch with the device
        received_data = receive_msg(connection=self.connection, verbose=self.verbose)
        if received_data.split(';')[0] != "done_training":
            print("[!] ERROR not done training")

        self.train_time = received_data.split(';')[1]

        # Step 6. Get the weights file from the device
        receive_file(connection=self.connection, target=f"Device{self.dev_idx}", target_host=self.dev_host,
                     local_path=self.cloud_path, verbose=self.verbose, msg="local models zip")
        close_connection(connection=self.connection, verbose=self.verbose)
