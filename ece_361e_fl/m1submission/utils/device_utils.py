from os import path
import os
import threading
import time
import telnetlib


def measure_pow_temp(stop_event, device_type, dev_idx, exp, r, verbose=False):
    """
    Records power consumption and temperature of the device, logs them to a file.

    Parameters:
        stop_event (threading.Event): Event to signal the function to stop.
        device_type (str): Type of the device ('rpi' or 'mc1').
        dev_idx (int): Index of the device.
        exp (int): Experiment number.
        r (int): Run number.
        verbose (bool, optional): If True, displays status messages during execution. Defaults to False.
    """

    # Check the device type and import the appropriate module
    if device_type == 'rpi':
        import gpiozero
    elif device_type == 'mc1':
        import utils.sysfs_paths as sysfs_paths

    # Ensure the logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Define the path for the log file
    log_file_path = path.join("logs", f"dev{dev_idx}_pow_temp_log_exp{exp}_run{r}.csv")

    # Open the log file and write the headers
    with open(log_file_path, 'w') as logger:
        logger.write('Timestamp,Power,Temperature\n')

    # If power logging is enabled, establish a Telnet connection
    sp2 = telnetlib.Telnet('192.168.4.1')
    time.sleep(2)

    if verbose:
        print("[+] Telnet connection successful")

    last_power = 0.0
    temp = 0.0
    pwr = 0.0

    # Print the recording status
    print(f'[+] Recording power and temperature...')

    # Start the measurement loop
    while not stop_event.is_set():
        loop_start = time.time()

        # If power logging is enabled, read and process power data
        sp2_readings = str(sp2.read_eager())
        # Find latest power measurement in the data
        idx = sp2_readings.rfind('\n')
        idx2 = sp2_readings[:idx].rfind('\n')
        idx2 = idx2 if idx2 != -1 else 0
        ln = sp2_readings[idx2:idx].strip().split(',')
        if len(ln) < 2:
            pwr = last_power
        else:
            pwr = float(ln[-2])
            last_power = pwr

        # If temperature logging is enabled, read and process temperature data
        if device_type == 'mc1':
            # For mc1 devices, calculate average temperature from 4 sensors
            for i in range(4):
                with open(sysfs_paths.fn_thermal_sensor.format(i)) as f:
                    temp += int(f.read()) / 1000
            temp /= 4
        elif device_type == 'rpi':
            # For rpi devices, use the gpiozero module to get the temperature
            temp = gpiozero.CPUTemperature().temperature

        # Open the log file and write the measurement data
        with open(log_file_path, 'a+') as logger:
            logger.write(f'{time.time()},{last_power},{temp}\n')

        # Sleep for the remaining time of the measurement rate which is 1 second
        time.sleep(max(0., 1 - (time.time() - loop_start)))
