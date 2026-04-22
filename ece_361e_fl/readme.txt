config_33.bash contains the best experiment
experiment 41 (using m3_generate_figs.py)

Cloud Command:
venv/Scripts/python.exe cloud.py --cloud_cfg configs/cloud_cfg_exp41_run1.json --dev_cfg configs/dev_cfg_exp41_run1.json


venv/Scripts/python.exe m3_generate_figs.py --accuracy --time --power --energy --communication --exps 41 --runs 1 --ex
p_labels NonIID --plot_title Exp41 --m3


Raspberry Pi Command:
python device.py --host sld-rpi-13.ece.utexas.edu --port 9090 --device_type rpi --dev_idx 0 --r 1 --seed 2 --exp 41

Odroid Command:
taskset --all-tasks 0xF0 python device.py --host sld-mc1-13.ece.utexas.edu --port 9090 --device_type mc1 --dev_idx 1 --r 1 --seed 2 --exp 41