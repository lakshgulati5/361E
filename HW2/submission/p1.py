import sysfs_paths as sysfs
import subprocess
import time

from matplotlib import pyplot as plt


def get_avail_freqs(cluster):
    """
    Obtain the available frequency for a CPU. Return unit in KHz by default!
    """
    # Read CPU freq from sysfs_paths.py
    freqs = open(sysfs.fn_cluster_freq_range.format(cluster)).read().strip().split(' ')
    return [int(f.strip()) for f in freqs]


def get_cluster_freq(cluster_num):
    """
    Read the current cluster freq. cluster_num must be 0 (LITTLE) or 4 (big)
    """
    with open(sysfs.fn_cluster_freq_read.format(cluster_num), 'r') as f:
        return int(f.read().strip())


def set_user_space(clusters=None):
    """
    Set the system governor as 'userspace'. This is necessary before you can change the
    cluster/CPU freq to customized values
    """
    print("Setting userspace")
    clusters = [0, 4]
    for i in clusters:
        with open(sysfs.fn_cluster_gov.format(i), 'w') as f:
            f.write('userspace')


def set_cluster_freq(cluster_num, frequency):
    """
    Set customized freq for a cluster. Accepts frequency in KHz as int or string
    """
    with open(sysfs.fn_cluster_freq_set.format(cluster_num), 'w') as f:
        f.write(str(frequency))

def q1_data_collection():
    print('Available freqs for LITTLE cluster:', get_avail_freqs(0))
    print('Available freqs for big cluster:', get_avail_freqs(4))
    set_user_space()
    set_cluster_freq(4, 2000000)   # big cluster
    # print current freq for the big cluster
    print('Current freq for big cluster:', get_cluster_freq(4))

    # execution of your benchmark    
    start = time.time()
    print("Start time: ", start)
    # run the benchmark
    command = "taskset --all-tasks 0x20 /home/student/HW2_files/TPBench.exe"   # 0x20: core 5 (7 6 5 4 3 2 1 0)
    proc_ben = subprocess.call(command.split())

    end = time.time()
    print("End time: ", end)
    total_time = end - start
    print("Benchmark runtime:", total_time)


def q1_plot_data():
    # 3 plots
    # W vs time
    # big core temp vs time
    # big core usage vs time
    data = dict()
    with open('log.txt', 'r') as f:
        column_names = f.readline().strip().split()
        for column_name in column_names:
            data[column_name] = []
        # print(column_names)

        for line in f:
            numbers = line.strip().split()
            for i in range(len(numbers)):
                data[column_names[i]].append(float(numbers[i]))
    
    time_data = data['time']
    power_data = data['W']

    usage_4 = [100 * d for d in data['usage_c4']]
    usage_5 = [100 * d for d in data['usage_c5']]
    usage_6 = [100 * d for d in data['usage_c6']]
    usage_7 = [100 * d for d in data['usage_c7']]

    temp_4 = data['temp4']
    temp_5 = data['temp5']
    temp_6 = data['temp6']
    temp_7 = data['temp7']

    initial_time = time_data[0]
    normalized_time_data = [t - initial_time for t in time_data]
    # print(normalized_time_data)

    plt.figure(1)
    plt.plot(normalized_time_data, power_data)
    plt.grid()
    plt.title('System Power Consumption vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('System Power Consumption (W)')
    plt.savefig('plot1.png')


    plt.figure(2)
    plt.plot(normalized_time_data, temp_4, label='core 4')
    plt.plot(normalized_time_data, temp_5, label='core 5')
    plt.plot(normalized_time_data, temp_6, label='core 6')
    plt.plot(normalized_time_data, temp_7, label='core 7')
    
    plt.grid()
    plt.legend()
    plt.title('Big Core Temperatures vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.savefig('plot2.png')


    plt.figure(3)
    plt.plot(normalized_time_data, usage_4, label='core 4')
    plt.plot(normalized_time_data, usage_5, label='core 5')
    plt.plot(normalized_time_data, usage_6, label='core 6')
    plt.plot(normalized_time_data, usage_7, label='core 7')

    plt.grid()
    plt.legend()
    plt.title('Big Core Usage vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Usage (%)')
    plt.savefig('plot3.png')


def q3_p1():
    print('Available freqs for LITTLE cluster:', get_avail_freqs(0))
    print('Available freqs for big cluster:', get_avail_freqs(4))
    set_user_space()
    set_cluster_freq(4, 2000000)   # big cluster
    # print current freq for the big cluster
    print('Current freq for big cluster:', get_cluster_freq(4))

    start = time.time()
    print("Start time: ", start)
    

    # blackscholes
    command = "taskset --all-tasks 0xF0 ./parsec_files/blackscholes 4 ./parsec_files/in_10M_blackscholes.txt blackscholes_out.txt"
    proc_ben = subprocess.call(command.split())

    end = time.time()
    print("End time: ", end)
    total_time = end - start
    print("Benchmark runtime:", total_time)


def q3_p2():
    print('Available freqs for LITTLE cluster:', get_avail_freqs(0))
    print('Available freqs for big cluster:', get_avail_freqs(4))
    set_user_space()
    set_cluster_freq(4, 2000000)   # big cluster
    # print current freq for the big cluster
    print('Current freq for big cluster:', get_cluster_freq(4))

    start = time.time()
    print("Start time: ", start)
    

    # bodytrack
    command = "taskset --all-tasks 0xF0 ./parsec_files/bodytrack ./parsec_files/sequenceB_261 4 260 3000 8 3 4 0"
    proc_ben = subprocess.call(command.split())

    end = time.time()
    print("End time: ", end)
    total_time = end - start
    print("Benchmark runtime:", total_time)



fig_counter = 0
def q3_plot_data(title, in_file, out_file1, out_file2):
    # 4 plots
    # W vs time
    # big core temp vs time

    global fig_counter

    data = dict()
    with open(in_file, 'r') as f:
        column_names = f.readline().strip().split()
        for column_name in column_names:
            data[column_name] = []
        # print(column_names)

        for line in f:
            numbers = line.strip().split()
            for i in range(len(numbers)):
                data[column_names[i]].append(float(numbers[i]))
    
    time_data = data['time']
    power_data = data['W']

    temp_4 = data['temp4']
    temp_5 = data['temp5']
    temp_6 = data['temp6']
    temp_7 = data['temp7']

    initial_time = time_data[0]
    normalized_time_data = [t - initial_time for t in time_data]
    # print(normalized_time_data)

    max_big_temp = []
    assert len(temp_4) == len(temp_5)
    assert len(temp_5) == len(temp_6)
    assert len(temp_6) == len(temp_7)

    for i in range(len(temp_4)):
        max_big_temp.append(max(temp_4[i], temp_5[i], temp_6[i], temp_7[i]))


    fig_counter += 1
    plt.figure(fig_counter)
    plt.plot(normalized_time_data, power_data)
    plt.grid()
    plt.title(f'{title} System Power Consumption vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('System Power Consumption (W)')
    plt.savefig(out_file1)

    fig_counter += 1
    plt.figure(fig_counter)
    plt.plot(normalized_time_data, max_big_temp)
    
    plt.grid()
    plt.title(f'{title} Max Big Core Temperature vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (C)')
    plt.savefig(out_file2)


def q3_table_values(title, file_path, runtime):
    print("\nTable values for", title)
    data = dict()
    with open(file_path, 'r') as f:
        column_names = f.readline().strip().split()
        for column_name in column_names:
            data[column_name] = []
        # print(column_names)

        for line in f:
            numbers = line.strip().split()
            for i in range(len(numbers)):
                data[column_names[i]].append(float(numbers[i]))
    
    time_data = data['time']
    power_data = data['W']

    usage_4 = [100 * d for d in data['usage_c4']]
    usage_5 = [100 * d for d in data['usage_c5']]
    usage_6 = [100 * d for d in data['usage_c6']]
    usage_7 = [100 * d for d in data['usage_c7']]

    temp_4 = data['temp4']
    temp_5 = data['temp5']
    temp_6 = data['temp6']
    temp_7 = data['temp7']

    max_big_temp = []
    assert len(temp_4) == len(temp_5)
    assert len(temp_5) == len(temp_6)
    assert len(temp_6) == len(temp_7)

    for i in range(len(temp_4)):
        max_big_temp.append(max(temp_4[i], temp_5[i], temp_6[i], temp_7[i]))

    initial_time = time_data[0]
    normalized_time_data = [t - initial_time for t in time_data]
   
    avg_power = sum(power_data)/len(power_data)
    avg_max_temp = sum(max_big_temp)/len(max_big_temp)
    max_temp_overall = max(*max_big_temp)
    energy = avg_power * runtime

    print("Runtime (s):", runtime)
    print("Average Power (W):", avg_power)
    print("Average Max Temp (C):", avg_max_temp)
    print("Max Temp (C):", max_temp_overall)
    print("Energy (J):", energy)
    print("\n")
 


#q1_data_collection()
#q1_plot_data()

#q3_p1()
#q3_p2()
#q3_plot_data('Blackscholes', 'log.txt', 'q3plot1.png', 'q3plot2.png')
#q3_plot_data('Bodytrack', 'log2.txt', 'q3plot3.png', 'q3plot4.png')

q3_table_values('Blackscholes', 'log.txt', 133.980)
q3_table_values('Bodytrack', 'log2.txt', 133.177)
