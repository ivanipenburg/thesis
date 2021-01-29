import subprocess
import time
import os
import signal

runs = 50

meta_archs = ["6-6-6", "6-6-6", "6-6-6"]
channels_list = ["36", "24", "12"]

for meta_arch_name, channels in zip(meta_archs, channels_list):
    meta_arch = meta_arch_name.replace("-", "", 2)

    for i in range(runs):
        print(f"Run {i}")
        test_string = "python3 test.py --auxiliary --arch 'SNAS_mild_edge_all' --model_path '../weights_" + meta_arch + "_" + channels + "c.pt' --timestamp 'time_" + meta_arch + "_" + channels + "c_" +    str(i) + ".csv' --batch_size 64 --init_channels " + channels + " --meta_arch '" + meta_arch_name + "'"
        test_script = subprocess.Popen(test_string, shell=True)

        done = False
        while done == False:
            if test_script.poll() != None:
                done = True
                print("Profiler terminated")
            else:
                time.sleep(5)

