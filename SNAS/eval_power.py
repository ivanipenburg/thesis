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
        with open("bash_command.txt", "r") as f:
            profiler_string = f.readline()
        
        profiler_string = profiler_string.replace("metaarch", meta_arch, 2)
        profiler_string = profiler_string.replace("channels", channels, 2)
        profiler_string = profiler_string.replace("run", str(i), 2)

        profiler = subprocess.Popen(profiler_string, shell=True, preexec_fn=os.setsid)

        try:

            test_string = "python3 test.py --auxiliary --arch 'SNAS_mild_edge_all' --model_path '../weights_" + meta_arch + "_" + channels + "c.pt' --timestamp 'time_" + meta_arch + "_" + channels + "c_" +    str(i) + ".csv' --batch_size 64 --init_channels " + channels + " --meta_arch '" + meta_arch_name + "'"
            test_script = subprocess.Popen(test_string, shell=True)

            done = False
            while done == False:
                if test_script.poll() != None:
                    done = True
                    os.killpg(os.getpgid(profiler.pid), signal.SIGTERM)
                    print("Profiler terminated")
                else:
                    time.sleep(5)

        except KeyboardInterrupt:
            os.killpg(os.getpgid(profiler.pid), signal.SIGTERM)

