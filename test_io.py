import subprocess
import sys
import time

import pandas as pd


def run(bs=4, numjobs=1):
    command = f"fio --randrepeat=1 --ioengine=posixaio --direct=1 --gtod_reduce=1 --name=test --filename=test --bs={bs}k --iodepth=64 --size=4G --readwrite=randread --numjobs={numjobs}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    r = str(stdout)
    r = r[r.find("READ:"):-3]
    print(f"bs = {bs}, numjobs = {numjobs}", r)



if __name__ == '__main__':
    for bs in range(10):
        for numjobs in range(1):
            run(1024 * (2**bs), 2**numjobs)
        print("")