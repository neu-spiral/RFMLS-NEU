'''
This source code was developed under the DARPA Radio Frequency Machine 
Learning Systems (RFMLS) program contract N00164-18-R-WQ80. All the code 
released here is unclassified and the Government has unlimited rights 
to the code.
'''

import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import collections
import psutil
import threading


from pynvml import (nvmlInit,
                     nvmlDeviceGetCount,
                     nvmlDeviceGetHandleByIndex,
                     nvmlDeviceGetUtilizationRates,
                     nvmlDeviceGetName)

def gpu_info():
    "Returns a tuple of (GPU ID, GPU Description, GPU % Utilization)"
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    info = []
    for i in range(0, deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        util = nvmlDeviceGetUtilizationRates(handle)
        desc = nvmlDeviceGetName(handle)
        info.append((i, desc, util.gpu)) #['GPU %i - %s' % (i, desc)] = util.gpu
    return info

utils = []

class SysMonitor(threading.Thread):
    shutdown = False

    def __init__(self):
        self.utils = collections.defaultdict(list)
        self.dt = []
        self.start_time = time.time()
        self.duration = 0
        threading.Thread.__init__(self)

    def run(self):
        utils = []
        print('Running SysMonitor')
        while not self.shutdown:
            self.dt.append(datetime.datetime.now())
            self.utils['gpu'].append([x[2] for x in gpu_info()])
            self.utils['io'].append(psutil.disk_io_counters().read_bytes/1000000)
            self.utils['cpu'].append(psutil.cpu_percent())
            time.sleep(.1)

    def stop(self):
        self.shutdown = True
        self.duration = time.time() - self.start_time

    def plot(self, title, vert=False):
        if not self.utils:
            print "Nothing to plot here."
            exit(1)
        if vert:
            fig, ax = plt.subplots(2, 1, figsize=(15, 6))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, size=24)
        ax[0].title.set_text('GPU Utilization')
        for i in range(len(self.utils['gpu'][0])):
            ax[0].plot([u[i] for u in self.utils['gpu']], label="gpu:%d"%i)
        ax[0].set_ylim([0, 100])
        ax[1].title.set_text('CPU')
        ax[1].plot(self.utils['cpu'])
        plt.legend(loc='best')
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])
        plt.savefig('./result/%s.png' % title)
