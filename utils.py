from pynvml import *

def find_device():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    infos = []

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        infos.append((i, info.free))

    infos.sort(key=lambda x: -x[1])
    device = infos[0][0]

    nvmlShutdown()

    return device