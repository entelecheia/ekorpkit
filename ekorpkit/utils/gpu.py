import GPUtil
import time
import gc
from threading import Thread
from .notebook import _clear_output


class GPUMon(Thread):
    def __init__(self, delay=10, show_current_time=True, clear_output=True):
        super(GPUMon, self).__init__()
        self.stopped = False
        self.delay = delay  # Time between calls to GPUtil
        self.show_current_time = show_current_time
        self.clear_output = clear_output
        self.start()

    def run(self):
        while not self.stopped:
            if self.clear_output:
                _clear_output()
            if self.show_current_time:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            GPUtil.showUtilization()
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

    @staticmethod
    def gpu_usage(all=False, attrList=None, useOldCode=False):
        return GPUtil.showUtilization(all=all, attrList=attrList, useOldCode=useOldCode)

    @staticmethod
    def get_gpus():
        return GPUtil.getGPUs()

    @staticmethod
    def get_gpu_info():
        gpus = GPUtil.getGPUs()
        gpu_info = []
        for gpu in gpus:
            gpu_info.append(
                {
                    "id": gpu.id,
                    "name": gpu.name,
                    "load": gpu.load,
                    "memory_used": gpu.memoryUsed,
                    "memory_total": gpu.memoryTotal,
                    "temperature": gpu.temperature,
                }
            )
        return gpu_info

    @staticmethod
    def get_first_available(
        order="first",
        maxLoad=0.5,
        maxMemory=0.5,
        attempts=1,
        interval=900,
        verbose=False,
        includeNan=False,
        excludeID=[],
        excludeUUID=[],
    ):
        return GPUtil.getFirstAvailable(
            order=order,
            maxLoad=maxLoad,
            maxMemory=maxMemory,
            attempts=attempts,
            interval=interval,
            verbose=verbose,
            includeNan=includeNan,
            excludeID=excludeID,
            excludeUUID=excludeUUID,
        )

    @staticmethod
    def release_gpu_memory():
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
