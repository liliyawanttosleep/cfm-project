"""限速：跑实验时不占满电脑，用户还能正常使用。

要完成的任务
    任务 1  降低本进程优先级（低于正常），只改这个进程，不改任何系统设置
    任务 2  限制 CPU 线程数，避免 k-means、SVM 占满所有核
    任务 3  占空比：每算完一批，就休息一段时间，让显卡平均只忙 duty 这一比例的时间
    任务 4  温度门槛：显卡超过 max_temp 就暂停，降到 resume_temp 以下再继续
    任务 5  内存门槛：系统可用内存低于 min_free_gb 就暂停，等其他程序释放内存

用法
    from common.throttle import Throttle
    th = Throttle(duty=0.5)
    th.reset()
    for batch in ...:
        ... 算一批 ...
        th.step()   # 在这里休息、检查温度和内存
"""
import os
import subprocess
import time

import psutil
import torch


def limit_threads(n):
    """任务 2：要在 import numpy / sklearn 之前调用才对 OpenMP 完全生效，所以调用方脚本开头也设置了环境变量。"""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(n)
    torch.set_num_threads(n)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except ImportError:
        pass


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.split()[0])
    except Exception:
        return None


class Throttle:
    def __init__(self, duty=0.5, max_temp=72, resume_temp=65, min_free_gb=3.0, threads=4, check_every=10.0, log=print):
        self.duty, self.max_temp, self.resume_temp, self.min_free_gb = duty, max_temp, resume_temp, min_free_gb
        self.check_every, self.log = check_every, log
        self.last_check, self.t0 = 0.0, time.time()
        self.paused = 0.0
        limit_threads(threads)
        try:  # 任务 1
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        except Exception as e:
            log("could not lower priority:", e)

    def reset(self):
        """开始一段新的显卡计算前调用，这样之前在 CPU 上花的时间（k-means、打标签）不会被算成显卡忙碌、再休息一次。"""
        self.t0 = time.time()

    def step(self):
        """在两批之间调用：先按占空比休息，再按温度和内存决定是否暂停。"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        busy = time.time() - self.t0
        if self.duty < 1:  # 任务 3
            time.sleep(busy * (1 / self.duty - 1))
        now = time.time()
        if now - self.last_check >= self.check_every:
            self.last_check = now
            self._wait_if_needed()
        self.t0 = time.time()

    def _wait_if_needed(self):
        t = gpu_temp()
        free = psutil.virtual_memory().available / 2 ** 30
        if (t is None or t < self.max_temp) and free >= self.min_free_gb:
            return
        self.log(f"[throttle] pause: GPU {t} C, free RAM {free:.1f} GB", flush=True)
        start = time.time()
        while True:  # 任务 4、5
            time.sleep(15)
            t = gpu_temp()
            free = psutil.virtual_memory().available / 2 ** 30
            if (t is None or t <= self.resume_temp) and free >= self.min_free_gb:
                break
        self.paused += time.time() - start
        self.log(f"[throttle] resume after {time.time() - start:.0f}s: GPU {t} C, free RAM {free:.1f} GB", flush=True)
