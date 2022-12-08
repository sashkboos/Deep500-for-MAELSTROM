from typing import List, ClassVar

import torch

from . timer import *
from . timer import _TimerImpl



class _GPUTimerImpl(_TimerImpl):
    """Time things on a GPU."""

    MAX_EVENTS: ClassVar[int] = 200  # Max number of CUDA events to create.

    def __init__(self):
        super().__init__()
        # Ensure CUDA is available and initialized.
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError('No CUDA support or GPUs available')
        torch.cuda.init()
        # Create pool of events.
        self._events = [torch.cuda.Event(enable_timing=True) for _
                        in range(self.MAX_EVENTS)]

    def _get_event(self):
        """Get a free CUDA event."""
        if not self._events:
            raise RuntimeError('No more CUDA events, try completing some')
        return self._events.pop()

    def start(self):
        event = self._get_event()
        event.record()
        return event

    def end(self):
        event = self._get_event()
        event.record()
        return event

    def get_time(self, start, end) -> float:
        end.synchronize()
        t = start.elapsed_time(end) / 1000  # ms -> s
        self._events.append(start)
        self._events.append(end)
        return t


class CPUGPUTimer:
    """Timer that supports timing both CPU and GPU runtimes."""

    def __init__(self):
        self.cpu_timer = Timer()
        self.gpu_timer = Timer(timer_impl=_GPUTimerImpl)

    def _do_call(self, f, gpu, *args, **kwargs):
        if gpu:
            f = getattr(self.gpu_timer, f)
        else:
            f = getattr(self.cpu_timer, f)
        return f(*args, **kwargs)

    def start(self, key: TimeType, gpu=False) -> None:
        self._do_call('start', gpu, key)

    def end(self, key: TimeType, gpu=False) -> None:
        self._do_call('end', gpu, key)

    def get_time(self, key: TimeType, gpu=False) -> List[float]:
        return self._do_call('get_time', gpu, key)

    def complete_all(self) -> None:
        self.cpu_timer.complete_all()
        self.gpu_timer.complete_all()

    def get_time_stats(self, key: TimeType, gpu=False) -> TimeStats:
        return self._do_call('get_time_stats', gpu, key)

    def print_all_time_stats(self, file: Any = None) -> None:
        print('CPU', file=file)
        self.cpu_timer.print_all_time_stats(file=file)
        print('GPU', file=file)
        self.gpu_timer.print_all_time_stats(file=file)

    def save_all_time_stats(self, filename: str) -> None:
        with open(filename, 'x', encoding='utf-8') as f:
            self.print_all_time_stats(file=f)

    def log_wb_all(self, prefix: str = '') -> None:
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        self.cpu_timer.log_wb_all(prefix=prefix + 'CPU/')
        self.gpu_timer.log_wb_all(prefix=prefix + 'GPU/')

    def log_mlflow_all(self, prefix: str = '') -> None:
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        self.cpu_timer.log_mlflow_all(prefix=prefix + 'CPU/')
        self.gpu_timer.log_mlflow_all(prefix=prefix + 'GPU/')
