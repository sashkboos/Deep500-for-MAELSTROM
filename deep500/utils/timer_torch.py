from typing import ClassVar

import torch

from . timer import *
from . timer import _TimerImpl, _CPUTimerImpl
from . timer_cpugpu import CPUGPUTimerBase


class _TorchGPUTimerImpl(_TimerImpl):
    """Time things on a GPU using PyTorch."""

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


class CPUGPUTimer(CPUGPUTimerBase):
    """Timer that supports timing both CPU and GPU runtimes."""

    def __init__(self):
        super().__init__(cpu_timer_impl=_CPUTimerImpl,
                         gpu_timer_impl=_TorchGPUTimerImpl)
