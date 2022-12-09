from typing import ClassVar

import cupy

from . timer import *
from . timer import _TimerImpl, _CPUTimerImpl
from . timer_cpugpu import CPUGPUTimerBase


class _CuPyGPUTimerImpl(_TimerImpl):
    """Time things on the GPU using CuPy.

    This will synchronize in each call to end.

    """

    MAX_EVENTS: ClassVar[int] = 200  # Max number of CUDA events to create.

    def __init__(self):
        super().__init__()
        # Create pool of events.
        self._events = [cupy.cuda.Event() for _ in range(self.MAX_EVENTS)]

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
        # Events are on the default stream and will synchronize all streams.
        event.synchronize()
        return event

    def get_time(self, start, end) -> float:
        t = cupy.cuda.get_elapsed_time(start, end) / 1000  # ms -> s
        self._events.append(start)
        self._events.append(end)
        return t

class CPUGPUTimer(CPUGPUTimerBase):
    """Timer that supports timing both CPU and GPU runtimes."""

    def __init__(self):
        super().__init__(cpu_timer_impl=_CPUTimerImpl,
                         gpu_timer_impl=_CuPyGPUTimerImpl)
