from typing import List, Any

from . timer import TimeType, TimeStats, Timer

class CPUGPUTimerBase:
    """Timer that supports timing both CPU and GPU runtimes."""

    def __init__(self, cpu_timer_impl: Any, gpu_timer_impl: Any) -> None:
        self.cpu_timer = Timer(timer_impl=cpu_timer_impl)
        self.gpu_timer = Timer(timer_impl=gpu_timer_impl)

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
