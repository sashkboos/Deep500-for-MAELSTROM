import time
import enum
import statistics
import abc
from collections import defaultdict
from typing import List, Any, NamedTuple
try:
    import wandb
except ImportError:
    wandb = None
try:
    import mlflow
except ImportError:
    mlflow = None


class TimeType(enum.Enum):
    """Different quantity types for semantic timing."""
    EPOCH = enum.auto()
    BATCH = enum.auto()
    FORWARD = enum.auto()
    BACKWARD = enum.auto()
    COMM = enum.auto()
    IO = enum.auto()

    # Special cases.
    OTHER = enum.auto()
    ALL = enum.auto()


class TimeStats(NamedTuple):
    """Statistics for times."""
    min: float
    mean: float
    median: float
    max: float
    stdev: float

    def __str__(self) -> str:
        s = f'min: {self.min:.5f}'
        s += f' mean: {self.mean:.5f}'
        s += f' median: {self.median:.5f}'
        s += f' max: {self.max:.5f}'
        s += f' stdev: {self.stdev:.5f}'
        return s


def _construct_stats(l: List[float]) -> TimeStats:
    """Compute the statistics for TimeStats from a list of times."""
    return TimeStats(
        min=min(l),
        mean=statistics.mean(l),
        median=statistics.median(l),
        max=max(l),
        stdev=statistics.stdev(l) if len(l) > 1 else 0.0
    )


class _TimerImpl(abc.ABC):
    """Base class for timer implementations."""

    @abc.abstractmethod
    def start(self) -> Any:
        """Start timing at the current call point."""

    @abc.abstractmethod
    def end(self) -> Any:
        """End timing at the current call point."""

    @abc.abstractmethod
    def get_time(self, start: Any, end: Any) -> float:
        """Return the time elapsed between start and end."""


class _CPUTimerImpl(_TimerImpl):
    """Time things on a CPU."""

    def start(self):
        return time.perf_counter()

    def end(self):
        return time.perf_counter()

    def get_time(self, start, end) -> float:
        return end - start


class Timer:
    """Time segments of code.

    Timing with different types of keys may be nested, but the same
    type cannot be.

    """

    def __init__(self, timer_impl: Any = _CPUTimerImpl) -> None:
        self.timer_impl = timer_impl()
        self.times = defaultdict(list)  # Recorded times.
        self.starts = defaultdict(list)  # Timer starts.
        self.ends = defaultdict(list)  # Timer ends.

    @staticmethod
    def _assert_not_all(key: TimeType) -> None:
        """Raise an error if key is TimeType.ALL."""
        if key == TimeType.ALL:
            raise ValueError('Cannot use TimeType.ALL here')

    def _complete_timing(self, key: TimeType, ignore_imbalanced=False) -> None:
        """Complete all outstanding timers for key and record times."""
        if not ignore_imbalanced and (len(self.starts[key]) != len(self.ends[key])):
            raise RuntimeError(f'Unbalanced start/end for region type {key}')
        if len(self.ends[key]) > len(self.starts[key]):
            raise RuntimeError(f'More ends than starts for region type {key}')
        # As there is no nesting in key types, starts/ends are paired
        # by their position.
        # When ignoring imbalanced cases, extra starts are ignored.
        for start, end in zip(self.starts[key], self.ends[key]):
            t = self.timer_impl.get_time(start, end)
            self.times[key].append(t)
        if ignore_imbalanced:
            # Only clear the completed starts.
            num_done = len(self.ends[key])
            self.starts[key] = self.starts[key][num_done:]
        else:
            self.starts[key].clear()
        self.ends[key].clear()

    def _complete_all_timing(self, ignore_imbalanced=False) -> None:
        """Complete all timing of all types."""
        if not ignore_imbalanced and (self.starts.keys() != self.ends.keys()):
            raise RuntimeError('Unmatched starts/ends')
        for key in self.starts.keys():
            self._complete_timing(key, ignore_imbalanced=ignore_imbalanced)

    def start(self, key: TimeType) -> None:
        """Start timing a region.

        @param key Type of region to time.

        """
        self._assert_not_all(key)
        # Check for nesting.
        if len(self.starts[key]) != len(self.ends[key]):
            raise RuntimeError(f'Detected nesting for timing region type {key}')
        self.starts[key].append(self.timer_impl.start())

    def end(self, key: TimeType) -> None:
        """End timing a region.

        @param key Type of region to complete timing for.

        """
        self._assert_not_all(key)
        if len(self.starts[key]) != (len(self.ends[key]) + 1):
            raise RuntimeError(f'Unbalanced end for timing region type {key}')
        self.ends[key].append(self.timer_impl.end())

    def get_time(self, key: TimeType) -> List[float]:
        """Return a list of the times spent in a region of type key.

        If no times are recorded for key, returns an empty list.

        """
        self._complete_timing(key)
        return self.times[key]

    def complete_all(self) -> None:
        """Ensure all timing is complete."""
        self._complete_all_timing(ignore_imbalanced=True)

    def get_time_stats(self, key: TimeType) -> TimeStats:
        """Return statistics for time spent in a region of type key."""
        return _construct_stats(self.get_time(key))

    def print_all_time_stats(self, file: Any = None) -> None:
        """Print statistics for all time regions.

        @param file Optional text stream to print to.

        """
        self._complete_all_timing()
        for key in self.times:
            stats = self.get_time_stats(key)
            print(key.name, file=file)
            print(stats, file=file)
        print('---', file=file)

    def save_all_time_stats(self, filename: str) -> None:
        """Save statistics for all time regions to a file.

        @param filename Name of the file. Must not exist.

        """
        with open(filename, 'x', encoding='utf-8') as f:
            self.print_all_time_stats(file=f)

    def log_wb_all(self, prefix: str = '') -> None:
        """Save means for all time regions to Weights & Biases.

        @param prefix Prefix string for all logging keys.

        """
        if wandb is None:
            raise RuntimeError('No Weights & Biases support')
        self._complete_all_timing()
        means = {}
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        for key in self.times:
            stats = self.get_time_stats(key)
            means[prefix + key.name] = stats.mean
        print(means)
        wandb.log(means)

    def log_mlflow_all(self, prefix: str = '') -> None:
        """Save means for all time regions to MLFlow.

        @param prefix Prefix string for all logging keys.

        """
        if mlflow is None:
            raise RuntimeError('No MLFlow support')
        self._complete_all_timing()
        means = {}
        if prefix and not prefix.endswith('/'):
            prefix += '/'
        for key in self.times:
            stats = self.get_time_stats(key)
            means[prefix + key.name] = stats.mean
        mlflow.log_metrics(means)
