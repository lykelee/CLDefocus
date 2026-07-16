from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, ClassVar, Optional


@dataclass
class TimerSettings:
    """Configuration for EasyTimer behavior."""

    output: Callable[[str], None] = field(default_factory=lambda: print)
    """Output function (e.g., print, logger.info)."""

    format_enter: Callable[[str, int], str] | None = None
    """Format function for enter message: (label, depth) -> str. None to skip enter log."""

    format_exit: Callable[[str, float, int], str] | None = None
    """Format function for exit message: (label, elapsed, depth) -> str. None uses default."""

    indent: str = "  "
    """Indentation string per nesting level."""

    precision: int = 3
    """Decimal places for elapsed time."""

    disabled: bool = False
    """Globally disable all timers."""

    log_enter: bool = True
    """Whether to log on entering the timer context."""

    track_live: bool = False
    """Whether to enable live elapsed time tracking (slight overhead)."""

    def copy(self, **overrides) -> TimerSettings:
        """Create a copy with optional overrides."""
        return TimerSettings(
            output=overrides.get("output", self.output),
            format_enter=overrides.get("format_enter", self.format_enter),
            format_exit=overrides.get("format_exit", self.format_exit),
            indent=overrides.get("indent", self.indent),
            precision=overrides.get("precision", self.precision),
            disabled=overrides.get("disabled", self.disabled),
            log_enter=overrides.get("log_enter", self.log_enter),
            track_live=overrides.get("track_live", self.track_live),
        )


class EasyTimer:
    """
    Configurable timer context manager with nesting support.

    Usage:
        # Simple (backward compatible)
        with easy_timer("task"):
            do_stuff()

        # Nested timers with automatic indentation
        with easy_timer("outer"):
            with easy_timer("inner"):
                pass

        # Configure globally
        EasyTimer.configure(output=logger.info, log_enter=True)

        # Per-call overrides
        with easy_timer("task", output=logger.debug):
            ...

        # Access elapsed time after block
        with easy_timer("task") as t:
            do_stuff()
        print(t.elapsed)

        # Live elapsed time (must enable track_live)
        with easy_timer("task", track_live=True) as t:
            do_step_1()
            print(f"Step 1: {t.elapsed:.2f}s")
            do_step_2()
    """

    settings: ClassVar[TimerSettings] = TimerSettings()
    _stack: ClassVar[ContextVar[list[str]]] = ContextVar("easy_timer_stack", default=[])

    def __init__(
        self,
        label: str = "",
        disable: bool = False,
        **overrides,
    ):
        """
        Create a timer context manager.

        Args:
            label: Label for this timer.
            disable: Disable this specific timer (overrides global setting).
            **overrides: Per-call setting overrides (output, format_enter, format_exit,
                         indent, precision, log_enter, track_live).
        """
        self.label = label
        self._disable = disable
        self._overrides = overrides

        self._start: float | None = None
        self._elapsed: float | None = None
        self._depth: int = 0

    def _get_setting(self, key: str):
        """Get setting value, preferring per-call override."""
        if key in self._overrides:
            return self._overrides[key]
        return getattr(self.settings, key)

    def _format_enter(self, label: str, depth: int) -> str:
        """Format enter message."""
        custom_format = self._get_setting("format_enter")
        if custom_format is not None:
            return custom_format(label, depth)
        indent = self._get_setting("indent") * depth
        label_str = f"[{label}] " if label else ""
        return f"{indent}{label_str}Starting..."

    def _format_exit(self, label: str, elapsed: float, depth: int) -> str:
        """Format exit message."""
        custom_format = self._get_setting("format_exit")
        if custom_format is not None:
            return custom_format(label, elapsed, depth)
        indent = self._get_setting("indent") * depth
        precision = self._get_setting("precision")
        label_str = f"[{label}] " if label else ""
        return f"{indent}{label_str}Elapsed: {elapsed:.{precision}f} seconds"

    def __enter__(self) -> EasyTimer:
        if self._disable or self._get_setting("disabled"):
            return self

        # Track nesting depth
        stack = self._stack.get()[:]  # Copy to avoid mutation issues
        self._depth = len(stack)
        stack.append(self.label)
        self._stack.set(stack)

        # Log enter if enabled
        if self._get_setting("log_enter"):
            msg = self._format_enter(self.label, self._depth)
            self._get_setting("output")(msg)

        # Start timing
        self._start = time.perf_counter()

        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self._start is None:
            return  # Timer was disabled

        # Calculate elapsed time
        self._elapsed = time.perf_counter() - self._start

        # Pop from stack
        stack = self._stack.get()[:]
        if stack:
            stack.pop()
        self._stack.set(stack)

        # Log exit
        msg = self._format_exit(self.label, self._elapsed, self._depth)
        self._get_setting("output")(msg)

        return False

    @property
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.

        After the context block: returns final elapsed time.
        During the context block (requires track_live=True): returns live elapsed time.
        Before entering: returns 0.0.
        """
        if self._elapsed is not None:
            return self._elapsed
        if self._start is not None and self._get_setting("track_live"):
            return time.perf_counter() - self._start
        if self._start is not None:
            # track_live is False, but we're inside the block
            return 0.0  # Indicate live tracking is disabled
        return 0.0

    @classmethod
    def configure(cls, **kwargs) -> None:
        """
        Configure global default settings.

        Args:
            output: Output function (e.g., print, logger.info).
            format_enter: Format function for enter: (label, depth) -> str.
            format_exit: Format function for exit: (label, elapsed, depth) -> str.
            indent: Indentation string per nesting level.
            precision: Decimal places for elapsed time.
            disabled: Globally disable all timers.
            log_enter: Whether to log on entering the timer context.
            track_live: Whether to enable live elapsed time tracking.
        """
        cls.settings = cls.settings.copy(**kwargs)

    @classmethod
    def reset_settings(cls) -> None:
        """Reset settings to defaults."""
        cls.settings = TimerSettings()

    @classmethod
    def current_depth(cls) -> int:
        """Get current nesting depth."""
        return len(cls._stack.get())


def easy_timer(label: str = "", disable: bool = False, **kwargs) -> EasyTimer:
    """
    Create a timer context manager. Backward compatible wrapper for EasyTimer.

    Args:
        label: Label for this timer.
        disable: Disable this specific timer.
        **kwargs: Setting overrides (output, format_enter, format_exit,
                  indent, precision, log_enter, track_live).

    Returns:
        EasyTimer context manager.

    Examples:
        # Simple usage (unchanged from original)
        with easy_timer("task"):
            do_stuff()

        # Nested timers
        with easy_timer("outer"):
            with easy_timer("inner"):
                pass
        # Output:
        #   [inner] Elapsed: 0.123 seconds
        # [outer] Elapsed: 0.456 seconds

        # With enter logging
        with easy_timer("task", log_enter=True):
            do_stuff()
        # Output:
        # [task] Starting...
        # [task] Elapsed: 0.1234 seconds

        # Custom output
        with easy_timer("task", output=logger.info):
            ...

        # Access elapsed time
        with easy_timer("task") as t:
            do_stuff()
        print(f"Took {t.elapsed}s")
    """
    return EasyTimer(label, disable, **kwargs)


class EasyTimestamp:
    def __init__(self, fn_label_wrapper: Optional[Callable[[str], str]] = None):
        self.last_label = None
        self.last_time = None
        self.lock = threading.Lock()

        self.fn_label_wrapper = fn_label_wrapper

    def __call__(self, label: str | None = None):
        now = time.perf_counter()
        with self.lock:
            msg = str(label)
            if self.last_time is not None:
                elapsed = now - self.last_time
                msg += f" <- {self.last_label}: {elapsed:.6f} seconds"
            if self.fn_label_wrapper is not None:
                msg = self.fn_label_wrapper(msg)
            print(msg)

            if label is None:  # No new stamp
                self.last_label = None
                self.last_time = None

            else:  # New stamp
                self.last_time = now
                self.last_label = label


def easy_timestamp(label: str | None = None, instance: Optional[EasyTimestamp] = None):
    """
    Starts a new timestamp with the given label. If there's a previous timer
    running in the given state (or the default hidden instance), it prints the
    elapsed time with the previous label, then starts a new timer.

    Thread safety is ensured both in the default instance creation and in the
    timing operations.

    Parameters:
      label (str): The label for the new timestamp.
      instance (EasyTimestamp, optional): An EasyTimestamp instance for storing state.
          If not provided, a hidden default instance is used.
    """
    if instance is None:
        if not hasattr(easy_timestamp, "_default_lock"):
            easy_timestamp._default_lock = threading.Lock()
        with easy_timestamp._default_lock:
            if not hasattr(easy_timestamp, "_default_instance"):
                easy_timestamp._default_instance = EasyTimestamp()
            instance = easy_timestamp._default_instance
    instance(label)


def easy_timestamp_set_instance(instance: EasyTimestamp):
    easy_timestamp._default_instance = instance


class Stopwatch:
    """
    NOTE: This is thread-safe.
    """

    def __init__(self):
        self._start = None
        self._last_elapsed = 0.0
        self._lock = threading.Lock()

    def reset(self):
        """
        Stops and initializes the elapsed time.
        """
        with self._lock:
            self._start = None
            self._last_elapsed = 0.0

    def start(self):
        """
        Starts the stopwatch from the previous elapsed time.
        Returns the elapsed time up to the moment this method is called.
        """
        if self._start is None:
            with self._lock:
                self._start = time.perf_counter()
            return self._last_elapsed
        else:
            return self.elapsed

    def stop(self):
        """
        Stops the stopwatch. The elapsed time is kept.
        Returns the elapsed time up to the moment this method is called.
        """
        if self._start is not None:
            with self._lock:
                self._last_elapsed += time.perf_counter() - self._start
                self._start = None

        return self._last_elapsed

    def restart(self):
        """
        Resets and starts the stopwatch.
        Returns the elapsed time up to the moment this method is called.
        """
        t = time.perf_counter()

        if self._start is None:
            elapsed = self._last_elapsed
        else:
            elapsed = self._last_elapsed + t - self._start

        with self._lock:
            self._start = t
            self._last_elapsed = 0.0

        return elapsed

    @property
    def running(self):
        return self._start is not None

    @property
    def elapsed(self):
        if self._start is None:
            return self._last_elapsed
        else:
            return self._last_elapsed + (time.perf_counter() - self._start)
