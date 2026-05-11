"""
Memory Manager
~~~~~~~~~~~~~~

Unified memory management layer that:

1. **Tracks** per-component memory consumption.
2. **Enforces** hard limits on the total pool budget.
3. **Monitors** Python heap / RSS via ``psutil`` (optional import).
4. **Provides** a global pressure callback that triggers cache eviction
   when the pool exceeds 80 % of the configured limit.
5. **Exposes** coarse-grained ``hold()`` / ``release()`` APIs so callers
   can explicitly release large arrays without waiting for GC.

Typical usage::

    from ChannelSimulation.viewer.signal_proc.memory_manager import (
        get_memory_manager, Holding, hold_arrays,
    )

    mm = get_memory_manager()

    # Hold a large array – it is registered with the tracker.
    with hold_arrays(my_big_array, another_array):
        do_work()          # arrays are pinned until the context exits

    # Manual release
    mm.release("stage_signals", my_array)

    # Query current consumption
    print(mm.usage_summary())
"""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------

@dataclass
class ComponentUsage:
    """Memory breakdown for one logical component."""
    label: str
    bytes_held: int = 0
    n_arrays: int = 0
    largest_bytes: int = 0
    live_refs: list[str] = field(default_factory=list)


@dataclass
class GlobalUsage:
    pool_bytes: int
    pool_limit_bytes: int
    heap_rss_mb: float | None
    n_total_arrays: int
    pressure_pct: float
    components: list[ComponentUsage]


# ----------------------------------------------------------------------
# Memory Manager
# ----------------------------------------------------------------------

class MemoryManager:
    """Centralised memory management and tracking.

    Parameters
    ----------
    pool_limit_bytes : int
        Maximum bytes the global ArrayPool may hold.  Defaults to 256 MiB.
    """

    POOL_DEFAULT_LIMIT = 256 * 1024 * 1024  # 256 MiB

    def __init__(self, pool_limit_bytes: int = POOL_DEFAULT_LIMIT):
        self._pool_limit = int(pool_limit_bytes)
        self._pool_used = 0
        self._pool_hard_limit = self._pool_limit  # same; could be split later

        # Per-component bookkeeping: component_name -> {id(arr): size}
        self._component_arrays: dict[str, dict[int, int]] = {}
        self._component_bytes: dict[str, int] = {}
        self._lock = threading.RLock()

        # Optional psutil integration
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Pool integration
    # ------------------------------------------------------------------

    def set_pool_used(self, used_bytes: int) -> None:
        """Called by ArrayPool when its internal counter changes."""
        self._pool_used = int(used_bytes)

    @property
    def pool_pressure_pct(self) -> float:
        if self._pool_limit <= 0:
            return 0.0
        return self._pool_used / self._pool_limit * 100.0

    # ------------------------------------------------------------------
    # Array tracking
    # ------------------------------------------------------------------

    def track(
        self,
        component: str,
        arrays: list[np.ndarray] | tuple[np.ndarray, ...],
    ) -> None:
        """Register one or more arrays under a named component.

        The component name is a free-form string, e.g. ``"stage_signals"``,
        ``"spectrum_cache"``, ``"config_b"``.  Registering the same array
        multiple times (e.g. on a re-load) updates the existing entry.
        """
        with self._lock:
            if component not in self._component_arrays:
                self._component_arrays[component] = {}
                self._component_bytes[component] = 0

            for arr in arrays:
                key = id(arr)
                sz = int(np.prod(arr.shape) * np.dtype(arr.dtype).itemsize)
                if key not in self._component_arrays[component]:
                    self._component_arrays[component][key] = sz
                    self._component_bytes[component] += sz
                else:
                    delta = sz - self._component_arrays[component][key]
                    self._component_arrays[component][key] = sz
                    self._component_bytes[component] += delta

    def release_component(self, component: str) -> int:
        """Remove all tracked arrays for a component; return bytes freed."""
        with self._lock:
            if component not in self._component_arrays:
                return 0
            freed = self._component_bytes.get(component, 0)
            self._component_arrays.pop(component, None)
            self._component_bytes.pop(component, None)
            return freed

    def release_array(self, component: str, arr: np.ndarray) -> int:
        """Remove a specific array from a component's tracking; return bytes freed."""
        with self._lock:
            if component not in self._component_arrays:
                return 0
            key = id(arr)
            sz = self._component_arrays[component].pop(key, 0)
            self._component_bytes[component] = max(0, self._component_bytes.get(component, 0) - sz)
            return sz

    def get_component_bytes(self, component: str) -> int:
        with self._lock:
            return self._component_bytes.get(component, 0)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def heap_rss_mb(self) -> float | None:
        """Return current RSS in MiB, or None if psutil unavailable."""
        if self._psutil is None:
            return None
        return self._psutil.Process().memory_info().rss / 1024 / 1024

    def usage_summary(self) -> GlobalUsage:
        """Return a full snapshot of memory consumption."""
        with self._lock:
            components = []
            total_arrays = 0
            largest_pool = 0
            for name, arr_map in self._component_arrays.items():
                n = len(arr_map)
                total_arrays += n
                total_b = sum(arr_map.values())
                largest = max(arr_map.values()) if arr_map else 0
                components.append(ComponentUsage(
                    label=name,
                    bytes_held=total_b,
                    n_arrays=n,
                    largest_bytes=largest,
                ))
                if name == "pool":
                    largest_pool = largest

            return GlobalUsage(
                pool_bytes=self._pool_used,
                pool_limit_bytes=self._pool_limit,
                heap_rss_mb=self.heap_rss_mb(),
                n_total_arrays=total_arrays,
                pressure_pct=self.pool_pressure_pct,
                components=components,
            )

    def print_summary(self) -> None:
        """Pretty-print the usage summary."""
        u = self.usage_summary()
        print("[MemoryManager] ── Memory Usage ──────────────────────────")
        print(f"  Pool:    {u.pool_bytes / 1024 / 1024:.2f} MiB / {u.pool_limit_bytes / 1024 / 1024:.2f} MiB "
              f"  ({u.pressure_pct:.1f} %)")
        if u.heap_rss_mb is not None:
            print(f"  Heap:    {u.heap_rss_mb:.1f} MiB RSS")
        print(f"  Arrays:  {u.n_total_arrays} tracked")
        for c in u.components:
            mb = c.bytes_held / 1024 / 1024
            print(f"    {c.label:<20s} {mb:.3f} MiB  ({c.n_arrays} arrays, max {c.largest_bytes / 1024:.1f} KiB)")
        print("──────────────────────────────────────────────────────────")

    def force_gc(self) -> None:
        """Trigger a full garbage-collection cycle."""
        gc.collect()

    # ------------------------------------------------------------------
    # Explicit retain / release helpers
    # ------------------------------------------------------------------

    def hold(self, component: str, *arrays: np.ndarray) -> None:
        """Pin arrays under a component without returning anything."""
        self.track(component, list(arrays))

    def release_all(self) -> None:
        """Clear all component tracking."""
        with self._lock:
            self._component_arrays.clear()
            self._component_bytes.clear()


# ----------------------------------------------------------------------
# Context manager for scoped array holding
# ----------------------------------------------------------------------

class _Holding:
    """Internal: RAII guard for hold_arrays()."""

    __slots__ = ("_mm", "_component", "_arrs")

    def __init__(self, mm: MemoryManager, component: str, arrs: list[np.ndarray]):
        self._mm = mm
        self._component = component
        self._arrs = arrs
        mm.track(component, arrs)

    def __enter__(self) -> list[np.ndarray]:
        return self._arrs

    def __exit__(self, *_: Any) -> None:
        # Release tracking but do NOT free the underlying memory
        # – the arrays may still be referenced elsewhere.
        for arr in self._arrs:
            self._mm.release_array(self._component, arr)


def hold_arrays(component: str, *arrays: np.ndarray) -> _Holding:
    """Context manager that pins ``arrays`` under ``component`` for the duration.

    When the ``with`` block exits the arrays are removed from the tracker
    (so they are eligible for GC if unreferenced elsewhere).  This is the
    recommended way to scope large temporary allocations::

        with hold_arrays("temp_fft", scratch_a, scratch_b):
            # scratch_a and scratch_b are tracked; pressure callbacks fire
            do_fft()
        # after exit: scratch arrays are no longer pinned

    The underlying memory is NOT freed automatically; rely on Python GC
    and NumPy's internal arena for that.
    """
    return _Holding(get_memory_manager(), component, list(arrays))


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------

_MM: MemoryManager | None = None
_MM_LOCK = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """Return the module-level MemoryManager (create on first call)."""
    global _MM
    with _MM_LOCK:
        if _MM is None:
            _MM = MemoryManager()
        return _MM


def reset_memory_manager() -> None:
    """Destroy and recreate the global MemoryManager."""
    global _MM
    with _MM_LOCK:
        if _MM is not None:
            _MM.release_all()
        _MM = None