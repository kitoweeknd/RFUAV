"""
Array Memory Pool
~~~~~~~~~~~~~~~~~

A zero-copy NumPy array memory pool that eliminates repeated allocation /
deallocation cycles for the most frequently sized temporary arrays used
inside the signal-processing pipeline.

Design goals
------------
1. **Zero-copy view reuse** – buffers are reused without touching the
   underlying data; callers receive a view into a pooled backing store.
2. **Power-of-two bucketing** – sizes are bucketed to the next power of two,
   so any request ≤ N bytes hits the same pool (avoids fragmentation).
3. **LRU eviction with soft limit** – the pool grows on demand but refuses to
   exceed ``_max_total_bytes``; least-recently-used buffers are returned to
   Python's GC when evicted.
4. **Thread-safe** – all public methods acquire the lock; the lock is
   a re-entrant (same-thread nested) lock so nested allocations inside a
   callback are safe.
5. **Memorypressure callback** – a caller can register
   ``on_pressure(used_bytes, limit_bytes)`` that fires when the pool exceeds
   80 % of the limit, allowing the application to e.g. drop a cache tier.

Usage
-----
    pool = ArrayPool(max_total_bytes=256 * 1024 * 1024)  # 256 MiB pool

    # Acquire a reusable buffer (cleared to zeros automatically)
    buf = pool.acquire(shape=(1024, 512), dtype=np.complex128)

    # Return it when done
    pool.release(buf)

    # For scatter/gather patterns (fixed size, high churn):
    pool.register_fixed("hann_window", shape=(4096,), dtype=np.float64)

    # Inspect health
    stats = pool.stats()     # dict with used_bytes, n_buffers, hit_rate
    pool.summarize()        # print human-readable report
"""

from __future__ import annotations

import gc
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------

def _next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    n = max(1, n)
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def _estimate_bytes(shape: tuple[int, ...], dtype: np.dtype) -> int:
    """Conservative byte estimate for an array."""
    # Normalise so np.complex64 (a type) or np.dtype(np.complex64) (an instance) both work.
    dtype = np.dtype(dtype)
    items = max(1, int(np.prod(shape)))
    return items * dtype.itemsize


# ----------------------------------------------------------------------
# Bucket entry
# ----------------------------------------------------------------------

@dataclass
class _BucketEntry:
    """One pooled buffer."""
    arr: np.ndarray           # The live buffer
    last_touch: float = 0.0   # Monotonic timestamp (time.monotonic())
    refcount: int = 0         # How many callers currently hold this entry
    fixed_key: str | None = None  # None = generic bucket; else a registered fixed slot
    epoch: int = 0            # Bumped on every acquire/release so stale GC finalizers no-op


# ----------------------------------------------------------------------
# Pool statistics
# ----------------------------------------------------------------------

@dataclass
class PoolStats:
    total_bytes: int
    n_buffers: int
    hit_rate: float        # 0..1 fraction
    n_misses: int
    n_hits: int
    pressure_pct: float    # 0..100 %
    buckets: dict[str, int]  # "key(size)" -> count in that bucket


# ----------------------------------------------------------------------
# Main pool class
# ----------------------------------------------------------------------

class ArrayPool:
    """Thread-safe memory pool for NumPy arrays.

    Parameters
    ----------
    max_total_bytes : int
        Hard upper bound on the total bytes held by the pool.  Defaults to
        256 MiB.  When the pool exceeds 80 % of this limit the
        ``on_pressure`` callback (if registered) is invoked.
    """

    # Default maximum pool size (256 MiB) — large enough for the viewer pipeline.
    DEFAULT_MAX_BYTES = 256 * 1024 * 1024

    def __init__(self, max_total_bytes: int = DEFAULT_MAX_BYTES):
        self._max_bytes = int(max_total_bytes)
        self._pressure_threshold = 0.80  # fire callback when >= 80 % full
        self._on_pressure: Callable[[int, int], None] | None = None

        # Bucket key = f"{fixed_key}|{dtype_str}|{size_power2}"
        self._buckets: dict[str, list[_BucketEntry]] = {}
        self._total_bytes = 0          # evictable (bucket) bytes only
        self._fixed_bytes = 0          # never-evicted fixed-slot bytes
        self._lock = threading.RLock()

        # Hit / miss counters
        self._hits = 0
        self._misses = 0

        # Registered fixed-size slots (never evicted from the pool)
        self._fixed: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_pressure_callback(self, cb: Callable[[int, int], None]) -> None:
        """Register a callback ``cb(used_bytes, max_bytes)`` called at 80 % pressure."""
        self._on_pressure = cb

    def _tracked_view(self, entry: "_BucketEntry") -> np.ndarray:
        """Return a view into ``entry`` and auto-decrement its refcount on GC.

        Calling ``release`` explicitly is still supported and improves reuse
        promptness, but many call sites forget to. Registering a weakref
        finalizer on the returned view guarantees the refcount is eventually
        restored when the caller drops the array, so buffers are not pinned
        forever (which would degrade the pool to a 100 %-miss allocator).

        The finalizer captures the entry's ``epoch`` at hand-out time and only
        releases when it still matches. Any explicit ``release`` or subsequent
        ``acquire`` bumps the epoch, so a stale finalizer firing after the
        entry was re-used is a safe no-op (avoids double-freeing a live entry).
        """
        gen = entry.epoch
        view = entry.arr.view()

        def _on_gc(entry=entry, gen=gen, lock=self._lock):
            with lock:
                if entry.epoch == gen and entry.refcount > 0:
                    entry.refcount = 0
                    entry.epoch += 1

        weakref.finalize(view, _on_gc)
        return view

    def acquire(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        fixed_key: str | None = None,
        zero: bool = True,
    ) -> np.ndarray:
        """Return a view into a pooled buffer.

        If a compatible pooled buffer exists it is reused (zeroed if ``zero=True``);
        otherwise a new one is allocated and inserted into the pool.

        The returned array is a **view** into the pooled backing store.  Mutating
        it is safe and does not affect any other live view as long as the slice
        stays within the allocated shape.  The pool tracks the buffer internally;
        the caller must ``release(buf)`` when done to keep the reference count
        correct.

        Parameters
        ----------
        shape      : tuple[int, ...]   desired array shape
        dtype      : np.dtype          element dtype
        fixed_key  : str | None         name for a registered fixed-size slot
        zero      : bool               zero the buffer before returning it

        Returns
        -------
        np.ndarray (view into pooled store)
        """
        with self._lock:
            dtype = np.dtype(dtype)  # normalise type → dtype instance
            key = self._make_key(dtype, shape, fixed_key)
            sz = _estimate_bytes(shape, dtype)

            # Try to reuse an existing entry
            if key in self._buckets:
                for entry in self._buckets[key]:
                    if entry.refcount == 0 and entry.arr.shape == shape and entry.arr.dtype == dtype:
                        entry.refcount = 1
                        entry.epoch += 1
                        entry.last_touch = time_monotonic()
                        if zero:
                            entry.arr.fill(0)
                        self._hits += 1
                        return self._tracked_view(entry)

            # No hit – allocate
            self._misses += 1
            arr = np.empty(shape, dtype=dtype)
            if zero:
                arr.fill(0)

            # Evict if over budget (skip fixed slots)
            self._maybe_evict(sz)

            entry = _BucketEntry(
                arr=arr,
                last_touch=time_monotonic(),
                refcount=1,
                fixed_key=fixed_key,
            )
            self._buckets.setdefault(key, []).append(entry)
            self._total_bytes += sz

            # Pressure check
            self._check_pressure()

            return self._tracked_view(entry)

    def acquire_like(self, template: np.ndarray, zero: bool = True) -> np.ndarray:
        """Acquire a buffer with the same shape & dtype as ``template``."""
        return self.acquire(template.shape, template.dtype, zero=zero)

    def release(self, arr: np.ndarray) -> None:
        """Decrement the reference count of a pooled buffer.

        Calling ``release`` is optional – the pool also reclaims entries
        automatically via weak-reference callbacks when arrays go out of
        scope.  Calling it explicitly can improve reuse rates.
        """
        with self._lock:
            for bucket in self._buckets.values():
                for entry in bucket:
                    if np.shares_memory(entry.arr, arr):
                        entry.refcount = max(0, entry.refcount - 1)
                        entry.epoch += 1  # invalidate this hand-out's GC finalizer
                        return  # found – done

    def register_fixed(self, key: str, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Register a named fixed buffer (never evicted, shared across calls).

        Returns the (zero-filled) array.  Repeated calls with the same ``key``
        return the cached buffer unchanged.
        """
        with self._lock:
            if key in self._fixed:
                return self._fixed[key]
            arr = np.zeros(shape, dtype=np.dtype(dtype))
            self._fixed[key] = arr
            # Fixed buffers are never evicted, so track them separately from the
            # evictable bucket budget. Counting them in `_total_bytes` would let
            # accumulating windows shrink the reusable pool and trigger
            # premature eviction of real buffers.
            sz = _estimate_bytes(shape, dtype)
            self._fixed_bytes += sz
            return arr

    def get_fixed(self, key: str) -> np.ndarray | None:
        """Return the registered fixed buffer, or None."""
        return self._fixed.get(key)

    def release_all(self) -> None:
        """Drop every buffer in the pool; force garbage collection."""
        with self._lock:
            self._buckets.clear()
            self._fixed.clear()
            self._total_bytes = 0
            self._fixed_bytes = 0
        gc.collect()

    def stats(self) -> PoolStats:
        """Return pool usage statistics."""
        with self._lock:
            total = self._total_bytes + self._fixed_bytes
            n = sum(len(v) for v in self._buckets.values())
            hits = self._hits
            misses = self._misses
            total_req = hits + misses
            hit_rate = hits / total_req if total_req > 0 else 0.0
            pressure = (total / self._max_bytes * 100) if self._max_bytes > 0 else 0.0

            bucket_counts = {}
            for key, entries in self._buckets.items():
                bucket_counts[key] = len(entries)

            return PoolStats(
                total_bytes=total,
                n_buffers=n,
                hit_rate=hit_rate,
                n_misses=misses,
                n_hits=hits,
                pressure_pct=pressure,
                buckets=bucket_counts,
            )

    def summarize(self) -> None:
        """Print a human-readable pool report."""
        s = self.stats()
        print(f"[ArrayPool] {s.n_buffers} buffers | {s.total_bytes / 1024 / 1024:.1f} MiB used "
              f"| pressure {s.pressure_pct:.1f}% | hit rate {s.hit_rate * 100:.1f}%")
        for key, cnt in sorted(s.buckets.items(), key=lambda x: -x[1]):
            total_sz = sum(
                _estimate_bytes(e.arr.shape, e.arr.dtype)
                for e in self._buckets[key]
            )
            print(f"  {key}: {cnt} buffers ({total_sz / 1024:.1f} KiB)")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, dtype: np.dtype, shape: tuple[int, ...], fixed_key: str | None) -> str:
        dtype = np.dtype(dtype)  # normalise type → dtype instance
        sz = _estimate_bytes(shape, dtype)
        p2 = _next_power_of_two(sz)
        if fixed_key:
            return f"{fixed_key}|{dtype.name}|{p2}"
        return f"{dtype.name}|{p2}"

    def _maybe_evict(self, incoming_sz: int) -> None:
        """Evict LRU entries until there is room for ``incoming_sz`` more bytes."""
        while self._total_bytes + incoming_sz > self._max_bytes:
            # Collect all evictable entries (refcount == 0, not a fixed slot)
            candidates = []
            for bucket in self._buckets.values():
                for entry in bucket:
                    if entry.refcount == 0 and entry.fixed_key is None:
                        candidates.append(entry)

            if not candidates:
                break  # nothing left to evict

            # Sort by last_touch (oldest first)
            candidates.sort(key=lambda e: e.last_touch)
            victim = candidates[0]

            # Remove from its bucket
            sz = _estimate_bytes(victim.arr.shape, victim.arr.dtype)
            self._total_bytes -= sz
            for bucket in list(self._buckets.values()):
                if victim in bucket:
                    bucket.remove(victim)
                    break

            # Let GC reclaim the backing store
            del victim

    def _check_pressure(self) -> None:
        if (
            self._on_pressure is not None
            and (self._total_bytes + self._fixed_bytes) / self._max_bytes >= self._pressure_threshold
        ):
            self._on_pressure(self._total_bytes + self._fixed_bytes, self._max_bytes)


def time_monotonic() -> float:
    """Monotonic timestamp – module-level so the import is only done once."""
    import time as _time
    return _time.monotonic()


# ----------------------------------------------------------------------
# Module-level singleton
# ----------------------------------------------------------------------

_POOL: ArrayPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool(max_bytes: int = ArrayPool.DEFAULT_MAX_BYTES) -> ArrayPool:
    """Return the module-level pool (create it on first call)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ArrayPool(max_total_bytes=max_bytes)
        return _POOL


def reset_pool() -> None:
    """Destroy and recreate the module-level pool."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.release_all()
        _POOL = None


# ----------------------------------------------------------------------
# Convenience helpers used throughout the viewer
# ----------------------------------------------------------------------

def pooled_empty(shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Allocate a zeroed buffer from the global pool."""
    return get_pool().acquire(shape, dtype, zero=True)


def pooled_like(template: np.ndarray) -> np.ndarray:
    """Allocate a zeroed buffer matching ``template`` shape & dtype."""
    return get_pool().acquire_like(template, zero=True)


def release_array(arr: np.ndarray) -> None:
    """Return ``arr`` to the global pool."""
    get_pool().release(arr)