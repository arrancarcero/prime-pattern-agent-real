"""Streaming primes: block by block, never the whole list in memory.

PrimeStore sieves [lo, hi) in blocks (default 1e8 numbers) and hands each block's primes
(from 5 on) to the caller, who keeps only running totals. With a cache directory it also
writes every 1e9-wide segment to disk once, as its first prime plus uint16 gaps (2 bytes per
prime; prime gaps below 1e12 are under 600), so later passes read instead of re-sieving.

Backends: NumPy (CPU) or CuPy (NVIDIA GPU). The GPU sieve marks multiples with one CUDA
kernel launch per block. Before a GPU run is trusted, cross_check() sieves the same blocks
on both and demands identical primes; if they ever differ the store falls back to the CPU and
says so. Nothing here changes a test: only how fast the same primes are produced.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from ppa_real.core.sieve import START, small_primes

SEGMENT = 10 ** 9           # cache file width
BLOCK = 10 ** 8             # numbers sieved at once

_KERNEL = r'''
extern "C" __global__
void mark(unsigned char* s, const long long* primes, const int np, const long long a,
          const long long len, const long long chunk, const long long nchunks) {
    long long tid = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= (long long)np * nchunks) return;
    long long i = tid / nchunks;
    long long c = tid - i * nchunks;
    long long q = primes[i];
    long long lo = a + c * chunk;
    long long hi = lo + chunk;
    if (hi > a + len) hi = a + len;
    long long start = ((lo + q - 1) / q) * q;
    if (start < q * q) start = q * q;
    for (long long m = start; m < hi; m += q) s[m - a] = 0;
}
'''


def get_backend(gpu: bool):
    """(xp, name). Falls back to NumPy if CuPy or a GPU is not available."""
    if gpu:
        try:
            import cupy as cp
            cp.zeros(1).sum()
            return cp, "GPU (CuPy, " + cp.cuda.runtime.getDeviceProperties(0)["name"].decode() + ")"
        except Exception as e:  # noqa: BLE001
            print(f"GPU not available ({type(e).__name__}: {e}); using CPU")
    return np, "CPU (NumPy)"


def to_numpy(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


class PrimeStore:
    def __init__(self, cache_dir: str | Path | None = None, gpu: bool = False, block: int = BLOCK):
        self.xp, self.backend = get_backend(gpu)
        self.block = block
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self._base = np.array([], dtype=np.int64)
        self._kernel = None
        self.cross_check_result = None

    # ---- sieving -----------------------------------------------------------------
    def _base_primes(self, hi: int) -> np.ndarray:
        need = math.isqrt(hi) + 1
        if len(self._base) == 0 or self._base[-1] < need:
            self._base = small_primes(max(need, 1000)).astype(np.int64)
        return self._base

    def _sieve_block_cpu(self, a: int, b: int) -> np.ndarray:
        s = np.ones(b - a, dtype=bool)
        for q in self._base_primes(b):
            q = int(q)
            if q * q >= b:
                break
            start = max(q * q, ((a + q - 1) // q) * q)
            s[start - a::q] = False
        if a < START:
            s[: START - a] = False
        return np.nonzero(s)[0].astype(np.int64) + a

    def _sieve_block_gpu(self, a: int, b: int):
        cp = self.xp
        if self._kernel is None:
            self._kernel = cp.RawKernel(_KERNEL, "mark")
        base = self._base_primes(b)
        base = base[base * base < b]
        s = cp.ones(b - a, dtype=cp.uint8)
        if len(base):
            chunk = 1 << 16
            nchunks = (b - a + chunk - 1) // chunk
            d_base = cp.asarray(base)
            total = len(base) * nchunks
            threads = 256
            self._kernel(((total + threads - 1) // threads,), (threads,),
                         (s, d_base, np.int32(len(base)), np.int64(a), np.int64(b - a),
                          np.int64(chunk), np.int64(nchunks)))
        if a < START:
            s[: START - a] = 0
        return cp.nonzero(s)[0].astype(cp.int64) + a

    def sieve_block(self, a: int, b: int):
        if self.xp is np:
            return self._sieve_block_cpu(a, b)
        return self._sieve_block_gpu(a, b)

    def cross_check(self, blocks=((10 ** 8, 10 ** 8 + 10 ** 7),)) -> bool:
        """GPU and CPU must produce identical primes on the same blocks; else fall back to CPU."""
        if self.xp is np:
            self.cross_check_result = "CPU only (no cross-check needed)"
            return True
        for a, b in blocks:
            g = to_numpy(self._sieve_block_gpu(a, b))
            c = self._sieve_block_cpu(a, b)
            if not np.array_equal(g, c):
                self.cross_check_result = (f"MISMATCH on [{a:.0e}, {b:.0e}): GPU {len(g)} primes, "
                                           f"CPU {len(c)}; falling back to CPU")
                print(self.cross_check_result)
                self.xp, self.backend = np, "CPU (NumPy, after failed GPU cross-check)"
                return False
        self.cross_check_result = f"GPU matched CPU exactly on {len(blocks)} block(s)"
        return True

    # ---- cache -------------------------------------------------------------------
    def _seg_path(self, k: int) -> Path:
        return self.cache / f"seg{k:06d}.npz"

    def _build_segment(self, k: int):
        lo, hi = k * SEGMENT, (k + 1) * SEGMENT
        parts = [to_numpy(self.sieve_block(a, min(a + self.block, hi))) for a in range(lo, hi, self.block)]
        P = np.concatenate(parts)
        first = int(P[0]) if len(P) else -1
        gaps = np.diff(P)
        assert gaps.size == 0 or gaps.max() < 65536
        np.savez(self._seg_path(k), first=first, count=len(P), gaps=gaps.astype(np.uint16))

    def _load_segment(self, k: int) -> np.ndarray:
        z = np.load(self._seg_path(k))
        if int(z["count"]) == 0:
            return np.array([], dtype=np.int64)
        P = np.empty(int(z["count"]), dtype=np.int64)
        P[0] = int(z["first"])
        np.cumsum(z["gaps"], dtype=np.int64, out=P[1:])
        P[1:] += P[0]
        return P

    def ensure(self, hi: int, verbose: bool = True):
        """Build every cache segment needed for [0, hi)."""
        if not self.cache:
            return
        missing = [k for k in range((hi + SEGMENT - 1) // SEGMENT) if not self._seg_path(k).exists()]
        t = time.time()
        for i, k in enumerate(missing):
            self._build_segment(k)
            if verbose and (i + 1) % 10 == 0:
                print(f"  prime cache: {i + 1}/{len(missing)} segments ({time.time() - t:.0f}s)")
        if verbose and missing:
            print(f"prime cache: built {len(missing)} segment(s) of {SEGMENT:.0e} in {time.time() - t:.1f}s "
                  f"on {self.backend}")

    # ---- streaming ---------------------------------------------------------------
    def ranged_blocks(self, lo: int, hi: int, xp=None):
        """Yield (a, b, P): the primes (>= 5) in [a, b), consecutive ranges covering [lo, hi).
        P is on the store's backend unless xp=np is asked for."""
        xp = xp or self.xp
        lo = max(lo, 0)
        if self.cache:
            for k in range(lo // SEGMENT, (hi + SEGMENT - 1) // SEGMENT):
                if not self._seg_path(k).exists():
                    self._build_segment(k)
                a, b = max(lo, k * SEGMENT), min(hi, (k + 1) * SEGMENT)
                P = self._load_segment(k)
                P = P[(P >= a) & (P < b)]
                yield a, b, (P if xp is np else xp.asarray(P))
        else:
            for a in range(lo, hi, self.block):
                b = min(a + self.block, hi)
                P = self.sieve_block(a, b)
                yield a, b, (P if xp is self.xp else (to_numpy(P) if xp is np else xp.asarray(P)))

    def blocks(self, lo: int, hi: int, xp=None):
        """Yield arrays of the primes (>= 5) in [lo, hi), in increasing order, block by block."""
        for _, _, P in self.ranged_blocks(lo, hi, xp):
            if len(P):
                yield P

    def blocks_with_carry(self, lo: int, hi: int, carry: int, xp=None):
        """Like blocks(), but each block starts with the last `carry` primes of the previous one,
        so consecutive-prime statistics (gaps, lags) cross block borders correctly.
        Yields (P, fresh) where P[fresh:] are this block's own primes."""
        xp = xp or self.xp
        tail = None
        for P in self.blocks(lo, hi, xp):
            if tail is not None and carry:
                P2 = xp.concatenate([tail, P])
                yield P2, len(tail)
            else:
                yield P, 0
            if carry:
                tail = P[-carry:] if len(P) >= carry else (xp.concatenate([tail, P])[-carry:]
                                                           if tail is not None else P)

    def count(self, lo: int, hi: int) -> int:
        return sum(int(len(P)) for P in self.blocks(lo, hi))

    def window(self, lo: int, hi: int) -> np.ndarray:
        """All primes in a small window, as one NumPy array (for skills that need a short stretch)."""
        parts = [to_numpy(P) for P in self.blocks(lo, hi, np)]
        return np.concatenate(parts) if parts else np.array([], dtype=np.int64)


_DEFAULT = None


def default_store() -> PrimeStore:
    """CPU store without a cache: used when a skill runs outside the agent (tests, examples)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PrimeStore()
    return _DEFAULT
