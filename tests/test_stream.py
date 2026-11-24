import math

import numpy as np

from ppa_real.core.primestore import PrimeStore
from ppa_real.core.sieve import primes_between, small_primes


def emulate_gpu_kernel(a, b, chunk):
    """The CUDA kernel's per-thread logic, run thread by thread in Python."""
    base = small_primes(math.isqrt(b) + 1).astype(np.int64)
    base = base[base * base < b]
    s = np.ones(b - a, dtype=np.uint8)
    nchunks = (b - a + chunk - 1) // chunk
    for tid in range(len(base) * nchunks):
        i, c = tid // nchunks, tid % nchunks
        q = int(base[i])
        lo = a + c * chunk
        hi = min(lo + chunk, a + (b - a))
        start = ((lo + q - 1) // q) * q
        start = max(start, q * q)
        for m in range(start, hi, q):
            s[m - a] = 0
    if a < 5:
        s[: 5 - a] = 0
    return np.nonzero(s)[0] + a


def test_gpu_kernel_logic_matches_cpu():
    for a, b, chunk in ((0, 50_000, 4096), (10 ** 6, 10 ** 6 + 120_000, 5000), (999_983, 1_100_017, 777)):
        cpu = PrimeStore()._sieve_block_cpu(a, b)
        assert np.array_equal(emulate_gpu_kernel(a, b, chunk), cpu)


def test_streaming_matches_whole_list(tmp_path):
    lo, hi = 10 ** 6, 3 * 10 ** 6 + 17
    whole = primes_between(lo, hi)
    for store in (PrimeStore(block=250_000), PrimeStore(tmp_path, block=250_000)):
        assert np.array_equal(np.concatenate(list(store.blocks(lo, hi))), whole)
        # carried blocks see every consecutive pair exactly once
        gaps = np.concatenate([np.diff(P) for P, _ in store.blocks_with_carry(lo, hi, carry=1)])
        assert np.array_equal(gaps, np.diff(whole))
    assert PrimeStore().cross_check()                    # CPU-only: nothing to compare


def test_cache_is_two_bytes_per_prime(tmp_path):
    s = PrimeStore(tmp_path)
    s.ensure(10 ** 9)
    f = next(tmp_path.glob("seg*.npz"))
    z = np.load(f)
    assert z["gaps"].dtype == np.uint16 and int(z["count"]) == 50847534 - 2   # pi(1e9) minus 2 and 3


def test_gpu_mode_hands_skills_numpy(tmp_path):
    """With a GPU sieve, skills must still get NumPy arrays (no CuPy helper kernels)."""
    s = PrimeStore(tmp_path / "c")
    s._cp = object()                                  # pretend a GPU is present
    s._sieve_block_gpu = s._sieve_block_cpu           # stand-in for the CUDA kernel
    assert s.xp is np and s.cross_check()
    lo, hi = 10 ** 6, 2 * 10 ** 6
    for P in PrimeStore(block=300_000).blocks(lo, hi):
        assert isinstance(P, np.ndarray)
    s2 = PrimeStore(block=300_000)
    s2._cp = object()
    s2._sieve_block_gpu = s2._sieve_block_cpu
    assert all(isinstance(P, np.ndarray) for P in s2.blocks(lo, hi))
    assert np.array_equal(np.concatenate(list(s2.blocks(lo, hi))), primes_between(lo, hi))
