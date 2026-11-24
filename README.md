# Prime-Pattern Agent, real primes only

A separate, stricter copy of [prime-pattern-agent](https://github.com/arrancarcero/prime-pattern-agent).
It measures only real primes, from 5 on, and judges every pattern only by theorems, exact
formulas, or other real primes. Every test was registered before it was run.

## The three rules

1. **Real primes only.** No simulated primes, planted signals, noise or random walks anywhere.
   2 and 3 are left out; where a theorem counts them, its formula is corrected exactly.
2. **Real yardsticks only.** A theorem checked on every prime, an exact formula (with its status:
   theorem, or a named conjecture such as Hardy-Littlewood), or real primes in another range.
   Where none exists, the result is labelled `measured` with its argument chain: what is
   proved, what it reduces to, and the open step.
3. **Tests first, never changed to fit data.** `prereg.toml` holds every test, threshold and
   range. It is fingerprinted and locked (`prereg.lock`); the agent refuses to run if it changes
   without a new version number. The registration commit comes before any result in the git
   history, and it lists what had already been seen before it was written.

See `CONTRACT.md` for the full rules.

## Install and run

```
pip install -e .
ppa-real prereg                    # the registered tests, version and fingerprint
ppa-real run --max 1e8             # every skill, at a registered range (1e7, 1e8, 1e9, 1e10)
ppa-real run --intent twins --max 1e9
ppa-real run --intent explore      # the planner, transitions only
ppa-real run --max 1e11 --gpu      # NVIDIA GPU (pip install cupy-cuda12x), cross-checked against the CPU
ppa-real skills | truth | contract | graveyard
```

## Speed and memory: streaming, cache, GPU

Primes are streamed block by block; every skill keeps only running totals, so memory no longer
grows with the range (the full 1e10 run used 3.9 GB). From 1e9 up, each 1e9-wide segment is
sieved once and cached on disk as its first prime plus 2-byte gaps (`.prime_cache/`, about
0.9 GB for 1e10 and 8 GB for 1e11); later passes read the cache. `--cache none` turns it off.

`--gpu` sieves on an NVIDIA GPU with one CUDA kernel launch per block (CuPy). Before a GPU run is
trusted, the same block is sieved on the GPU and the CPU and must match exactly; on any mismatch
the run falls back to the CPU and the ledger says so (`meta.backend`, `meta.cross_check`). None of
this changes a test: it only changes how fast the same primes arrive.

Checked: the streaming version reproduces the v1 ledgers exactly at 1e8 and 1e9, and the
author's 1e10 PC ledger except for the gates overflow fix, timing text, last-digit float rounding
and the order of exactly tied FFT peaks (periods 6 and 3, and 4.667 and 3.5, have identical power).

Registration v2 adds the ranges 1e11 and 1e12 and changes nothing else; it was committed before
any run at those sizes.

## What is in it

| Skill | Yardstick |
| --- | --- |
| calibrate | exact pi(x) from the truth tuple, minus 2 and 3 |
| transitions | stage A: independence built from the same real primes; stage B: the real gap rule |
| dr_gap, spectrum, gates | exact facts and real train/test holdout |
| fano | the Montgomery-Soundararajan formula B = 1 - gamma - log 2 pi |
| curves | Hasse-Weil bound (theorem), Sato-Tate law (theorem) |
| candles | explicit formula with every zero on the line; lines must sit on the known zero heights |
| spacing | Riemann-von Mangoldt count; Poisson, GOE and GUE gap formulas |
| walk | real heights halfway between known zeros must not come home |
| twins | Hardy-Littlewood: count, 4-tuple pair correlation, Fano per block; proved lemmas |

Left out, because they are not about real primes or they judge with simulated data: factoring,
quantum, randomness, nature, the fractal and noise tests, the digit checks.

## Results under registration v1 (fingerprint 135fd394a9b3ad12)

All registered tests pass at 1e8, and the twins tests at 1e9, with no test changed after the
registration commit.

**Twin primes, 1e9** (Hardy-Littlewood, computed exactly; no simulated twins):

| Test | Real | Formula | Verdict |
| --- | --- | --- | --- |
| Twin count | 3,424,506 | 3,425,308 | ratio 0.99977 |
| Twin pairs at distance 6..180 | 2,021,736 | 2,021,890 | ratio 0.9999, largest \|z\| 1.8 (30 distances) |
| Share of those at distance 0 mod 18 | 0.3175 | 0.3177 | z = -0.6 |
| Fano per 30030-block (99 stretches) | 0.777 | 0.785 | z = -1.6 |

**The digital-root argument chain** (proved, then checked on every twin to 1e9):
- Lemma 1: every lower twin is 5 mod 6 (holds on all 3,424,505).
- Lemma 2: two numbers share a digital root exactly when they are equal mod 9.
- Reduction: two lower twins share a digital root exactly when their distance is 0 mod 18
  (holds on all 3,424,504 consecutive pairs).
- Prediction: Hardy-Littlewood gives 0.3177 of nearby twin pairs at distance 0 mod 18; real 0.3175.
- Open step: consecutive twins repeat their digital root 0.312 of the time. Predicting that
  needs "no twin in between", which pair counts alone do not give, so it is measured only.

**Other skills at 1e8:** 10 prime spectral lines within 0.005 of the known zero heights, every
zero's implied real part between 0.497 and 0.505; zero gaps fit GUE best at low and high
heights; Fano B = -1.335 +- 0.181 (formula -1.4151); Hasse bound holds for every prime.

## Run at 1e10 (registration v1, on the author's PC)

The first run at this size under the frozen tests. Every registered test passes:

| Test | Real | Formula | Verdict |
| --- | --- | --- | --- |
| Primes from 5 to 1e10 | 455,052,509 | pi(1e10) - 2 | exact |
| Twin count | 27,412,679 | 27,411,417 | ratio 1.00005 |
| Twin pairs at distance 6..180 | 12,898,810 | 12,897,555 | ratio 1.0001, largest \|z\| 1.9 |
| Share at distance 0 mod 18 | 0.3179 | 0.3177 | z = +1.8 |
| Twin Fano per 30030-block (999 stretches) | 0.8255 | 0.8268 | z = -0.6 |
| Fano constant B | -1.415 +- 0.017 | -1.4151 | agrees |

Both lemmas held on all 27,412,678 lower twins. Consecutive twins repeat their digital root
0.3151 of the time (measured only). One instrument bug showed up: the gates XOR check multiplied
two primes near 5e9 and overflowed 64-bit integers, so it reported "XOR rule holds: False". The
check now reduces mod 24 before multiplying; the test itself is unchanged. The transitions seeds
were skipped by the graveyard (buried up to 1e10), as registered.

This agent cannot prove the twin prime conjecture or any other statement about infinitely many
primes: it measures, and checks the measurements against theorems and stated conjectures.
