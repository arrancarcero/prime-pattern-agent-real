# Prime-Pattern Agent

A research agent that looks for patterns in primes and in natural measurements, and only keeps a pattern if it:

1. beats a named null model,
2. survives on a held-out range it was never fitted on,
3. passes the mirror check (reversed and reflected), and
4. is not already in the known-results tuple.

Companion document: **The Prime-Pattern Map** (every idea, why it works, and how it connects).

## Install (Windows PowerShell, conda base)

```powershell
cd prime-pattern-agent
python -m pip install -e ".[nature,dev]"
```

`numpy` is the only required package. `[nature]` adds statsmodels, scipy and pooch for the
natural-measurement skill. `[dev]` adds pytest.

## Run

```powershell
ppa truth                         # recheck the truth tuple (about 20 s at --exp 9)
ppa skills                        # list skills and the null model each one must beat
ppa contract                      # print the rules every run follows
ppa tools                         # fingerprint and calibrate the factoring tool tuple
ppa check "1234*5679=7007886"     # fast digital-root + mod 11 check, then exact
ppa graveyard                     # ideas already tested and buried (ppa graveyard <key> to check one)
ppa run --dig                     # ignore the graveyard once and retest buried ideas
ppa blindspots                    # where the digital root is blind and what mod 11 sees
ppa trail 3952806                 # overflow trail: rebuild a number from its running roots
ppa run                           # everything, using agent.toml
ppa run --intent explore          # only the self-directed exploration loop
ppa run --intent nature           # only natural measurements
ppa run --intent curves           # Hasse-Weil bound and Sato-Tate angles on elliptic curves
ppa run --intent factoring        # the factoring toolbox, calibrated first
ppa run --intent candles          # redshift + standard-candle test of the zeta zeros
ppa run --intent spacing          # gaps between zeta zeros vs nuclear energy-level laws
ppa run --intent walk             # zeros walk home: eta walk vs two-way Riemann-Siegel walk
ppa run --intent quantum          # simulated quantum computer: Shor on 15, 21, 35, 143 + the memory wall
ppa run --intent twins            # twin primes: count, dr transitions, gaps, per-cycle Fano vs two nulls
ppa run --max 1e10 --steps 60     # bigger range, longer exploration
python -m pytest                  # 11 tests: core, truth, agent, factoring
```

If `ppa` is not found, use `python -m ppa run ...` instead.
Each run writes `runs/ledger-YYYYMMDD-HHMMSS.json`.

## Folder structure

```
prime-pattern-agent/
├── agent.toml               settings: range, thresholds, planner budget, null model
├── CONTRACT.md              the 10 rules every run obeys
├── pyproject.toml           package + `ppa` command
├── runs/                    ledgers (one JSON file per run)
├── tests/                   test_core, test_truth, test_agent
└── src/ppa/
    ├── cli.py               ppa run | skills | truth | contract
    ├── config.py            reads agent.toml
    ├── core/                the instruments
    │   ├── projections.py   dr, mirror, wheel, gate_bits, joint_state (Layers 1, 2, 6)
    │   ├── sieve.py         segmented sieve
    │   ├── nulls.py         Cramer-wheel random-prime model
    │   ├── stats.py         transition tables, residual z, holdout score, Fano curve, constant B
    │   ├── factoring.py     trial, Fermat, Pollard rho, Pollard p-1, elliptic-curve method
    │   └── ledger.py        Finding + labels, writes runs/*.json
    ├── truth/               frozen tuples (read-only during a run)
    │   ├── truth_tuple.py       known values: pi(10^k), twins, structure, constants
    │   ├── randomness_tuple.py  the six-row signature of true randomness
    │   ├── nature_truth.py      known natural periods (sunspots, CO2, El Nino)
    │   ├── known_results.py     published results + retired ideas
    │   └── tool_tuple.py        the factoring tools, frozen, fingerprinted, calibrated
    ├── skills/              one detector per file, each names its null model
    │   ├── calibrate.py     sieve vs truth tuple (always first; stops the run on failure)
    │   ├── transitions.py   mod-M, lag-k transitions: stage A vs random, stage B vs gap rule
    │   ├── dr_gap.py        digital-root transitions vs the gap rule
    │   ├── fano.py          Fano factor vs window size
    │   ├── spectrum.py      FFT peaks vs wheel periods
    │   ├── gates.py         gate occupancy + 3-bit XOR rule
    │   ├── randomness.py    prime-derived sign sequences through the randomness tuple
    │   ├── nature.py        natural measurements: known periods + randomness tuple
    │   ├── curves.py        elliptic curves: |a_p| <= 2 sqrt(p) and Sato-Tate angles
    │   ├── factoring.py     each factoring tool against numbers with different weaknesses
    │   ├── candles.py       zeta zeros as spectral lines (redshift) and standard candles (brightness)
    │   ├── spacing.py       gaps between zeros vs Poisson, GOE (nuclei) and GUE
    │   ├── walk.py          zeros walk home: one-way eta walk vs two-way Riemann-Siegel walk
    │   └── quantum.py       state-vector quantum simulator: Shor's algorithm and the 2^n wall
    └── agent/
        ├── planner.py       proposes hypotheses: seed, deepen, combine, refine
        └── loop.py          calibrate -> fixed skills -> explore -> ledger
```

## How the agent finds further concepts

The planner starts with every modulus in `start_moduli`, then learns from each result:

| Rule | Trigger | New hypothesis |
| --- | --- | --- |
| deepen | a pattern beats the random model | same modulus, one lag further |
| combine | two moduli both show patterns | their lcm (the Chinese Remainder / moire idea) |
| refine | a true candidate appears | 2M and 3M, ahead of everything else |

Transitions are tested in two stages. Stage A compares real primes with random wheel numbers.
Real primes usually win, and that is the known Lemke Oliver-Soundararajan bias. Stage B compares
them with the gap rule (next class = this class + gap, gaps taken from a coarser class). Only
stage B can produce a real **candidate**: structure that gap sizes do not explain.

## The tool tuple (check the tools, not just their results)

The factoring tools live in a frozen tuple. Before any of them runs, each one must reproduce a
known factorization and must fail on a prime, and the tuple's fingerprint (a hash of every tool's
source code) is printed. A stubbed tool that always claims success fails both checks and changes
the fingerprint, so it is caught three ways.

| Tool | Catches |
| --- | --- |
| trial | a small factor |
| fermat | two factors close together (N = a^2 - b^2) |
| rho | any factor up to about 1e12 (a walk mod p that loops) |
| p_minus_1 | p - 1 built from small primes |
| ecm | some elliptic curve with a smooth size p + 1 - a_p (inside the Hasse range) |

## Fast pre-check: casting out nines + mod 11

`truth_tuple.precheck(op, a, b, got)` checks an answer without redoing the calculation.
The digital root (mod 9) the answer *must* have comes from the inputs' digital roots;
if the answer's digital root differs, it is wrong. The digital root cannot see digit
order, so a second view, the alternating digit sum (mod 11), is checked too. Together
they work as one mod 99 check. `verify_calc` runs the pre-check first and only does the
exact calculation if it passes. `dr_pairs(6)` lists the 9 digit pairs that lead to 6.

Measured on 200,000 wrong multiplication answers:

| Mistake | Digital root | Mod 11 | Both |
| --- | --- | --- | --- |
| One wrong digit | 97.9% | 100% | 100% |
| Two digits swapped | 0% | 54% | 54% |
| Random wrong answer | 89.0% | 90.9% | 99.0% |

A failed pre-check proves the answer is wrong; a pass only means "probably right".
Going backwards does not work: 1,000,000 seven-digit numbers share each digital root.

### Blind spots (`ppa blindspots`, `truth/blind_spots.py`)

On the 9-position clock, 9 is a full lap, so 9 and 0 do nothing. That is why a 9 can be
skipped when taking a digital root, and also why the digital root cannot see a 9 or 0
added, dropped or traded, or any change in digit order. On the 11-position clock,
10 = -1, so digit places alternate + and -. That second view sees most of the gap:

| Mistake (all invisible to the digital root) | Mod 11 catches | Why |
| --- | --- | --- |
| 0 written as 9, or 9 as 0 | 100% | changes the value by 9 x 10^k = +-9 mod 11, never 0 |
| Neighbouring digits swapped | 100% | changes it by +-9(a - b), never 0 |
| A 9 or 0 dropped or added | ~92% | every digit to its left flips sign; misses ~1 in 11 |
| Any two digits swapped | ~57% | same-sign places (even gap) are invisible to mod 11 too |

### Overflow trail (`ppa trail N`, `truth/overflow_trail.py`)

When adding digits for a digital root, a step that passes 9 (like 14 -> 5) is one lap of
the 9-clock. Keep the running root and a lap flag at every step, and each digit comes back
as `d = new_root - old_root + 9 * lap`. Adding 0 never laps but adding 9 always does
(once the root is at least 1), so the flags remove the 0-vs-9 blind spot.
For 3952806 (read right to left) the laps are 3, so the digit sum is 6 + 3 x 9 = 33.

How many 7-digit numbers match 3952806 at each level of information:

| What you keep | Numbers that match |
| --- | --- |
| Digital root only | 1,000,000 |
| Root + lap count (= digit sum 33) | 464,653 |
| Root + where the laps happened | 41,057 |
| Every running root, no lap flags | 4 (only 0 vs 9 left open) |
| Full trail (roots + lap flags) | 1 |

The full trail is as long as the number, so it re-encodes the number rather than
compressing it. The order lives in where the laps happen.

## Standard candles (`ppa run --intent candles`)

The prime error `(psi(x) - x)/sqrt(x)` is a sum of waves, one per zeta zero beta + i*gamma.
Like galaxy light: each zero's wave stretches as x grows (period 2 pi x / gamma in n), which
is a fixed spectral line at gamma in log x (redshift). After dividing by sqrt(x), a zero on
the line shines equally at every distance; an off-line zero brightens by x^(beta - 1/2), like
the supernovae that revealed dark energy. Far/near brightness gives the implied beta.

Measured, primes to 10^8: 10 lines within 0.005 of the known zero heights; implied beta
0.497 to 0.505 for all 10. Calibration: a fake zero planted at beta = 0.6, gamma = 35 is read
back as 0.601. This cannot prove RH: an off-line zero far higher up would stay invisible.

Noise check: white noise is added at 0.5x to 20x the signal's own spread. The brightest zero
(14.13) still reads beta = 0.5 vs 0.6 apart at 5x; dim zeros (near 50) hold to about 1x.
Static spreads over every frequency, but a zero sits at one, so averaging cancels the static.
Random Cramer-model primes (no zeros) show no lines at all, only a louder hum everywhere.

## Zero spacing (`ppa run --intent spacing`)

Montgomery (1972) found zeta zeros repel; Dyson recognized random-matrix statistics, the model
for energy levels of heavy nuclei. The skill finds zeros from sign changes of Hardy's Z(t)
(Riemann-Siegel formula), checks the count against Riemann-von Mangoldt, and compares the
gaps with three laws: Poisson (no repulsion), GOE (nuclei, repulsion ~ s) and GUE (~ s^2).

| Zeros up to height | Gaps | KS: Poisson | KS: GOE | KS: GUE |
| --- | --- | --- | --- | --- |
| 1,000 | 648 | 0.326 | 0.113 | 0.045 |
| 3,000 | 2,468 | 0.315 | 0.108 | 0.040 |
| 10,000 | 10,141 | 0.307 | 0.095 | 0.027 |

Higher up, the **FFT batch walk** (`core/batch_walk.py`, Odlyzko-Schonhage style) evaluates the
two-way walk at 262,144 heights with one non-uniform FFT: near height 10^8 that takes 0.06 s
instead of ~67 s one height at a time, matching it to 5e-7. It finds 10,554 zeros in
[10^8, 10^8 + 4000] (Riemann-von Mangoldt: ~10,557), and their gaps fit GUE with KS 0.010,
inside the 5% critical value, four times closer than the low zeros.

GUE wins at every height, and its distance shrinks as the zeros get higher (the small gap that
remains is a known low-height effect). Zeros repel more strongly than nuclear levels, which is
why a Hilbert-Polya system behind them would have to break time-reversal symmetry.

A side check of element abundances (Lodders 2010, solar system): after removing the steep trend
and the even/odd (Oddo-Harkins) effect, odd elements with a prime number of protons look ~5x more
common (p = 0.04), but that is driven by nitrogen (next to spallation-depleted boron) and fluorine.
Without Z < 11, p = 0.16; beyond the iron peak, p = 0.62. No prime signal in abundances.

## Quadratic sieve (`quadratic_sieve` in the tool tuple)

Dust settles near the ground: Q(x) = x^2 - N for x just above sqrt(N) is small, so many values
break completely into small primes (the factor base). Each smooth value is one observer's partial
clue; combining clues whose prime exponents are all even (linear algebra over GF(2)) gives
X^2 = Y^2 (mod N), and gcd(X - Y, N) is a factor. This flag (smoothness near sqrt N) exists for
every N; good key choices cannot remove it, which is why the sieves (and the number field sieve
that factored RSA-250) work on any key. Measured, balanced factors: 40-bit in 0.2 s, 50-bit
(30-digit N) in 0.8 s, where rho needs ~30 s; this simple one-sided version runs out of clues
at 60-bit factors. Its null prime (10^12 + 37) gives no factor.

The flat-course grave (`ppa graveyard factoring:flat_course_shortcut`) records nine other
attempts to find one hole on flat ground faster than covering it, each with its numbers; the
sqrt(N) ceiling is proven. The test for a new idea: does it make the ground tilt, or only
cover faster?

## Sign flips + FFT (`sign_fft` in the tool tuple)

The Jacobi sign pattern (a / N), a = 0..N-1, is computed by reciprocity flips without knowing the
factors. It is two hidden sign patterns (periods p and q) multiplied together, so its FFT glows
at exactly sqrt(N) at every frequency except dark lines at multiples of p and q; the first dark
line's gcd with N is a factor (77 -> 7, 1,022,117 -> 1009). It deciphers any odd N with distinct
prime factors, but the pattern is N long, so it costs ~N steps where trial division needs
~sqrt(N). Shor's quantum algorithm makes the same move (hide the factors in a period, then
Fourier transform) on about log N qubits. Its null check uses the prime 1,000,003, which shows
no dark lines.

## Twin primes (`ppa run --intent twins`)

Counts twin pairs against Hardy-Littlewood (1e9: 3,424,506 vs 3,425,308, ratio 0.99977), then
measures how twins follow each other: does the next twin repeat the lower twin's digital root
(2, 5 or 8)? how many tiny and huge gaps? how regular is the count per 30030-block (Fano)?
Two nulls: stage A keeps only the 22,275 admissible positions of every 510,510 (wheel to 17);
stage B also strikes positions for every prime up to Q = sqrt(x)/30. Both pick at the real
density.

| At 1e9 | Real | Random | Wheel to 17 (z) | Sieve to 1000 (z) |
| --- | --- | --- | --- | --- |
| Next twin repeats dr | 0.312 | 0.333 | 0.314 (-9.0) | 0.312 (-0.4) |
| Steps 2 -> 5 -> 8 | 0.348 | 0.333 | 0.347 (+3.6) | 0.347 (+1.9) |
| Tiny gaps | 0.198 | 0.221 | 0.204 (-36) | 0.197 (+3.4) |
| Huge gaps | 0.0445 | 0.0498 | 0.0463 (-14) | 0.0445 (+0.7) |
| Fano per 30030 | 0.778 | 1 | 0.925 (-15) | 0.803 (-4.2) |

Every effect is real and every one is explained by small primes (rediscoveries). The lesson:
sharper data needs a null that knows more primes (a sieve to 97 was enough at 1e8, not at 1e9).

## Simulated quantum computer (`ppa run --intent quantum`)

A full state-vector simulator: n qubits = 2^n complex amplitudes. Shor's order finding puts a
counting register in superposition, multiplies by a^j mod N (a permutation of basis states, so
unitary), applies the inverse quantum Fourier transform (an FFT along the counting register),
then measures; continued fractions give the period r and gcd(a^(r/2) +- 1, N) the factors.

| N | Factors | Period | Qubits | State size |
| --- | --- | --- | --- | --- |
| 15 | 3 x 5 | 4 | 12 | 64 KB |
| 21 | 7 x 3 | 6 | 15 | 0.5 MB |
| 35 | 7 x 5 | 12 | 18 | 4 MB |
| 143 | 11 x 13 | 60 | 24 | 256 MB |

The memory wall: every qubit doubles the state (x4 per 2 qubits measured), and the full QFT time
grows about x5 per 2 qubits. 30 qubits fill an RTX 5080's 16 GB; Shor on RSA-2048 (~4,099 qubits)
would need ~10^1235 bytes. Set `[quantum] gpu = true` in agent.toml and `pip install cupy-cuda12x`
to run it on an NVIDIA GPU.

**Threads across the mirror.** Splitting the qubits (circuit cutting, tensor networks) and
describing each side by a gradient works when few "threads" (entanglement, Schmidt rank)
cross the cut. Each thread after the QFT is a linear phase gradient across the spikes (for 15:
0, 1/4, 1/2, 3/4 turns, and its mirror 0, 3/4, 1/2, 1/4). Measured across the cut between the
counting and work registers, the thread count is exactly the period r (15: 4, 21: 6, 35: 12,
143: 60); a cut through the middle of the counting register is worse (143: 228). For RSA-2048
r is ~2^2048, so the split costs as much as the full state. A 3-way mirror (repetition code) protects bits; quantum error
correction must do it without copying (no-cloning), which Shor's 9-qubit code (1995) achieves
with entanglement and parity checks.

## Walk home (`ppa run --intent walk`)

**One-way walk:** eta(s) = 1 - 2^-s + 3^-s - ... adds arrows of length 1/sqrt(n) turned by
t log n. eta = (1 - 2^(1-s)) zeta, so at a zeta zero the walk ends at the origin. Plain partial
sums wobble by ~1/(2 sqrt N), so it walks toward infinity; averaging neighbouring stops helps.

**Two-way walk (counting back from infinity):** with M = floor(sqrt(t/2pi)) arrows forward,
the infinite tail folds back as the mirror walk, and one small correction arrow (the
Riemann-Siegel remainder) closes the gap. Its distance from home is |Z(t)|.

| Zero | Two-way walk | One-way, averaged | One-way, plain |
| --- | --- | --- | --- |
| 14.13 | 2 arrows | 148 | 64,773 |
| 37.59 | 4 arrows | 369 | 142,554 |
| 40.92 | 4 arrows | 471 | 250,011 |

(arrows needed to get as close to home as the two-way walk does). Null: between zeros neither
walk comes home (one-way at least 1.42 away, two-way at least 0.71). The zero heights agree
with the spacing skill's zero finder. `examples/walk_home.py` draws both walks.

**Fractal dimension of the walks** (box counting, `core/fractal.py`, calibrated on a line = 0.98
and the Koch curve = 1.29 vs 1.262): the one-way walk has D = 1.20 at zeros and 1.19 between them,
against 1.53 for a null walk with the same steps in random directions. Its local dimension holds
near 1.23 down to 1/1024 of the walk's size, but that comes from the zig-zag of the alternating
signs: the smoothed (midpoint) spiral slides from 1.32 to 1.03, a smooth curve up close. Being a
zero decides where the walk ends, not its dimension. `examples/walk_fractal.py` draws it.

## The graveyard (`ppa graveyard`, `truth/graveyard.py`)

A separate layer that remembers what was wrong, so the agent does not test it again.
Frozen graves (fingerprinted like the truth tuple) hold each buried idea's claim, the test that
killed it, the numbers, what was tried, and the rule for digging it up: a bigger range, a method
not yet tried, or new data. Ideas ruled out by proof or counting never reopen. The learned layer,
`runs/graveyard.json`, records every planner hypothesis that failed, with the range it failed at.
Before each test the planner asks the graveyard; buried ideas are skipped without using a step.
Because everything up to mod 72, lag 3 was already explained by gaps at 1e10, `ppa run --intent
explore` below 1e10 now skips those seeds and says so: use `--dig`, or raise `range.max_n`.

## Moire and zone-plate tools (in the tool tuple)

**moire:** grids spaced p and q beat every N/(q - p). Measure the beat gap d and
p = (sqrt(d^2 + 4N) - d)/2. Scanning d = 0, 2, 4, ... is Fermat's method in costume
(N = a^2 - b^2 with b = d/2). Inverting the grids (dark <-> light) leaves the beat unchanged.

**zone_plate:** the lens pattern cos(pi x^2 / N) repeats every N. Sampled every s-th point it
repeats every N / gcd(s^2, N), so a stride sharing a factor with N collapses into a shorter
cycle (aliasing). Trying strides 2, 3, 5, ... is trial division seen through a lens.

Because each claims to be an old method in disguise, `ppa tools` also runs a **twin check**:
on 200 test numbers, moire must agree with Fermat and zone_plate with trial division
(both 200/200). `examples/moire_zone_plates.py` draws the beat, the inverted beat, two zone
plates making straight fringes, and the aliasing.

## Example: RSA-PSS signatures (`examples/pss_demo.py`)

RSA-PSS built by hand from RFC 8017 (EMSA-PSS, MGF1 with SHA-256, 32-byte salt) and tested three ways:

1. **Checked by the real library:** every signature is verified by `cryptography` (OpenSSL). The same
   message gives a different signature each time, and a tampered message is rejected by both.
2. **Salts through the randomness tuple:** `os.urandom` passes; a reused salt fails (and makes PSS
   deterministic); Python's Mersenne Twister *passes* but is predictable from its seed. Passing the
   tuple is necessary, not sufficient.
3. **Keys through the tool tuple:** a badly generated 1024-bit key (primes close together) is factored
   by Fermat in one step, and a forged signature is accepted by OpenSSL. A properly generated key
   survives every tool.

```powershell
python -m pip install -e ".[crypto]"
python examples/pss_demo.py
```

## Labels

`established`, `conjecture`, `hypothesis`, `speculation`, `retired`, `rediscovery`,
`candidate`, `not-supported`, `check`. The agent can never assign anything higher than
`candidate`. Promotion needs a proof or a published result.

## Extend it

- **New skill:** copy any file in `skills/`, set `SKILL = Skill(name, null_model, description, run)`,
  add the module name to `MODULES` in `skills/__init__.py`.
- **New projection:** add a function to `core/projections.py`, use it inside a skill.
- **Your own measurements:** add a series in `load()` in `skills/nature.py`
  (for example spinning-fluid speeds or a tire runout trace).
- **New known result:** add a `Known(...)` row to `truth/known_results.py`, so the agent
  labels it as a rediscovery instead of new.

## First results (2026-11-21, primes to 10^9)

- Calibration: pi(10^9) = 50,847,534, matching the truth tuple.
- Transitions mod 3 and mod 5 beat random wheel numbers (z = 11 to 69). All of it is
  explained by the real gap distribution, so they are logged as rediscoveries.
- No candidates. Nothing beyond the gap rule survived at any modulus tested up to 72 and lag 3.
- Fano factor below 1 and falling with window size, as the Montgomery-Soundararajan conjecture predicts.
- The second-order constant B = 1 - Euler's constant - log(2 pi) = -1.4151, measured from prime
  counts in windows: -1.443 +/- 0.070 at 10^9, -1.415 +/- 0.017 at 10^10 (middle windows H = 10^3, 10^4).
- Natural measurements: the instrument found the 11-year sunspot cycle, yearly CO2 cycle and
  yearly El Nino cycle unaided.
