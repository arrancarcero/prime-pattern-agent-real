# Agent contract (real primes only)

These rules bind every skill and every run. The code enforces them; this file states them.

1. **Real primes only, from 5 on.** Every measurement uses primes produced by the sieve,
   starting at 5. 1 is not prime; 2 and 3 are left out of every pattern test. Where a theorem
   counts all primes (pi(x), the explicit formula), its formula is corrected exactly for 2 and 3.
2. **No simulated data anywhere.** No random primes, planted signals, added noise or random
   walks, in a measurement or in a verdict.
3. **Every yardstick is real.** A pattern is judged only by one of:
   - a theorem, checked on every prime or twin in range (one failure would refute it);
   - an exact formula, named with its status (theorem, or a stated conjecture such as
     Hardy-Littlewood);
   - real primes in another range (train/test holdout), or a rule built from the same real
     primes (independence, the real gap rule).
4. **No yardstick, then an argument.** A measurement with no formula is labelled `measured`
   and comes with its argument chain: what is proved, what it reduces to, and the open step.
5. **Tests come first and never change to fit data.** Every test, threshold and range is in
   `prereg.toml`, fingerprinted and locked in `prereg.lock`. If the file changes while its
   version stays the same, the agent refuses to run. Changing a test means a new version;
   every ledger names the version and fingerprint it was judged under. The registration lists
   what had already been seen before it was written.
6. **Only registered ranges.** A run may use only a range listed in the registration.
7. **Calibrate first.** The sieve must reproduce pi(max_n) - 2 from the truth tuple, or the run stops.
8. **Hold out.** Transition patterns are fitted on the train range, frozen, and scored on the test range.
9. **Check what is already known.** A surviving pattern is compared with the known results;
   a match is a rediscovery, not new.
10. **Never promote by computation alone.** "Candidate" is the highest label the agent can give.
    A computation cannot prove a statement about infinitely many primes.
