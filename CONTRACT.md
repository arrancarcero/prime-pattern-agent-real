# Agent contract

These rules bind every skill and every run. The code enforces them; this file states them.

1. **Calibrate first.** Before any skill runs, the sieve must reproduce pi(max_n) from the
   truth tuple. If it does not, the run stops.
2. **No pattern without a null model.** Every skill names the null model it is compared with
   before it looks at the data.
3. **Hold out.** Fit on the train range, freeze, score on the test range. Only held-out
   scores decide whether a pattern survives.
4. **Check the mirror.** Every surviving transition pattern is also scored reversed
   (b -> a) and reflected (r -> -r mod M).
5. **Check what is already known.** A surviving pattern is compared with the known-results
   tuple. A match is logged as a rediscovery, not as new.
6. **Label everything.** Every ledger entry carries one label: established, conjecture,
   hypothesis, speculation, retired, rediscovery, or candidate.
7. **Never promote by computation alone.** "Candidate" is the highest label the agent can
   give. Moving to "established" needs a proof or a published result.
8. **Retired stays retired.** Ideas in the retired list are never proposed again.
9. **Frozen truth.** Truth tuples are Python tuples with a printed fingerprint. A run whose
   fingerprint differs from the last run is flagged.
10. **Check the tools, not just their results.** Every factoring tool lives in the frozen tool
    tuple, is fingerprinted from its own source code, must reproduce a known answer, and must
    fail on a prime. A tool that fails any of these is never run.
