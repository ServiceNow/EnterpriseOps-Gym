# step04_v4 train analysis (9 regressions vs v3)

Aggregate: tool calls/task 11.8–12.4 (v3) → 14.0; agent text +15%; explicit notification `status` share fell (test 18 → 14 of ~60). No over-writing; regressions are dropped fields / mis-mapped values.

- dc6af68f — dilution: PLAN became a list of search terms; SLA stand-down rule no longer quoted.
- 8487a7f6 — noise (same choice in v3_r2).
- be0bd794 — dilution + "unset needs a policy search": searched, concluded "no default", left unset.
- cadddb04 — sentence-level quoting cut off a continuation sentence carrying the duty.
- 9c6f3e01 — noise (digit grouping slip despite readback).
- b4505336 — v4-caused: enum + unset clauses made user-stated fact insufficient for a structured field.
- 7e9733a5 — v4-caused: "pick the one with the same meaning" mapped "high … criticality" → critical.
- 92ddef90 — dilution: per-parameter check skipped on a notification.
- b09b6872 — count fired (good), off by one (noise).

Gains (train+test that failed both v3 runs): search grid (2), all-branch quoting (1), End state + ordering (2), both-record refs (1), enum (1, but harmful elsewhere). 48850a98 first pass (enum).

v5 = v3 + keep {all-branch quoting incl. continuation sentences, End state + ordering + dependent values, both-record refs, search grid, unconditional count, audit vs End state}; drop {same-meaning enum, unset-needs-search, informal nouns, across-records note}. Restore v3's literal mapping.
