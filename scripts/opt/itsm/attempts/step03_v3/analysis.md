# step03_v3 train analysis

## Persistent failures (failed 7/7)
- 48850a98 — FIXABLE: PLAN quoted the default sentence but decided no enum matched "not yet provided"; enum has a same-meaning value. v3's "map literally" may have pushed word-for-word matching. Clause: map wording to the same-meaning allowed value after listing all enum values.
- 8cb006f5 — DEFECT: v3 first run to write the note; verifier requires trailing period that the policy sentence makes ambiguous (end-of-sentence vs part of template). Golden fails.
- 18cc7ee8 — FIXABLE (medium): indirect nouns ("write-up", "help toward resolving") defeat keyword sweep; note-format rule across all linked items should fire after the last link. Golden fails.
- ccb66a9a — intentional user-vs-policy test; agent read a positional value as a menu and applied "keep fields" only to unrequested fields. Fix would require wording close to that task's policy → SKIP (leakage + over-preservation risk).

## Regressions vs v2 (7)
No spurious-extra-field, forbidden-default or audit-probe-write failures. Turns 6.8 → 7.0 avg; no max_iterations; no text-only early endings.
- 27934d6b — v3-caused: "quote briefly" → model elided the first branch of an ordered if/otherwise rule and applied a later branch.
- 2932a556 — v3-caused: write-by-write reasoning set a dependent record (SLA stage) from the current state instead of the requested end state; audit noticed but claimed no tool could fix.
- 423c11bb — v3-caused: two-sided link; set one side (original_task) and treated the link as done, never set the other side on the earlier-updated record.
- b09b6872 — counting skipped because estimate (209) looked under the limit (truth 242). Counting method is conditional in practice.
- 44904b62 — NOISE / VERIFIER DEFECT: v3 cut to exactly 225 correctly; verifier expects 224 chars.
- fad8a09d — amplified: per-param sweep listed original_task but "leave other optional fields unset" let it drop it with "not specified".
- 525b4783 — partly noise: narrower search grid (2 term×field combos vs 6 in v2).
Proposals v4: quote all branches of conditional rules; write End state before first write and order writes accordingly; two-sided links; unset needs a policy search; counting unconditional; search grid; audit can fix.
