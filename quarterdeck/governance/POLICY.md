# Quarterdeck Policy — VN2 (Execution Broker Rules)

Binary pass/fail rules. Ambiguous cases **fail** until clarified by a human.

## P1 — Raw data immutability

The following are **read-only** for automated agents:

- `data/raw/` competition extracts (Initial State, Week * Sales CSVs) unless a human explicitly authorizes a versioned replacement.

**Pass:** Agent does not modify, delete, or overwrite files under `data/raw/`.  
**Fail:** Any write or destructive operation under `data/raw/`.

## P2 — Diff-only changes

All code and configuration changes must appear as **git-tracked diffs** with a commit message stating purpose.

**Pass:** Changes committed (or presented as a single coherent patch) with non-empty message.  
**Fail:** “Chat-only” code that never lands in version control.

## P3 — Run manifest (recommended for production runs)

For any pipeline run that feeds a **published number** (cost, leaderboard rank), the run must record:

- Git SHA (or dirty flag)
- Config file hash or embedded config snapshot
- Input data fingerprints (hashes or paths + modification times where hashing unavailable)
- UTC timestamp
- Row counts / task counts completed

**Pass:** A manifest file (JSON/YAML/text) exists alongside outputs for that run.  
**Fail:** Claiming a benchmark result with no reproducibility artifact.

*(Repository does not yet standardize manifest format — **gap**; broker should reject “official” claims until format is adopted.)*

## P4 — New scripts registration

Any new **first-class** script (invoked by humans or CI for core metrics) must be listed in `quarterdeck/governance/MANIFEST.yaml` with description.

**Pass:** MANIFEST updated in same change as script addition.  
**Fail:** New top-level pipeline script omitted from manifest.

## P5 — Assumption registry

`ASSUMPTION_REGISTRY.md` is **human-authoritative**. Agents may propose additions via patch but **must not** silently delete or mark assumptions “validated” without human approval.

**Pass:** Only additive PRs or explicit human-edited status.  
**Fail:** Agent marks ASM-* as validated without evidence attachment.

## P6 — No silent self-execution

Agents **propose** commands; a **human or broker-approved runner** executes them. (Cursor-style agents may run sandboxed commands only for verification explicitly scoped by the user.)

**Pass:** Execution tied to user-approved run or CI job.  
**Fail:** Unattended agent runs full eval on shared production data without record.

## P7 — Evidence for claims

Statements of the form “tests passed” or “cost = €X” require:

- Captured command output or log artifact, and  
- Pointer to manifest (P3) or input hashes.

**Pass:** Artifact path cited.  
**Fail:** Bare assertion.

## P8 — No duplicate utilities

If `MANIFEST.yaml` lists a utility for a purpose (e.g. `vn2.analyze.model_eval.run_evaluation`), agents **must not** reimplement full eval loops in a one-off script without broker waiver.

**Pass:** Reuse or explicit waiver documented in PR.  
**Fail:** Copy-paste eval loop with divergent metric definitions.

## P9 — Cost evaluation consistency

When comparing policies or service levels, **evaluation** costs (cu, co) must not be conflated with **ordering** cost adjustments unless the experiment design explicitly requires both and they are documented.

**Pass:** Matches `full_L3_simulation` pattern (eval_costs vs order costs).  
**Fail:** Using adjusted cu/co for final leaderboard totals without disclosure.

## P10 — Secrets and credentials

Do not commit API keys, tokens, or machine-specific passwords into the repo.

**Pass:** No new secrets in tracked files.  
**Fail:** Credentials in source or config.

## P11 — Dependency bounds

New dependencies must respect `pyproject.toml` style (version caps) and be declared in `pyproject.toml`, not ad-hoc `pip install` only.

**Pass:** Declared dependency with version spec.  
**Fail:** Import of undeclared package in committed code.

---

**Review cadence:** Human owner before each release or quarterly.
