# Progress

## 3 November 2026 (MacBook Pro, later) — pre-CERES code review: findings fixed

**Why.** The WP2 independent review had missed the size-cap hole found earlier today (below), so before any CERES campaign the code was reviewed again. Three reviews ran:
- two independent Claude agents: one on the counter and caps across all arms, one on the runner, CERES scripts and campaign statistics. Neither was told about the earlier finding;
- ChatGPT, given the same brief by Michael;
- my own 400-generation drift check of all 11 arms on even-6 and comp L3 R2. It tracked size, cost and time per generation, and compared charged cost with the reference interpreter on sampled programs. It found no drift and no mismatches.

**Findings and decisions.** Every finding was reproduced before fixing. Commits `14469e2` (counter and caps) and the campaign commit that follows it.

| # | Finding | Found by | Decision |
| --- | --- | --- | --- |
| 1 | A2 executed and charged ADF arguments the body never reads, on every call, so its cost was unbounded. A valid program cost 12,991 units per case at expanded size 127. In real runs, 15–19% of A2's units bought nothing. | counter agent | **[flag: counter change]** Dead code is neither executed nor charged, in every arm: unread ADF arguments (A2), like inactive modules (A3–A6). Cost per case ≤ expanded size now holds for every arm (property-tested). The reference interpreter is now call-by-need, an independent mechanism from the evaluator's static masks. |
| 2 | The functional-size check (added earlier today) counted A2's dead code, while inactive CGP nodes and modules were free. | counter agent, ChatGPT | **[flag: fairness]** Caps measure live code only, in every arm: expanded size ≤ 500 and depth ≤ 17. The functional-size check is removed. A2's stored size (dead code included) is capped at 1,000 nodes, an engineering bound on genotype length; the graph arms' storage is bounded by construction. |
| 3 | A4's readout charge left out the k − 1 weight constants. | counter agent | **[flag]** The readout is charged 3(k − 1), since constants cost 1 everywhere. A4 was being undercharged by about 24%. |
| 4 | A4's depth ignored the readout's sum. | counter agent, ChatGPT | The depth is that of the shallowest equivalent tree, less linear scaling's two uncounted levels. |
| 5 | A4 is charged for zero-weight readout columns. | ChatGPT | Kept: the readout is part of A4's fixed representation, as a constant subtree is part of an A1 program. Documented. |
| 6 | A5-copy's 64-module limit per level rejected legal programs. | ChatGPT | The limit is now 500, which can never bind. |
| 7 | Even-5 parity was both a tuning task and a screening task. | runner agent, ChatGPT | **[flag]** Screening uses even-6 to even-8 plus 11-mux. Even-5 stays as a tuning task. |
| 8 | Missing runs were silently skipped. The G2 test could PASS on 1 of 53 tasks. | runner agent, ChatGPT | The tuning report writes nothing, and the G2 test gives no verdict (INCOMPLETE), unless every planned run is present. The G2 denominator is all planned tasks. |
| 9 | 950 of the 3,600 tuning runs were exact duplicates. The CGP arms had 1–9 distinct configurations, not 20. | runner agent, ChatGPT | **[flag]** CGP tunes its grid: A3 its rows, columns and levels-back; A3-lb1 and A3-lbmax their rows. `n_const` is not tuned. Every arm gets 20 configurations that are distinct in the settings actually used. This also fixes a latent bug: the CGP variant settings overrode tuned parameters. |
| 10 | A half-written `final.parquet` counted as done; resubmitting could run the same run twice. | runner agent | Outputs are written atomically, with `final.parquet` last. A `RUNNING` heartbeat lock makes jobs skip live runs and take over dead ones. |
| 11 | The git commit was recorded at the end of a run; `logs/` was not ignored; tuning had no checkpoints. | runner agent | Versions are captured at the start, and `git_commit` is a column of `final.parquet`. `logs/` is ignored. Tuning checkpoints every 25 generations. |
| 12 | No implemented rule picked A5 or A6 for G2. | runner agent | The tuned A5 and A6 are compared per tuning task; the arm better on more tasks wins, and a tie goes to A5. The result is written to `tuned.yaml` as `g2_arm`. |
| 13 | Two docstrings misstated the counter. | counter agent | Corrected. |

Reviewed and found sound (by at least two reviewers):
- the budget is never exceeded;
- checkpoint/resume is exact for all arms;
- runs are byte-identical across hash seeds;
- the tuning (100–104) and screening (0–29) seed indices don't overlap;
- run IDs are unique;
- the SGE index mapping is correct;
- there is no train/validation/test leakage;
- the Mann–Whitney direction and Holm correction are correct;
- validation is never charged;
- cached modules are charged once;
- A5-copy's cost equals its expanded size.

**Antigravity's review** (commit `f85d210`) arrived after the fixes above. Five of its 11 findings were already fixed: A2's dead arguments, even-5, A5-copy, CGP tuning duplicates with the A5/A6 rule, and A4's readout. Six were new; each was checked and fixed:

| # | Finding | Decision |
| --- | --- | --- |
| 14 | **The G2 test could not detect wins on Boolean tasks.** Runs stop at a solution with error 0, so once A1 solved half its runs no arm could beat it, however little compute it needed. | **[flag]** The G2 score on Boolean tasks is units to solution ÷ budget for a solved run, and 1 + final error for an unsolved one. Any solved run beats any unsolved one; solved runs are ordered by compute used. Regression still uses test MSE. |
| 15 | **Many offspring changed only dead code.** Measured on 40-generation runs: A2 29–34%, A4 23–47%, A6 25–34%, A5 9–21%, against A1 2–6% and A3 0% (CGP's mutation guarantees an active change). | **[flag: fairness]** One rule for every arm: an offspring identical to a parent in active code is retried like an invalid one. No-ops are now 0–6% for every arm (the remainder are parent copies after 10 failed tries). |
| 16 | `mse()` ranked NaN/Inf predictions better than finite ones with MSE above 1e30, which could corrupt champion selection. | `mse()` caps per-case error exactly as the fitness does, so non-finite is always worst. |
| 17 | CGP charged a constant once per program (as a cached module), where every other arm pays per read. | **[flag: counter]** CGP constants are ERC leaves, charged 1 per read. |
| 18 | A4's tuning space was broken: `n_trees` varied with the input and window sizes fixed at 10, so 10 trees had no second layer and 30 trees almost never initialised within the cap. | `n_trees` ∈ {10, 14, 20, 24}, with half the trees reading the inputs and a window of half the trees (the published proportions). Hierarchical initialisation falls back from "full" to "grow" before accepting a duplicate. |
| 19 | Minor: inlined sizes were computed for inactive CGP nodes (wasted time); screening had no checkpoints; the generation count was off by one at the budget boundary. | All fixed. |

Not changed:
- A3-lb1 and A3-lbmax are limited to depth 6 by their 6 columns. That is the ablation's documented design, not a bug.
- A4's readout is not covered by the reference interpreter; it has its own test.
- `encapsulate` often yields a structural rather than a functional change. With rule 15, its true no-ops are now retried, and its rate is tuned.

A 150-generation drift check of all 11 arms after these changes: cost ≤ expanded size everywhere, no reference mismatches, no growth, and time per generation similar or lower. 243 tests pass.

**G1 after these changes.** A1 is unchanged: every A1 node is live. Finding 15 changes A1's search too, so both arms were re-run on the Boolean tasks (code `f85d210`; table below). **ADFs cut even-5 effort by 23×** (1.33e8 against 3.09e9 units; 10× in individuals), and A2 now solves even-5 in 30 of 30 runs (A1: 16 of 30). The G1 known-result criterion holds. The SRBench check (A1) will be re-run on CERES.

**Next**
1. Michael: CERES setup, then the pilot array (`docs/CERES.md` §1–2). It measures speed only, so it can start now. Then the portability check (§ "Portability check").
2. The even-6 G1 batch (`configs/g1/parity6.yaml`, 60 runs) and the SRBench G1 check (`configs/g1/srbench.yaml`) on CERES, as the first real batches through the pipeline.
3. ChatGPT re-review of the changes since `7b85581`, before the tuning campaign.
4. From the pilot timings: decide whether screening keeps the 1e11 budget, and whether A2 needs Numba (its cheap programs make wall time per unit about 10× A1's).
5. The tuning campaign (`configs/wp2/tuning.yaml`, regenerated: 3,600 runs, 20 distinct configurations per arm, checkpointed), then `make_tuning.py report`, `make_screening.py write`, the screening, and the G2 test. Ask Michael at the G2 gate.

## 3 November 2026 (MacBook Pro) — slow A2 run diagnosed: a hole in the size cap

*Partly superseded by the review above: the functional-size check described here was replaced by the dead-code rule and A2's stored-size bound.*

### What changed
- **Size cap now also bounds functional size [flag: fairness-relevant, decided].** Every arm's offspring must satisfy expanded size ≤ 500, depth ≤ 17 **and functional size ≤ 500**. Functional size is the number of stored nodes, not counting the nodes that only pass a value on (ADF CALL and ARG nodes, and module REFs). Commit `c4bee68`.
  - *Why.* The expanded size inlines the program, so it leaves out (a) an ADF argument the body never reads and (b) an ADF that is never called. Under by-value semantics, (a) is still executed and charged. The cap therefore never saw this code, and A2 used it as an unbounded bloat reservoir. On even-6, one program had expanded size 443 and stored 4,397 nodes, and cost 5,077 units per case (`ADF1 := and(ARG1, ARG1)` with huge subtrees in its unused third argument). Population stored size reached 23k nodes, and time per generation kept rising. **This is why the A2 even-6 1e10 pilot run never finished.**
  - *Why it is neutral.* For A1, functional size equals expanded size. For A3–A6 it never exceeds expanded size, because every active module is read at least once (tested). For A2 without hidden code it is also at most the expanded size (tested). So the check only binds on hidden code. A raw stored-size cap was rejected, because it would charge the pass-through nodes that the counter treats as free and would slightly penalise legitimate modular programs. Koza also bounded ADF programs by their stored size.
  - The counter is unchanged. A1 is unchanged (same results, same random streams). A2's G1 numbers are re-run below.
- **Faster inlined size and depth** (commit `ad80cc9`): an exact integer fast path for the result-producing branch. These pure-Python passes had been about 40% of A2's run time, more than evaluation itself. The results matched the previous code on 13,500 bred programs, and a seeded run reproduces its trajectory exactly.

### After the fix (A2, even-6, seed 0, laptop)
- 1e10 run: functional size is capped. Stored size, counting pass-through nodes, levels off at about 1,100. Time per generation is flat.
- A2's evolved programs are cheap (about 34 units per case), so 1e10 means about 4.6M evaluations. Python's per-evaluation overhead, not the counter, sets wall time. *(Full-run time: see below.)*

### G1 Boolean, A1 and A2 re-run (final code after all three reviews, `f85d210`)
Same settings as on 2 November: 30 runs per cell, pop 500, at most 100 generations, lexicase. Earlier runs are kept in `results/g1_old_counter/`, `results/g1_superseded_cap/` and `results/g1_superseded_noop/`.

| Task | Arm | Solved | Effort (units) [95% CI] | Effort (individuals) | Median solution size |
| --- | --- | --- | --- | --- | --- |
| 6-mux | A1 | 30/30 | 1.73e7 [9.94e6, 1.73e7] | 4.5e3 | 33 |
| 6-mux | A2 | 30/30 | 4.26e7 [2.13e7, 5.57e7] | 9.5e3 | 92 |
| even-4 | A1 | 30/30 | 3.85e7 [1.93e7, 4.09e7] | 2.2e4 | 204 |
| even-4 | A2 | 30/30 | 3.38e7 [1.59e7, 3.46e7] | 2.3e4 | 280 |
| even-5 | A1 | 16/30 | 3.09e9 [2.10e9, 5.29e9] | 3.46e5 | 442 |
| even-5 | A2 | 30/30 | **1.33e8** [6.65e7, 1.47e8] | 3.35e4 | 290 |

- **ADFs reduce effort on even-5 parity by 23× in units** and 10× in individuals, with CIs far apart. ✔
- On even-4 the two arms are now level (CIs overlap). On 6-mux, A2 costs 2.5× A1: the task is too easy for ADFs to pay off. The advantage grows with problem size, as Koza (1994) reports.
- History of the even-5 ratio: 6.8× under the old counter, 11.6× after the first cap fix, 12.1× after the dead-argument rule, and 23× after the no-op offspring rule. Most of the last jump comes from A2 no longer spending about 30% of its offspring on dead-code changes.

## 3 November 2026 — G1 self-certified (gate waived); WP2 built; hand-over to the MacBook Pro

**Hand-over state (read this first on the Pro).** All local runs on the Air were stopped at 07:40. Nothing is running. Code and docs are in git; `results/`, `data/` and Claude's project memory travel through Box with `scripts/handover.py` (see CLAUDE.md).

### What changed since 2 November
- **G1 gate waived by Michael**; G1 is self-certified on the criteria below. WP2 has started.
- **Compute counter revised [flag].** One unit per node of the equivalent inlined tree that is actually executed, per case. Nodes that only pass a value on cost 0: A2's call and ARG nodes, and A3–A6's module REFs. ADF arguments are still passed by value, and a cached module output is charged once per evaluation. The independent review found that the old rule charged the modular arms 20–50% more than A1 for the same function, by a different amount per arm. A1 is unaffected; **A2's G1 Boolean numbers below were obtained under the old rule and must be re-run.**
- **WP2 arms built** (`lmgp/core/graph.py`, `lmgp/arms/{hier,a4_mmtgp,a3_cgp}.py`): A3, A3-lb1, A3-lbmax, A4, A5, A6, A6-inputs, A5-copy, A5-frozen. The design and every decision are in `docs/wp2-design.md`. An independent review was done and its findings fixed (commit `a11a70f`).
- **Tuning and screening campaigns** generated: `configs/wp2/tuning.yaml` (3,600 runs, 20 configurations × 9 arms × 4 tuning tasks × 5 seeds at 1e10), and `scripts/make_screening.py` (screening batch and the G2 test).
- **CERES resolved**: SGE, set up as in llm-primitives-stgp. Scripts are in `scripts/ceres/`, and the instructions and commands are in `docs/CERES.md`.
- 225 tests pass, including the hand-over tests.

### Pilot (laptop, old counter, indicative only)
Wall seconds per run (one seed; Boolean runs do not stop at a solution):

| Task | A1 1e8 / 1e9 / 1e10 | A2 1e8 / 1e9 / 1e10 |
| --- | --- | --- |
| comp L2 R2 | 2.5 / 28.7 / 179 | 3.4 / 20.7 / 241 |
| comp L4 R4 | 2.7 / 20.9 / 156 | 2.8 / 23.9 / 188 |
| even-6 | 5.6 / 48.8 / 480 | 6.1 / 35.5 / **not finished** |
| 11-mux | 0.7 / 5.1 / 34.5 | 1.7 / 6.1 / 44.2 |
| SRBench 227_cpu_small | 8.3 / 46.4 / 179 | 6.6 / 24.2 / 140 |

**Open problem: A2 on even-6 at 1e10** *(solved later the same day: see the MacBook Pro entry above).* The run had not finished after 7.3 hours; about 6 minutes was expected. A trace under the new counter (`scripts/trace_run.py`) shows:
- A2 runs at about 900 evaluations/s and 2.3e6 units/s, and gets slower as it goes.
- The population solves the task by generation 130 and keeps running to the budget, with mean expanded size about 325.
- That projects to more than 1.2 h per run. A1 runs at 2.1e7 units/s.

So under the new counter, A2 does far more evaluations per unit, because its call and ARG nodes no longer cost anything. Python's per-evaluation overhead then dominates. Why the old-counter run took over 7 hours is still unexplained. **First job on the Pro:** run `trace_run.py` to completion, profile it, and decide whether evaluation needs Numba (allowed when profiling shows the need). The pilot should then be repeated on CERES, not the laptop.

### Next steps, in order (superseded: see "Next" at the top)
1. Diagnose the slow A2 even-6 run (above).
2. Re-run A2's G1 Boolean cells under the new counter: `mkdir -p results/g1_old_counter && mv results/g1/a2__parity4* results/g1/a2__parity5* results/g1/a2__mux6* results/g1_old_counter/`, then `.venv/bin/python scripts/run_local.py configs/g1/boolean.yaml --out results/g1 --jobs 8`, then `scripts/g1_report.py --boolean`.
3. Run the even-6 G1 batch (`configs/g1/parity6.yaml`, not started), to check that the ADF advantage grows with n.
4. Michael: run the CERES setup and the pilot array from `docs/CERES.md`. From its timings, decide whether the screening keeps the 1e11 budget; G2 needs only two winning budgets.
5. Tuning campaign on CERES (`configs/wp2/tuning.yaml`), then `scripts/make_tuning.py report`, then `scripts/make_screening.py write`, then the screening, then the G2 test. Ask Michael at the G2 gate.

## 2 November 2026 — G1 (WP0 + WP1) complete (gate later waived)

### What changed

- **WP0 literature check** is in `docs/lit-check.md`. The research plan now treats A6 as a test of an existing idea (skip connectivity in modular GP) at scale, and adds two WP2 baselines: A3 with levels-back 1 versus maximum, and A6 with skips to the inputs only.
- **Shared core** (`lmgp/core`):
  - program interface;
  - vectorised NumPy evaluator;
  - compute counter;
  - linear scaling;
  - seeded RNG streams from `configs/seeds.txt`;
  - an independent scalar reference interpreter, used only by the tests.
- **Selection**: standard lexicase, and epsilon-lexicase with semi-dynamic MAD epsilon.
- **Arms A1 and A2**: one shared tree implementation (`lmgp/arms/tree.py`). A2 adds Koza-style ADFs: two of them, ADF0 with 2 arguments and ADF1 with 3, hierarchical.
- **Benchmarks**:
  - synthetic compositional regression (L, R, noise);
  - even-n parity;
  - k-multiplexer;
  - SRBench loader via `pmlb`.
- **Runner**:
  - driven by YAML configs;
  - Parquet logs per generation and per run;
  - checkpoint and resume;
  - package versions and git commit recorded with every run.
- **Scripts**:
  - local parallel batch runner (`scripts/run_local.py`);
  - G1 report (`scripts/g1_report.py`);
  - pilot timing and CPU-hour estimate (`scripts/ceres/pilot_timing.py`).
- **Tests**: 71 tests, all passing (`pytest`).

All G1 numbers below come from code at commit `0daf0b4`. They were run locally on Michael's MacBook Air (8 cores).

### Decisions taken (scientific decisions delegated, 2 Nov)

**Compute counter**
- One unit is one node evaluated on one training case. A program's cost is `cost_per_case × n_train`.
- Every ADF call is charged in full (call node + arguments + body), even with identical arguments. There is no memoisation in A1/A2.
- ADF arguments are passed by value: each argument subtree runs once per call, however often the body reads it. An ADF program can therefore cost less per case than its inlined equivalent; that is the reuse effect under study. *Counter-semantics choice; decided, see below.*
- Validation evaluations are not charged; they are logged separately (`val_units`).
- A run stops as soon as the next program's evaluation would exceed the budget. The budget is never overshot (this is tested).

**Size cap**
- The cap is 500 expanded nodes and depth 17. Size and depth are measured on the fully inlined program, so the cap is the same quantity for every arm. *Fairness-relevant; decided, see below. It costs A2 noticeably.*

**Population rule**
- pop = 250 at B = 1e8 and doubles with each tenfold budget increase.
- This gives 250, 500, 1000, 2000 and 4000 for budgets 1e8 to 1e12. The rule is the same for every arm.

**Initialisation**
- Ramped half-and-half over depths 2–6, rejecting duplicates and programs that break the cap.
- A slot that fails 20 times at its depth tries one level lower, with the same method.
- The step-down is logged (`init_depth_lowered`). A2 needs it for about 21% of slots; A1 almost never does.

**Variation**
- 90% subtree crossover (branch-typed) and 10% subtree mutation (grow, depth 0–4), with the 90/10 internal/leaf rule.
- An invalid offspring is retried up to 10 times; after that the first parent is copied.

**Selection and fitness**
- Lexicase for Boolean tasks; epsilon-lexicase (semi-dynamic MAD) for regression.
- Linear scaling on regression. Non-finite outputs get error 1e30.
- There is no fitness caching for clones: every evaluation is charged.

**Synthetic tasks**
- Library of 6 two-argument sub-functions (g0–g5); d = 5 inputs on U[−2, 2]; units standardised.
- 256 train, 256 validation and 2048 test cases.
- Noise is added to train and validation only; test is noise-free.
- Ground truth and sub-function definitions are stored with each task.

**SRBench**
- 75/25 train/test split per seed, as in SRBench.
- 25% of train is held out for validation (champion selection), so GP fits on about 56% of rows against SRBench's 75%.
- Training is capped at 10k rows.

### G1 results

**Validation suite: passes.** 71 tests, covering:
- offspring validity;
- size and depth caps;
- determinism for a given seed;
- checkpoint and resume;
- cost equals the reference interpreter's node count, for random and bred programs of both arms;
- identical programs cost the same;
- repeated ADF calls charged in full;
- `expanded_size` and depth equal an explicit textual inlining;
- the runner charges exactly the sum of program costs and never exceeds the budget.

*Caveat:* the criterion "a cached module output is charged once per program evaluation" has no counterpart in A1/A2, which have no module cache. That test will be added with A5 in WP2.

**Boolean known results** (*A2 rows: old counter, superseded; see 3 November*). 30 runs per cell, pop 500, at most 100 generations, lexicase, stop at the first exact solution. Effort is Koza's minimum computational effort at z = 99%, with 95% bootstrap CIs.

| Task | Arm | Solved | Effort (node-eval units) | Effort (individuals) | Median gen. to solution | Median solution size (expanded) |
| --- | --- | --- | --- | --- | --- | --- |
| 6-mux | A1 | 30/30 | 2.15e7 [1.08e7, 2.39e7] | 5.5e3 | 6 | 41 |
| 6-mux | A2 | 30/30 | 4.47e7 [4.19e7, 4.47e7] | 8.5e3 | 10 | 89 |
| even-4 | A1 | 30/30 | 4.03e7 [2.35e7, 4.03e7] | 1.65e4 | 15 | 214 |
| even-4 | A2 | 30/30 | 9.13e7 [3.43e7, 1.05e8] | 3.3e4 | 17.5 | 256 |
| even-5 | A1 | 15/30 | 2.90e9 [1.88e9, 4.89e9] | 3.46e5 | 84 | 417 |
| even-5 | A2 | 26/30 | 4.24e8 [2.00e8, 6.56e8] | 1.08e5 | 42.5 | 389 |

- Tree GP solves 6-mux and even-4 parity in every run, at under 5e7 units. ✔
- **ADFs reduce effort on even-5 parity**, by 6.8× in units and 3.2× in individuals, with non-overlapping CIs. ✔
- They do **not** reduce effort on even-4 or 6-mux, where A1 needs only 6–15 generations. Koza (1994) reports the ADF advantage growing with parity order. With lexicase, even-4 is easy enough that A2's extra branches are pure overhead. (I have not checked Koza's exact even-3 and even-4 figures against the book.)
- To confirm that the advantage grows with n, an even-6 batch (same settings) is queued after the pilot. Its results will be added here.
- Before the review fixes, when the cap was on the uninlined cost, A2 on even-5 reached 29/30 at an effort of 1.3e8. Moving the cap to the inlined size (the correct, arm-neutral quantity) costs A2 about 3×, because hierarchical ADF calls inflate the inlined size quickly. A2 still clearly beats A1.

**SRBench sanity check (A1).** 10 runs per dataset, pop 1000 × 500 generations = 500k evaluations (SRBench's cap). Values are median test R², compared with medians from the published SRBench black-box results.

| Dataset | lmgp A1 | IQR | gplearn | EPLEX | AFP | Operon | Linear |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1027_ESL | 0.852 | 0.021 | 0.815 | 0.857 | 0.831 | 0.858 | 0.872 |
| 227_cpu_small | 0.956 | 0.013 | 0.526 | 0.954 | 0.930 | 0.972 | 0.709 |
| 230_machine_cpu | 0.780 | 0.124 | 0.716 | 0.815 | 0.832 | 0.685 | 0.802 |
| 556_analcatdata_apnea2 | 0.836 | 0.058 | −0.075 | 0.882 | 0.907 | 0.857 | −0.001 |
| 561_cpu | 0.965 | 0.022 | 0.928 | 0.927 | 0.937 | 0.993 | 0.856 |

- Tree GP with epsilon-lexicase and linear scaling lands among the published GP methods on all five datasets. ✔
  - It is above gplearn everywhere.
  - It is within about 0.05 of EPLEX and AFP.
  - It does this while fitting on fewer rows, because of the validation holdout.
- Median expanded model size is 210–340 nodes.
- Compute calibration: 500k evaluations cost 1.5e10–5.4e11 units on these datasets. The planned top budget of 1e12 is therefore a few times SRBench's budget. The pilot below gives timings.

**Pilot timings and CPU-hour estimate.** *(Filled in when the pilot finishes; see `results/pilot/pilot_estimate.txt`.)*

### Flagged decisions and open items

*Decided rather than left open (Michael, 2 Nov: no need to wait on scientific or methodology matters). Flagged here because they touch the counter and fairness between arms:*

- **By-value ADF arguments: kept.** An argument subtree is evaluated once per call, as in Koza's implementation, and any real interpreter would do the same. Re-evaluating it at every read would charge A2 as if it were inlined, removing the reuse benefit by construction.
- **Size cap on the inlined size: kept.** It is the only reading of "expanded nodes" that is the same quantity for every arm. It costs A2 a lot on Boolean tasks, but a cap on stored size would let modular arms represent much larger functions than A1 under the same nominal cap. That would confound H1, and the plan already has A1-size for exactly this question.

1. **CERES** (*resolved 3 November: see `docs/CERES.md`*). The original questions were:
   ```
   which sbatch qsub bsub condor_submit; sinfo --version
   sinfo -o "%P %l %c %m"
   sacctmgr show assoc user=$USER format=account,partition,qos
   scontrol show config | grep -iE "MaxArraySize|MaxJobCount|MaxTime"
   module avail 2>&1 | grep -iE "python|anaconda|conda|uv"
   df -h $HOME /scratch 2>/dev/null
   curl -sI https://github.com | head -1   # do nodes/login have internet (for pmlb)?
   ```
2. **A4's connectivity: checked in the authors' code** (the paper is paywalled); details in `docs/lit-check.md` §2.11.
   - As published, a tree reads up to 10 earlier trees (a sliding window).
   - Only the first 10 of the 20 trees see the raw inputs.
   - Every tree feeds a ridge model at the output.
   - So A4 is already partly A6-like. The paper must state that A6's additions are raw inputs at every level and unbounded lower-level reads.
3. **Laptop timings.** The MacBook Air throttles under sustained load. Treat the CPU-hour estimate as indicative until the pilot is repeated on CERES as a single job.
