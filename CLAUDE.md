# CLAUDE.md

Guidance for Claude Code in this repository.

## Project

layered-modular-gp tests whether hierarchical, modular genetic programming turns extra compute into better generalisation after standard tree GP plateaus, and why. The full methodology is in `docs/research-plan.md`. Read it before any design decision.

Michael Kampouridis (University of Essex) leads the project and supervises. You are the research engineer who builds and runs it.

## Current milestone: G1 (work packages WP0 and WP1)

Build and validate the foundations, then stop and report. Do not start later work packages until Michael signs off G1.

In scope:

1. **WP0 literature check**, saved to `docs/lit-check.md`: skip or residual connections in GP; hierarchical, layered and modular GP (ADF variants, modular multi-tree GP, modular Cartesian GP, Tangled Program Graphs). For each: what it does, how it differs from arms A5 and A6, and what it means for novelty.
2. **Shared core**: program interface, vectorised evaluation, the compute counter, epsilon-lexicase selection, linear scaling, a config-driven runner and logging.
3. **Arms A1 (tree GP) and A2 (tree GP + ADFs).**
4. **Stage 1 benchmarks**: synthetic compositional regression, even-parity and multiplexer, plus an SRBench loader for a small subset (via the `pmlb` package).
5. **Validation suite** meeting the G1 criteria below.
6. **CERES pilot**: a timing script and a job-array template.

Out of scope until G1 is signed off: arms A3–A6, ablations, full scaling runs, Stage 2 and 3 experiments, finance. A3 and A4 are built in WP2, alongside A5.

## G1 pass criteria

- Property tests pass: offspring are valid, size and depth limits hold, and a run is deterministic for a given seed.
- Compute-counter tests pass: identical programs cost the same, and a cached module output is charged once per program evaluation.
- Known results reproduce: tree GP reliably solves the 6-bit multiplexer and even-4 parity at modest budgets; ADFs reduce computational effort on even-parity, as in Koza (1994); tree GP on a few SRBench datasets lands near published SRBench results.
- Pilot timings exist for each budget level, with a CPU-hour estimate for the full study.

Report the numbers in `docs/progress.md`, then stop.

## Arms (quick reference; full definitions in the plan)

| Arm | What it is |
| --- | --- |
| A1 | Tree GP with size and depth limits (main baseline) |
| A2 | Tree GP + automatically defined functions (ADFs) |
| A3 | Cartesian GP |
| A4 | Modular multi-tree GP (Zhang et al., IEEE TEVC 2024) |
| A5 | Hierarchical GP: an evolving library of modules in levels; each level reads only the level below |
| A6 | A5 plus skips: modules may also read raw inputs and any lower level |
| A5-copy | A5 with every module call given a private copy, so nothing is shared |
| A5-frozen | Random, frozen lower-level modules; only the top level evolves |
| A1-size | Tree GP allowed A5's expanded size |

## Hard rules

- Python only. Vectorise evaluation over training cases with NumPy; add Numba only where profiling shows a need.
- No paid compute or services. Large runs go to Essex's CERES cluster.
- Michael submits all CERES jobs. Before writing job scripts, ask him which scheduler and environment CERES uses. Tell him exactly what you need to know and give him the terminal commands, both to find it out and to submit.
- One compute counter for every arm: one unit per node evaluated per training case. Budgets are set in these units. Never change its semantics without flagging it to Michael; every comparison depends on it.
- Also log wall-clock time and program evaluations.
- Never tune on test tasks or test splits. Tuning uses separate tuning tasks, with the same tuning budget for every arm.
- Every run is reproducible from its config and seed. Seeds come from a fixed list in `configs/seeds.txt`.
- No experiment result counts until the G1 validation passes.
- Make small, clear commits and push straight to `main`; no pull requests.

## Working with Michael

- Keep updates short and concrete. Keep `docs/progress.md` current: date, what changed, results, open questions.
- Before building anything non-trivial, propose the design in a few lines and wait for his OK.
- Ask before deviating from the plan or making any choice that could affect fairness between arms. Give brief plain-English context with each question.
- Use British English in documents and comments.

## Design defaults (change only with Michael's agreement)

- In the first session, review Michael's EA/GP framework (github.com/MKampouridis/Evolutionary-Algorithms) and reuse what fits. The compute counter must still behave identically across all arms, so a fresh, lean core is the likely outcome.
- Function sets: regression {add, sub, mul, analytic quotient, sin, cos} plus ephemeral random constants; even-parity {AND, OR, NAND, NOR}; multiplexer {AND, OR, NOT, IF}.
- Selection: epsilon-lexicase with semi-dynamic epsilon (median absolute deviation) for regression; standard lexicase for Boolean tasks.
- Linear scaling of outputs for regression (Keijzer 2003).
- Ramped half-and-half initialisation; subtree crossover and subtree mutation, with rates set in config.
- The size cap is measured in expanded nodes and is identical across arms.

## Stage 1 synthetic tasks

- A fixed library of small two-argument sub-functions.
- Each target composes sub-functions over L levels (1–4), with each sub-function called R times (1–4), plus Gaussian noise at 0%, 1% or 10% of the target's standard deviation.
- Inputs are sampled uniformly on a fixed range, with separate train, validation and test samples.
- Save the ground-truth expression and the sub-function definitions with each task; the structure-recovery metric needs them later.

## Proposed layout (confirm with Michael in the first session)

```
lmgp/
  core/        program interfaces, evaluation, compute counter
  arms/        a1_tree.py, a2_adf.py; later a3–a6 and ablations
  selection/   lexicase variants
  bench/       stage 1 generators, parity, multiplexer, SRBench loader
  run/         config-driven runner, logging, checkpoints
  analysis/    metrics, scaling fits, statistics
configs/       YAML configs and seeds.txt
scripts/ceres/ job-array templates and the pilot timing script
tests/
docs/          research-plan.md, lit-check.md, progress.md
```

## Runs and outputs

- One run is (arm, task, budget, seed). Write per-generation logs and final metrics to Parquet.
- Checkpoint long runs so they can resume.
- Record Python and package versions with every batch of results.

## Two Macs, one active at a time

Michael works on a MacBook Air and a MacBook Pro. Git carries code and docs; `scripts/handover.py` carries `results/`, `data/` and Claude's project memory through Box (`handover_dir` in `config/paths.toml`, copied from `config/paths.example.toml`). Start a session with `.venv/bin/python scripts/handover.py pull` and end it with `... push`, after committing and pushing. Check `... status` first if unsure which Mac holds the project. Stop local runs before pushing.
