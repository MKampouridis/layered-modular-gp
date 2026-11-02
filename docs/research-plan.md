# Deep Hierarchical GP: Methodology Research Plan

Michael Kampouridis · version of 2 November 2026

This plan tests whether hierarchical, modular GP converts extra compute into better generalisation after standard tree GP plateaus, and why.

## Research question and hypotheses

Can a GP representation that builds and reuses modules across levels keep improving on unseen data as compute grows, after tree GP plateaus?

| ID | Hypothesis | Rejected if |
| --- | --- | --- |
| H1 Scaling | Hierarchical GP keeps lowering test error at budgets where tree GP has plateaued. | Fewer than half the Stage 1 tasks show a lower fitted error floor for a hierarchical arm (non-overlapping 95% bootstrap intervals). |
| H2 Reuse | The gain comes from discovered, reused modules, not from extra search or a bigger model. | The no-sharing and size-matched controls perform as well as the hierarchical arm. |
| H3 Information flow | Skip connections keep deep modules responsive to variation, so deeper hierarchies stay useful. | Skips change neither mutation sensitivity at depth nor test error. |
| H4 Structure recovery | On tasks built from known sub-functions, evolved modules match those sub-functions more often than tree GP or ADFs do. | Recovery rates do not differ. |
| H5 Finance (exploratory) | Hierarchical GP matches or beats tree GP out of sample on DC-based tasks, with smaller models. | Reported either way; not a pass/fail test for the programme. |

H1 and H2 are the core claims. H3 and H4 explain them, and H5 tests transfer to finance.

## Methods compared

Nine arms share one codebase, selection scheme, function set and evaluation counter, so only the representation differs.

| Arm | Representation | Purpose |
| --- | --- | --- |
| A1 Tree GP | Standard trees with size and depth limits | Main baseline |
| A2 Tree GP + ADFs | Trees with Koza's automatically defined functions | Classic modularity baseline |
| A3 Cartesian GP | Graph of nodes with implicit reuse, standard single-row settings | Reuse without explicit modules |
| A4 Modular multi-tree GP | Trees that read other trees' outputs ([Zhang et al., IEEE TEVC 2024](https://gpbib.cs.ucl.ac.uk/gp-html/Hengzhe_Zhang_ieeeTEC.html)) | Recent published modular baseline |
| A5 Hierarchical GP (proposed) | Evolving library of small modules in levels; each level reads only the level below | Tests H1 and H4 |
| A6 Hierarchical GP + skips (proposed) | A5, but modules may also read raw inputs and any lower level | Tests H3 |
| A5-copy (ablation) | A5 with every module call given a private copy, so nothing is shared | Isolates reuse (H2) |
| A5-frozen (ablation) | Random lower-level modules, frozen; only the top level evolves | Learned versus random features (H2) |
| A1-size (control) | Tree GP allowed A5's expanded size | Capacity versus structure (H2) |

WP0 found close precedents for both proposed arms (ADFs, module acquisition, ARL, DL'GP and embedded CGP for A5; CGP levels-back and Modular GP-GOMEA for A6). The contribution is therefore the compute-matched scaling test and its controls, not the representations. Two cheap additions are noted for WP2: A3 with levels-back 1 versus maximum (a skip ablation inside CGP), and an A6 variant with skips to raw inputs only.

A5's variation operators are module mutation, rewiring a call, encapsulating a subtree as a new module, and deleting unused modules. All arms use epsilon-lexicase selection and linear scaling of outputs. For trading policies in Stage 3, A5 takes the team-and-graph form of Tangled Program Graphs ([Kelly & Heywood 2017](https://gpbib.cs.ucl.ac.uk/gp-html/Kelly_2017_EuroGP.html)).

## Benchmarks

Benchmarks run in three stages: known structure first, public benchmarks second, finance last.

| Stage | Tasks | What it tests |
| --- | --- | --- |
| 1 Controlled | Synthetic regression built from a known library of sub-functions, varying levels (1–4), reuse (1–4 calls per sub-function) and noise (0%, 1%, 10%). Boolean even-parity (5–8 bits) and multiplexer (11 and 20 bits). | H1–H4, with ground truth for structure recovery; parity is the classic case where ADFs help |
| 2 Public | SRBench ground-truth sets (Feynman, Strogatz) and a black-box subset ([La Cava et al. 2021](https://arxiv.org/abs/2107.14351)); a PSB2 subset as a stretch goal ([Helmuth & Kelly 2021](https://arxiv.org/abs/2106.06086)) | External validity and comparison with published results |
| 3 Finance | Semi-synthetic series: a known compositional signal plus noise at falling signal-to-noise ratios. Then DC-based forecasting and trading on real data, walk-forward, after costs. | H5; separates "hierarchy does not help" from "noise hides it" |

Stage 1 decides H1–H4. Stage 3 starts only after gate G3.

## Experimental protocol

Every comparison is made at equal compute, counted in node evaluations, with equal tuning effort for every arm.

- **Compute accounting.** One counter charges every node evaluated on every training case, for all arms. A cached module output is charged once per program evaluation. Wall-clock time is reported alongside.
- **Scaling axis.** Five budgets, each ten times the last; the range is set in the pilot. Population size grows with budget by one fixed rule for all arms.
- **Tuning.** Each arm gets the same random-search budget (20 configurations) on separate tuning tasks, never the test tasks.
- **Data splits.** Synthetic and SRBench tasks use train, validation and test sets, with the final model chosen on validation. Finance uses walk-forward windows; test periods stay unseen during design.
- **Replication.** 30 independent runs per arm, budget and task, from a fixed seed list.
- **Statistics.** Medians with interquartile ranges. Mann–Whitney U with Holm correction within a task; Friedman test with post-hoc comparisons across tasks; Vargha–Delaney A effect sizes.

Scaling is summarised per arm and task by fitting test error against budget C:

```math
\text{error}(C) = a + b\,C^{-\alpha}
```

The error floor a and exponent α are compared with bootstrap 95% intervals. The plateau budget is the smallest budget after which median error improves by less than 1% per tenfold increase.

## Metrics and mechanism analysis

Performance metrics show whether hierarchy helps; the mechanism measures show why, and they decide H2–H4.

| Question | Measure |
| --- | --- |
| Does it generalise better? | Test RMSE or R² (regression); exact-solution rate (Boolean); out-of-sample Sharpe ratio, return and drawdown after costs (finance) |
| Does it scale? | Fitted error floor and exponent; plateau budget; budget needed to reach a target error |
| How big is the model? | Expanded size versus stored (shared) size; number of modules; hierarchy depth |
| Are modules reused? | Calls per module; share of output variance carried by reused modules |
| Do modules matter? | Knockout: replace a module by its mean output and measure the rise in error |
| Is the true structure found? | Stage 1: best correlation between each true sub-function and any evolved module on probe inputs |
| Does depth stay evolvable? | Mutation sensitivity: share of mutations at each level that change program output, following Langdon's work on deep trees ([Langdon & Banzhaf 2022](https://discovery-pp.ucl.ac.uk/id/eprint/10152700)) |
| Do modules transfer? | Seed a run on task B with task A's module library; measure the budget saved |

Lineage logging records when each module appears, how often it is called and when it disappears. That turns "it scales" into a description of what was learned.

## Implementation and compute

Claude Code builds the system in Python, Michael reviews the operators, and experiments run on CERES at no cost.

1. **Codebase.** One Python package with all nine arms behind a common interface. Evaluation is vectorised over training cases with NumPy, adding Numba only where profiling shows a need. Michael's [Evolutionary-Algorithms](https://github.com/MKampouridis/Evolutionary-Algorithms) framework is reused where it fits.
2. **Validation before any experiment.** Property tests on every operator (offspring valid, module calls acyclic). Budget-accounting tests (identical programs, identical charges). Known results reproduced, such as ADFs helping on even-parity and tree GP on SRBench near published figures.
3. **Independent review.** A second agent reviews the operators and the evaluation counter without seeing the build. Fast-written evolutionary code can produce convincing but wrong curves.
4. **Runs on CERES.** One run is one task in an array job, set by a config file and seed. Results go to Parquet files, with checkpoints so long runs can resume. Michael submits the jobs; Claude prepares the scripts and the exact commands.
5. **Cost.** Measured in the WP1 pilot. A first guess for the full Stage 1–2 ladder is of the order of 10,000 CPU-hours, dominated by the top budget.
6. **Reproducibility.** This repository holds the configs, seed lists, raw results and analysis notebooks.

## Work plan

| Work package | Months | Gate at the end |
| --- | --- | --- |
| WP0 Set-up and literature check | M0–M1 | |
| WP1 Baselines and validation | M1–M2 | G1 Baselines validated (M2) |
| WP2 Hierarchical design screening | M2–M4 | G2 Go/no-go on hierarchy (M4) |
| WP3 Scaling study and ablations | M4–M6 | G3 Scaling result confirmed (M6) |
| WP4 Mechanism analysis | M6–M8 | |
| Paper 1 writing | M7–M9 | Paper 1 submitted (M9) |
| WP5 Finance and noise ladder | M8–M12 | Paper 2 draft (M12) |

G2 passes if a hierarchical arm beats tree GP at two or more budgets on at least half the Stage 1 tasks. If it fails, the fallbacks under Risks replace WP3.

## Risks and fallbacks

The main risk is that hierarchy does not scale. Each risk has an early sign and a fallback, so even a no-go yields a publishable result.

| Risk | Early sign | Fallback |
| --- | --- | --- |
| No scaling benefit | G2 fails: no hierarchical arm beats tree GP on Stage 1 | Test skips inside tree GP alone; or make LLM-guided variation the scaling lever, as in [FunSearch](https://www.nature.com/articles/s41586-023-06924-6); or publish a rigorous negative result with the mechanism measures |
| Module bloat | Module count grows while reuse falls | Cap the library size; add size as a second objective; prune unused modules |
| Unfair-comparison critique | Gains vanish under the size-matched control | Report A1-size and A5-copy prominently; keep tuning budgets equal |
| Evaluation cost too high | Pilot run times at the top budget | Cache module outputs; use down-sampled lexicase; drop the top budget on Stage 2 |
| Bugs that produce false curves | Known results not reproduced | G1 blocks all experiments until validation passes |
| Finance noise hides any effect | Gain shrinks to zero along the signal-to-noise ladder | Report the noise threshold itself as the finance result |
| Prior work on skips in GP | Confirmed in WP0 (`docs/lit-check.md`) | Applied: A6 is framed as a test of an existing idea at scale; A5's novelty rests on the compute-matched scaling study and the controls |

## Sources

- [Langdon & Banzhaf (2022), Long-Term Evolution Experiment with GP, Artificial Life](https://discovery-pp.ucl.ac.uk/id/eprint/10152700)
- [Kelly & Heywood (2017), Emergent Tangled Graph Representations for Atari, EuroGP](https://gpbib.cs.ucl.ac.uk/gp-html/Kelly_2017_EuroGP.html)
- [Zhang et al. (2024), Modular Multi-Tree GP for Feature Construction, IEEE TEVC](https://gpbib.cs.ucl.ac.uk/gp-html/Hengzhe_Zhang_ieeeTEC.html)
- [La Cava et al. (2021), SRBench, NeurIPS Datasets and Benchmarks](https://arxiv.org/abs/2107.14351)
- [Helmuth & Kelly (2021), PSB2, GECCO](https://arxiv.org/abs/2106.06086)
- [Romera-Paredes et al. (2024), FunSearch, Nature](https://www.nature.com/articles/s41586-023-06924-6)

Koza's ADFs, Cartesian GP and lexicase selection are cited from memory. The WP0 literature check (`docs/lit-check.md`) found that skip-like connectivity is established in GP under other names (CGP levels-back, linear GP registers, Koza's result-producing branch, Modular GP-GOMEA), so A6 is framed as a controlled test of an existing idea at scale.
