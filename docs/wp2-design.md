# WP2 design: arms A3–A6 and the screening protocol

Decided on 3 November 2026, under Michael's delegation of scientific and methodology decisions. Choices that touch the counter or fairness between arms are marked **[flag]** and are copied to `docs/progress.md`.

## 1. One graph representation for A3–A6

A3 (CGP), A4 (modular multi-tree), A5 and A6 are all directed acyclic graphs of computed values. They share one program class, `GraphProgram`, and one evaluator. The arms differ only in:
- which earlier values each node may read (connectivity);
- the shape of their nodes;
- their variation operators.

That keeps the compute counter identical by construction.

**The GraphProgram class**
- A `GraphProgram` is an ordered list of **modules** m_1 … m_n followed by an **output tree**.
- Each module body is a prefix tree over the primitive set. Its leaves are ERCs or **REF** terminals.
- A REF terminal reads a raw input x_j or the output of an earlier module.
- Each module is assigned a level. The arm's connectivity rule fixes which REFs a module at level l may hold. The output tree is treated as one level above the top.

**Evaluation**
- Modules reachable from the output (*active* modules) are evaluated once each, in order, per program evaluation.
- Each module's output vector is cached, and every REF to it reads the cache.
- Inactive modules are not evaluated.

## 2. Compute accounting (the same counter as A1/A2) **[flag]**

- **One unit per node of the equivalent inlined tree that is executed, per training case** (revised 3 November 2026 after the independent review; this rule also applies to A2).
  - Function, constant and input-REF nodes cost 1. An input REF is a variable leaf.
  - A module REF costs 0. It only passes on a value already computed and vanishes on inlining, as A2's call and argument nodes do; those now also cost 0.
  - So a program without reuse costs exactly what A1 pays for the same function.
  - The first version charged every REF 1 unit. That made the graph arms pay 20–50% more than A1 for the same function, by a different amount per arm.
- **Cost per case** = the sum of body sizes of the active modules + the output tree size.
- **A cached module output is charged once per program evaluation**, the G1 criterion deferred to WP2. Further reads are free.
- **A5-copy** gives every module read a private copy, in its genes as well as its computation. After every operator, a module read by several REFs is duplicated, so each REF owns its module. Its cost therefore equals its expanded size, as for a tree. The per-level module limit is lifted for A5-copy (set to 500, which can never bind; it was 64 until the pre-CERES review, and that did bind); the shared expanded-size cap bounds it.
- **Expanded size** is the size of the fully inlined tree: each REF is replaced by a copy of the referenced module's inlined body. Expanded depth is defined the same way.
  - The cap stays at 500 expanded nodes and depth 17, identical for all arms, as CLAUDE.md requires.
  - **Dead code is free and uncapped, in every arm [flag, pre-CERES review, 3 Nov].** Inactive modules (A3–A6) and ADF arguments that the inlined body never reads (A2) cannot reach the output, so they are neither executed nor charged, and the caps measure live code only. Hence cost per case ≤ expanded size for every arm (property-tested). A2's stored size, dead code included, is bounded at 1,000 nodes as an engineering limit on genotype length; the graph arms' storage is bounded by construction.
  - Reuse therefore cannot buy a larger function. What it can buy is cheaper evaluation, since cost per case is less than expanded size, so more evaluations fit in a budget. It also gives a more compact genotype for search to work on.

**Consequence for A1-size [flag, decide before WP3].** With an identical expanded-size cap, A5 can never exceed A1's cap, so "A1 allowed A5's expanded size" coincides with A1. A1-size only has meaning if the modular arms' cap is lifted to the same cap on *stored* size. That would change the design default in CLAUDE.md, which says to change it only with Michael's agreement.
- My recommendation for WP3: keep the identical expanded cap for every main comparison.
- In addition, run one "relaxed" pair, A5 with a stored-size cap of 500 and A1-size with A5's median realised expanded size, as the capacity control.
- WP2 screening does not need A1-size.

## 3. The arms

| Arm | Modules and levels | What a module at level l may read | Output |
| --- | --- | --- | --- |
| **A5** | L = 4 levels; 1–8 modules per level (variable) | modules of level l−1 only; level 1 reads the raw inputs | output tree reads level L only |
| **A6** | as A5 | the raw inputs and every module of levels < l | output tree reads every level and the inputs |
| A6-inputs | as A5 | level l−1 and the raw inputs | output tree reads level L and the inputs |
| **A4** | 20 trees, sliding window of 10, depth ≤ 2 each (the published MMTGP configuration) | trees x−10 … x−1; trees 0–9 also read the raw inputs, trees 10–19 do not | least-squares readout over all 20 trees (A4's own representation) **[flag]** |
| **A3** | standard single-row CGP: 1 row, 100 nodes, levels-back = 100 (Miller's recommended default); each node is one primitive whose arguments are REFs | any earlier node or input | one output gene |
| A3-lb1 | 10 rows × 6 columns, levels-back 1 | the previous column only (the first column reads the inputs): a strictly layered network | one output gene reading the last column. With levels-back 1, every active path crosses every column, so the column count is the depth and must respect the caps (6 levels give at most 127 expanded nodes) |
| A3-lbmax | same 10 × 6 grid, levels-back 6 | any earlier column or input | as above |

- **A5's module bodies:** grow trees, initial depth 1–2 and depth at most 4 per module. A single-REF module is a pass-through, so programs whose true structure has fewer than L levels stay expressible.
- **A5 and A6 initialisation:** each level starts with 2–4 modules, drawn uniformly, plus a ramped output tree.
- **A4's readout [flag]:** A4's least-squares readout is part of its published representation. The other arms use the standard two-parameter linear scaling of a single output.
  - I keep A4 as published so that it is a faithful baseline.
  - Fitting the readout is not charged, as linear scaling is not. But its k − 1 extra weighted terms beyond a + b·f are charged and counted as nodes: 3(k − 1) per case (weight constant, multiplication, addition), in both cost and expanded size. (Revised after the WP2 review; the weight constants were added in the pre-CERES review, since every ERC costs 1.)
  - The readout's sum also counts towards depth: the depth of the shallowest equivalent tree, less the two levels that linear scaling adds, uncounted, for every arm.
  - The full readout is charged even when the fit gives a tree zero weight (a constant tree): it is part of A4's fixed representation, as a constant subtree is part of an A1 program.
  - The synthetic targets are sums of top-level units, which favours a linear readout. This is a caveat for the paper.

## 4. Variation operators

All rates are config parameters. In WP2 they are set by the equal-effort tuning (§6).

**A5 and A6 share all of these:**
1. **Subtree crossover.** Pick a module in parent 1, with probability proportional to body size, including the output tree. The partner is the module at the same level and index in parent 2, or a random module of the same level if parent 2 has no module at that index. Swap in a random subtree under the 90/10 rule. The REF terminals that arrive must be legal at the receiving position; any that are not are re-drawn among the legal ones.
2. **Module mutation.** Subtree mutation inside one module body, with grow depth 0–2.
3. **Rewiring.** Re-point one REF leaf to another legal target.
4. **Encapsulation.**
   - Choose a module M at level l < L and a subtree S of its body of depth ≥ 1. S reads exactly what M may read, so it is a valid level-l module.
   - Add S as a new module N at level l, then re-point one random REF at level l+1, or in the output tree if l = L, to N.
   - If level l is full, the operator is not applied.
   - **Encapsulation is not semantics-preserving.** Strict levels rule out replacing S inside M by a read of N, because M would then read its own level. The operator is therefore a structured mutation that offers a useful subtree as a reusable module one level up.
5. **Deletion.** Remove one random module of a level holding more than one, and re-point every REF to it to another module of the same level. In addition, *unused* modules (inactive and unreferenced) are removed after every operator. They would never be evaluated, but they distort stored size and the structure-recovery measures.

**The ablations use the same operators:**
- **A5-copy** is A5 with private copies (§2).
- **A5-frozen** draws levels 1 … L−1 at random once and never varies them; only level L and the output tree evolve.

**A4** uses its published operators:
- per-tree subtree crossover at matching indices (the macro crossover), with partner trees drawn from the same block of 10;
- per-tree subtree mutation.

MMTGP's adaptive operator rates and its EDA terminal mutation are **not** reproduced. Every arm uses fixed rates chosen by the same tuning procedure, so tuning effort stays equal. This is noted as a deviation.

**A3** uses Goldman and Punch's (2013) single-active mutation and no crossover, as is standard for CGP. Genes are mutated one at a time until an active gene changes, so no evaluation is spent on a phenotypically identical offspring. It runs under the shared lexicase loop. Using population-based lexicase selection rather than Miller's (1+4)-ES keeps "only the representation differs" intact; this is noted as a deviation from CGP's usual practice.

**A3-lb1 on Boolean tasks:** with only binary functions, every phenotype is the full depth-6 tree (127 nodes), so its size cannot vary. This is a property of the strictly layered grid, noted for interpretation.

## 5. Shared loop

Selection is epsilon-lexicase (regression) or lexicase (Boolean), with linear scaling, the counter and the stopping rule all unchanged. For every arm, an offspring that breaks the cap, **or is identical to a parent in its active code** (a change to dead code only), is retried up to 10 times, and the parent is copied after that **[flag, pre-CERES review]**. CGP constants are leaves charged per read, as constants are in every other arm.

## 6. Screening protocol (WP2)

**Goal:** the G2 decision. "A hierarchical arm beats tree GP at two or more budgets on at least half the Stage 1 tasks."

**Tuning**
- 20 configurations per arm, from arm-specific ranges covering the operator rates, module counts and depths: the default plus 19 random, all distinct in the settings the arm actually runs with (redrawn on a duplicate).
  - CGP has no operator rates to tune (single-active mutation), so A3 tunes its grid (rows 1/2/4, columns 50–300, levels-back 20–100% of the columns), and A3-lb1 and A3-lbmax tune their number of rows (2–40). Their 6 columns and levels-back are the ablation itself. The number of constants is not tuned: it has no effect on Boolean tasks, and tuning it made most CGP configurations duplicates (pre-CERES review).
  - A4 tunes its tree count (10, 14, 20 or 24) with the published proportions kept: half the trees read the inputs and the window is half the trees (pre-CERES review).
- Tuning runs on **separate tuning tasks**: synthetic instances 100–102 with (L, R, noise) = (2, 2, 1%), (3, 3, 1%), (4, 2, 1%), and even-5 parity. Even-5 is therefore **not** a screening task: parity has no instances, so it is the same task whatever the seed (pre-CERES review).
- Tuning runs checkpoint every 25 generations.
- The report writes the chosen configurations only when every planned tuning run is present.
- 5 seeds per configuration, at budget 1e10.
- Each configuration is ranked by median validation error on the tuning tasks, and the best one is fixed for that arm.
- The tuning budget is identical for every arm, and the screening and test tasks (instances 0–2) are never used.

**Screening grid**
- The 48 synthetic cells (L 1–4 × R 1–4 × noise 0/1/10%), instance 0.
- Even-parity 6–8 and 11-multiplexer (even-5 is a tuning task).
- Budgets 1e8 to 1e11 (four of the five levels; 1e12 is kept for WP3), and 10 seeds per cell.

**Arms screened:** A1, A2, A3, A3-lb1, A3-lbmax, A4, A5, A6, A6-inputs.

**Compute:** this is sized from the pilot, and runs on CERES.

**G2 test**
- For each task, compare each hierarchical arm with A1 at each budget. The test is a Mann–Whitney U on test error, Holm-corrected within the task.
- An arm beats tree GP at a budget if the difference is significant and A1 has the higher median error.
- G2 passes if one hierarchical arm, specified before the runs, achieves two or more budgets on at least half the Stage 1 tasks.
- The arm is specified as A5 or A6, whichever scores more on the tuning tasks. Fixing it in advance avoids multiple comparisons over arms.
  - Rule: compare the tuned A5 and tuned A6 on each tuning task by median error (Boolean: then median units to solution); the arm better on more tasks is the G2 arm, and a tie goes to A5. The tuning report writes it to `tuned.yaml` (`g2_arm`), and the G2 test reads it from there.
- The per-run score is test MSE on regression. On Boolean tasks it is units to solution ÷ budget for a solved run and 1 + final error for an unsolved one, so solving with less compute counts (pre-CERES review).
- The G2 verdict needs every planned run of A1 and the G2 arm; "half the tasks" is half of all planned tasks. Missing runs give INCOMPLETE, never PASS.

## 7. Build order

1. `GraphProgram`, its evaluator and the counter, with tests: the cached module charged once, A5-copy equal to the inlined cost, expanded size equal to explicit inlining, and cost equal to the reference interpreter.
2. A5 and A6 (and A6-inputs), with property tests on every operator: offspring valid, references legal and acyclic, caps respected, deterministic.
3. A4.
4. A3 and its two variants.
5. An independent review of the operators and the counter, by a second agent that does not see this document's reasoning.
6. The tuning and screening configs; the CERES commands for Michael.
