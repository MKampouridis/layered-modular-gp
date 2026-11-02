# WP0 literature check

2 November 2026

## Purpose and scope

This note checks prior work that bears on the novelty of the two proposed arms in `docs/research-plan.md`:

- **A5, hierarchical GP**: an evolving library of small modules in levels; each level reads only the level below; modules are shared across calls. Operators: module mutation, rewiring a call, encapsulating a subtree as a new module, and deleting unused modules.
- **A6, hierarchical GP with skips**: A5, but modules may also read raw inputs and any lower level.

The project's central claim is that such hierarchy turns extra compute into better generalisation after tree GP plateaus (H1), because of reuse (H2), helped by information flow through skips (H3).

Searches used the web, the GP Bibliography (gpbib), arXiv and the Consensus paper index. Each entry gives a citation, a URL checked during this review, what the work does, how it differs from A5 and A6, and what it means for novelty. Citations marked **unverified** could not be confirmed online during this check. The two citations that most affect the novelty verdict, Modular GP-GOMEA (§1.4) and DL'GP (§2.5), were checked a second time against their arXiv and gpbib pages.

---

## 1. Skip, residual and shortcut connections in GP

**Summary.** No GP paper found uses the words "residual" or "skip connection" for evolved symbolic programs in the ResNet sense. But the mechanism itself, letting a node read raw inputs or any earlier layer rather than only the one immediately below, is long established in graph-based GP. It goes by other names: levels-back, register reuse, DAG connectivity, "any earlier tree". The plan's statement that "web searches found no GP-specific work on skip connections" should be corrected.

### 1.1 Cartesian GP and the levels-back parameter

- **Citation.** J. F. Miller and P. Thomson (2000). Cartesian Genetic Programming. *EuroGP 2000*, LNCS 1802, pp. 121–132, Springer. <https://gpbib.cs.ucl.ac.uk/gp-html/miller_2000_CGP.html>
- **What it does.** CGP encodes a program as an indexed acyclic graph of nodes laid out in columns. The levels-back parameter *l* lets a node take inputs from any of the previous *l* columns, and program inputs are available too. With *l* = 1 the graph is strictly layered; with a single row and *l* equal to the number of columns, any feed-forward DAG is allowed.
- **Versus A5 and A6.** CGP with *l* > 1 is, structurally, a skip-connected layered graph: a node can read raw inputs and any earlier layer. That is exactly A6's connectivity rule, applied at the level of single nodes rather than modules. CGP has implicit reuse (a node's output can feed several later nodes) but no explicit module library or encapsulation in its basic form.
- **Novelty.** This is the main prior art for A6's connectivity. A6 cannot be described as introducing skip connections to GP. Levels-back should be named explicitly in any paper as the GP antecedent of skips.

### 1.2 Empirical evidence that levels-back and register reuse matter

- **Citation.** L. F. D. P. Sotto, P. Kaufmann, T. Atkinson, R. Kalkreuth and M. P. Basgalupp (2021). Graph representations in genetic programming. *Genetic Programming and Evolvable Machines* 22(4):607–636. doi:10.1007/s10710-021-09413-9. <https://gpbib.cs.ucl.ac.uk/gp-html/Sotto_GPEM.html>
- **What it does.** It compares CGP, linear GP, evolving graphs by graph programming (EGGP) and tree GP under several evolutionary algorithms. The authors report that the reuse of intermediate results, set by LGP's number of registers and CGP's levels-back, is "of utmost importance" for convergence on problems that benefit from code reuse.
- **Versus A5 and A6.** This is the closest empirical study of a skip-like connectivity setting in GP. But it measures convergence and search performance, not generalisation as compute grows. It does not separate reuse from skip connectivity, and it has no module library.
- **Novelty.** It gives prior support for H3-style effects. A6 should cite it and position H3 as a controlled, compute-matched extension that also measures generalisation and mutation sensitivity at depth.

### 1.3 Linear GP registers

- **Citation.** M. Brameier and W. Banzhaf (2007). *Linear Genetic Programming*. Springer. **Unverified** (no publisher page checked; widely cited, including in Sotto et al. 2021).
- **What it does.** Instructions read and write a small set of registers, so a program decodes to a DAG in which any later instruction can read any earlier result or input register.
- **Versus A5 and A6.** As with CGP, any-earlier-result connectivity is inherent; there are no explicit levels or a module library.
- **Novelty.** Further prior art for skip-like connectivity; no direct threat to A5.

### 1.4 Modular GP-GOMEA (closest recent anticipation of A6-style connectivity)

- **Citation.** J. Harrison, P. A. N. Bosman and T. Alderliesten (2025). Thinking Outside the Template with Modular GP-GOMEA. arXiv:2505.01262 (preprint; no peer-reviewed venue listed). <https://arxiv.org/abs/2505.01262>
- **What it does.** It evolves several small trees per individual for symbolic regression. "Any tree at location *i* in the representation can make use of any tree *j* with *j* < *i*." Subexpression terminal sets include input features as well as function arguments, and an appendix compares this against Koza's arguments-only formulation. The authors report that it generally beats single-template GP-GOMEA and recovers ground-truth expressions faster on synthetic data.
- **Versus A5 and A6.** Its connectivity is a dense, skip-connected DAG of reusable sub-expressions: any earlier module plus raw inputs, which is essentially A6's rule. It has no explicit level structure, no encapsulation or deletion operators, no A5-style strictly layered contrast, and no compute-matched scaling study. Comparisons use wall-clock budgets.
- **Novelty.** This anticipates A6's representation closely. It also anticipates part of H4: recovering known sub-expressions on synthetic tasks. It does not test layered versus skip-connected structures. **Flag: close anticipation of A6's connectivity rule.**

### 1.5 Koza ADFs: the result-producing branch already "skips"

- See §2.1. In Koza's ADF architecture, the result-producing branch can read the raw inputs and call every ADF. With hierarchical references, a later ADF can also call earlier ADFs, so the top of the hierarchy already has skip-like access. A6 differs in giving skips to every module at every level, not only to the top branch.

### 1.6 GP feature learning that reuses internal-node features

- **Citation.** Q. Fan, Y. Bi, B. Xue and M. Zhang (2023). Genetic Programming for Image Classification: A New Program Representation With Flexible Feature Reuse. *IEEE Transactions on Evolutionary Computation* 27(3):460–474. doi:10.1109/TEVC.2022.3169490. <https://gpbib.cs.ucl.ac.uk/gp-html/QinglanFan_ieeeTEC.html>
- **What it does.** It points out that tree-based GP image classifiers normally use only the root's output. The proposed structure "can flexibly reuse features generated from different nodes including internal nodes" of the program.
- **Versus A5 and A6.** This is a skip or dense-connection idea in spirit: intermediate representations are exposed downstream. It is limited to image feature learning, has no module library and no scaling study, and the abstract does not give the routing details.
- **Novelty.** Minor additional prior art on exposing intermediate representations; cite as related work.

### 1.7 Neural and neuroevolution-adjacent uses of skips

- **OccamNet.** O. Dugan et al. (2020+). OccamNet: A Fast Neural Model for Symbolic Regression at Scale. arXiv:2007.10784. <https://arxiv.org/abs/2007.10784> It is a neural symbolic-regression model whose layers of primitive functions use DenseNet- and ResNet-style skip connections. The authors argue the skips let the model draw on every level of composition and reuse shallow primitives. It is not evolutionary, but it is the clearest prior use of explicit skips over layers of symbolic primitives. The author list beyond the first author was not checked.
- **CGP-designed CNNs.** M. Suganuma, S. Shirakawa and T. Nagao (2017). A Genetic Programming Approach to Designing Convolutional Neural Network Architectures. arXiv:1704.00764 (GECCO 2017). <https://arxiv.org/abs/1704.00764> CGP evolves CNN architectures; levels-back and the summation and concatenation nodes allow ResNet-like shortcut connections.
- **Differentiable CGP for neural architecture search.** M. Märtens and D. Izzo (2019). Neural network architecture search with differentiable Cartesian genetic programming for regression. *GECCO 2019 Companion*. Found via Consensus; **URL not independently verified.** The evolutionary operators "introduce skip connections" among other changes.
- **NEAT.** K. O. Stanley and R. Miikkulainen (2002). Evolving Neural Networks through Augmenting Topologies. *Evolutionary Computation* 10(2):99–127. PDF: <https://www.cse.unr.edu/~sushil/class/gas/papers/NEAT.pdf>. The DOI 10.1162/106365602320169811 is **unverified** (seen only in a secondary source). NEAT evolves arbitrary feed-forward and recurrent connectivity, so skips arise freely.
- **Novelty.** Skips over layers of evolved or learned computation are well known in the neural and neuroevolution literature. A6's contribution must be the controlled test in symbolic GP, not the idea.

### 1.8 Deep and layered GP for representation learning (layer-to-layer, mostly without skips)

- **Citation.** L. Rodriguez-Coayahuitl, A. Morales-Reyes and H. J. Escalante (2018). Structurally Layered Representation Learning: Towards Deep Learning through Genetic Programming. *EuroGP 2018*, LNCS 10781, pp. 271–288. doi:10.1007/978-3-319-77553-1_17. <https://gpbib.cs.ucl.ac.uk/gp-html/Rodriguez-Coayahuitl_2018_EuroGP.html> Extended preprint: arXiv:1802.07133.
- **What it does.** It builds a sequence of intermediate representations, each layer's features produced by GP trees over partitioned subsets of the previous layer, with an autoencoder as the case study.
- **Versus A5 and A6.** It is strictly layered, like A5, but its layers are fixed representation stages built in sequence, not an evolving shared library. There is no reuse across calls, no skips and no compute-scaling study.
- **Novelty.** This is prior art for "deep" layered GP. It supports A5's layered framing but does not anticipate the library, reuse or scaling aspects.

- **Citation.** J.-Y. Lin, H.-R. Ke, B.-C. Chien and W.-P. Yang (2007). Designing a classifier by a layered multi-population genetic programming approach. *Pattern Recognition* 40(8):2211–2225. <https://gpbib.cs.ucl.ac.uk/gp-html/Lin20072211.html> The co-author list after Lin was taken from memory and is **unverified**; the gpbib page and title were confirmed.
- **What it does.** In layered GP (LAGEP), populations in each layer produce features that form the next layer's training set.
- **Versus A5 and A6.** It is strictly layer-to-layer, trained layer by layer, with no shared library and no skips.
- **Novelty.** Low threat; cite as related work.

- **Citation.** Y. Bi, B. Xue and M. Zhang (2018). An Automatic Feature Extraction Approach to Image Classification Using Genetic Programming. *EvoApplications 2018*, LNCS 10784, pp. 421–438. doi:10.1007/978-3-319-77538-8_29. <https://gpbib.cs.ucl.ac.uk/gp-html/Bi_2018_evoApplications.html>
- **What it does.** Multi-layer GP (MLGP) uses a program structure with layers for region detection, image descriptors and feature construction, then classification.
- **Versus A5 and A6.** Its "layers" are fixed type-constrained stages within a single tree, not evolving reusable modules. There are no skips and no scaling study.
- **Novelty.** Low threat.

- **Citation.** J. Merta and T. Brandejsky (2022). Two-layer genetic programming. *Neural Network World*. Found via Consensus; **URL not independently verified**; the second author's name is also **unverified**.
- **What it does.** The first layer evolves sub-models for data segments; the second layer evolves a non-linear combination of them.
- **Novelty.** Low threat.

---

## 2. Hierarchical, layered and modular GP

### 2.1 Automatically defined functions (ADFs) and architecture-altering operations

- **Citations.**
  - J. R. Koza (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection*. MIT Press. Confirmed via library catalogues and the Consensus index.
  - J. R. Koza (1994). *Genetic Programming II: Automatic Discovery of Reusable Programs*. MIT Press, ISBN 0-262-11189-6. Confirmed via the BnF catalogue: <https://catalogue.bnf.fr/ark:/12148/cb37486441t>
  - Architecture-altering operations: Koza (1995), the original publication, is **unverified**. Related verified entries are Koza, Andre, Bennett and Keane (1996), <https://gpbib.cs.ucl.ac.uk/gp-html/koza_1996_ADFaacs.html>, and Koza et al.'s patent on automatically defined stores, loops and recursions (ADS/ADL/ADR), <https://gpbib.cs.ucl.ac.uk/gp-html/Koza_2003_patent.html>.
- **What it does.** Each individual has a result-producing branch plus function-defining branches (ADFs) that evolve with it. ADFs can be hierarchical: a later ADF may call an earlier one. Architecture-altering operations (branch duplication, creation and deletion; argument duplication and deletion) let the number of ADFs and their arguments evolve. *GP II* reports that ADFs reduce computational effort on even-parity, which is the G1 target to reproduce.
- **Versus A5.** ADFs are per-individual, not a population-level library, and are copied with the individual. ADF hierarchies have been demonstrated, but the number of levels is fixed or small, and there are no strict level-only-below rules or encapsulation from existing subtrees. Architecture-altering operations resemble A5's "delete unused module" and duplication operators.
- **Versus A6.** The result-producing branch reads raw inputs and all ADFs, so the top of the hierarchy already has skip-like access.
- **Novelty.** ADFs with architecture-altering operations are the canonical precedent for evolving reusable, hierarchical modules. A5's distinctive elements are the explicit library with levels, encapsulation from subtrees, and, above all, the compute-scaling question with reuse and capacity controls.

### 2.2 Module acquisition (GLiB)

- **Citation.** P. J. Angeline and J. B. Pollack (1993). Evolutionary Module Acquisition. In D. Fogel and W. Atmar (eds), *Proceedings of the Second Annual Conference on Evolutionary Programming*, pp. 154–163, La Jolla, CA. <https://gpbib.cs.ucl.ac.uk/gp-html/angeline_1993_ema.html>
- **What it does.** Its "compress" operator turns a subtree into a named module in a library shared by the population; "expand" inverts it. This protects promising partial solutions.
- **Versus A5.** It is very close to A5's "encapsulate subtree as a new module" operator and to a population-level library. There are no explicit levels or level-only-below rules, and modules are frozen once compressed (they do not evolve).
- **Versus A6.** No skip structure.
- **Novelty.** This is the direct precedent for encapsulation into a shared library. A5 must cite it as the origin of the encapsulation operator.

### 2.3 Adaptive representation (ARL)

- **Citations.**
  - J. P. Rosca and D. H. Ballard (1994). Hierarchical Self-Organization in Genetic Programming. *ICML 1994*, pp. 251–258. <https://gpbib.cs.ucl.ac.uk/gp-html/Rosca_1994_hsoGP.html>
  - J. P. Rosca (1995). Towards Automatic Discovery of Building Blocks in Genetic Programming. *AAAI Fall Symposium on GP, Working Notes*. <https://gpbib.cs.ucl.ac.uk/gp-html/rosca_1995_tadbb.html>
  - J. P. Rosca and D. H. Ballard (1996). Evolution-Based Discovery of Hierarchical Behaviors. *AAAI 1996*. <https://mlanthology.org/aaai/1996/rosca1996aaai-evolution>
- **What it does.** It finds useful blocks from the evolution trace, generalises them into new functions added to the function set, and deletes them when they stop being useful. Using an MDL criterion, it builds a hierarchy of discovered subroutines.
- **Versus A5.** It is close to A5's evolving, shared library with creation and deletion, and is explicitly hierarchical. Its subroutines are added to a global function set rather than organised in strict levels, and the reasons for creation are heuristic. There are no reuse or capacity controls and no compute-scaling study.
- **Versus A6.** No explicit skip study.
- **Novelty.** This is a strong precedent for "evolving library plus deleting unused modules plus hierarchy". **Flag: anticipates A5's library dynamics in substance.**

### 2.4 Hierarchical GP with local modules (hGP)

- **Citation.** W. Banzhaf, D. Banscherus and P. Dittrich (1998/2000). Hierarchical Genetic Programming using Local Modules. Technical Report 50/98, University of Dortmund; also *InterJournal Complex Systems* 228 (2000). <https://gpbib.cs.ucl.ac.uk/gp-html/oai_CiteSeerPSU_324880.html>
- **What it does.** Modules are local to a context, not global, and can define local sub-modules, which gives a natural hierarchy.
- **Versus A5.** It is hierarchical, but modules are local, not shared population-wide. It is the opposite of A5's shared library and is conceptually close to A5-copy.
- **Novelty.** A useful precedent for the A5 versus A5-copy distinction; cite it when motivating H2.

### 2.5 DL'GP: a dynamic lattice of hierarchically shared subroutines

- **Citation.** A. Racine, M. Schoenauer and P. Dague (1998). A dynamic Lattice to Evolve Hierarchically Shared Subroutines: DL'GP. *EuroGP'98*, LNCS 1391, pp. 220–232. doi:10.1007/BFb0055941. <https://gpbib.cs.ucl.ac.uk/gp-html/racine_1998_dlehss.html>
- **What it does.** Hierarchical cooperative coevolution: populations on a lattice each evolve a sub-ability, and the results are shared modules "at disposal of higher level subroutines". The authors claim it avoids repeated searches for the same sub-components and lowers solution depth.
- **Versus A5.** Conceptually this is very close: levels of shared modules, each built on lower ones. It differs in using separate coevolving populations with credit assignment rather than a single library evolved with the program. The abstract does not say whether a level reads only the level below or any lower level. There is no compute-matched scaling study.
- **Novelty.** **Flag: closest early anticipation of A5's levelled, shared-module idea.** The full paper should be read in WP2 to settle its connectivity rule, which also matters for A6.

### 2.6 Automatically defined macros (ADMs)

- **Citation.** L. Spector (1995). Evolving Control Structures with Automatically Defined Macros. *AAAI Fall Symposium on GP, Working Notes*, pp. 99–105. <https://gpbib.cs.ucl.ac.uk/gp-html/spector_1995_ADM.html>
- **What it does.** It extends ADFs to macros that control the evaluation of their arguments, producing new control structures.
- **Versus A5 and A6.** Different semantics (control flow), no levels or library.
- **Novelty.** Low threat; cite for completeness.

### 2.7 Kinnear's ADF alternatives

- K. E. Kinnear Jr (1994). Alternatives in automatic function definition: a comparison of performance. In *Advances in Genetic Programming*, MIT Press. **Unverified.** It compares ADFs with module acquisition and finds ADFs more effective on the problems tested, a useful caution for A5's encapsulation operator.

### 2.8 Embedded and modular CGP (ECGP)

- **Citations.**
  - J. A. Walker and J. F. Miller (2008). The Automatic Acquisition, Evolution and Reuse of Modules in Cartesian Genetic Programming. *IEEE Transactions on Evolutionary Computation* 12(4):397–417. doi:10.1109/TEVC.2007.903549. <https://gpbib.cs.ucl.ac.uk/gp-html/Walker_2008_TEC.html>
  - Precursor: Walker and Miller (2004), *EuroGP 2004*, LNCS 3003, pp. 187–197. <https://gpbib.cs.ucl.ac.uk/gp-html/walker_2004_eurogp.html>
- **What it does.** It adds compress (encapsulate a set of active nodes as a module), expand, and module mutation to CGP, with modules held in a library shared across the population. It reports faster solutions than plain CGP, with larger gains on larger problems, on circuits, two symbolic-regression problems, lawnmower and hierarchical if-and-only-if. Evolved modules often compute recognisable functions.
- **Versus A5.** This is a very close analogue of A5's operators (encapsulate, mutate module, library) in a graph setting, and its "recognisable functions" finding anticipates H4 in spirit. It has no strict level structure. The metric is speed to solution, not generalisation as compute grows.
- **Versus A6.** The underlying CGP has levels-back connectivity (§1.1).
- **Novelty.** **Flag: anticipates A5's operator set closely.** A5 must cite it and say clearly that its novelty lies in levels, the scaling question and the controls.

### 2.9 Run Transferable Libraries (RTL)

- **Citations.**
  - M. Keijzer, C. Ryan and M. Cattolico (2004). Run Transferable Libraries: Learning Functional Bias in Problem Domains. *GECCO 2004*, LNCS 3103. Listing: <https://www.cs.york.ac.uk/rts/docs/GECCO_2004/Conference%20proceedings/bibs/3103/31030531.htm>
  - C. Ryan, M. Keijzer and M. Cattolico (2004). Favorable Biasing of Function Sets Using Run Transferable Libraries. *GPTP II*, pp. 103–120. <https://gpbib.cs.ucl.ac.uk/gp-html/ryan_2004_GPTP.html>
  - Keijzer, Ryan, Murphy and Cattolico (2005). Undirected Training of Run Transferable Libraries. *EuroGP 2005*. <https://gpbib.cs.ucl.ac.uk/gp-html/eurogp_KeijzerRMC05.html>
- **What it does.** It keeps a library of functions learned across runs and problem instances. It solves benchmarks considerably faster than GP with ADFs, and libraries tend to converge on a few functions.
- **Versus A5.** Its library persists across runs, whereas A5's library evolves within a run. There are no levels or skips.
- **Novelty.** Low direct threat. Its finding that libraries converge on a few functions is a useful warning for A5's library diversity.

### 2.10 Tag-based modules

- **Citations.**
  - L. Spector, B. Martin, K. Harrington and T. Helmuth (2011). Tag-based modules in genetic programming. *GECCO 2011*, pp. 1419ff. doi:10.1145/2001576.2001767. <https://edoc.mdc-berlin.de/id/eprint/19309/>
  - Spector, Harrington and Helmuth (2012). Tag-based modularity in tree-based genetic programming. *GECCO 2012*. <https://gpbib.cs.ucl.ac.uk/gp-html/Spector_2012_GECCO.html>
- **What it does.** Code is tagged and called by inexact tag matching, without a fixed module architecture. The 2012 tree-GP study found large benefits on lawnmower but **not on even-4-parity**.
- **Versus A5 and A6.** Different addressing (tags, not levels); no levels or skips.
- **Novelty.** Low threat, but the parity null result is relevant to G1's ADF-on-parity check and to A5's expected behaviour on Boolean tasks.

### 2.11 Modular multi-tree GP (arm A4)

- **Citation.** H. Zhang, Q. Chen, B. Xue, W. Banzhaf and M. Zhang (2024). Modular Multi-Tree Genetic Programming for Evolutionary Feature Construction for Regression. *IEEE Transactions on Evolutionary Computation* 28(5):1455–1469. doi:10.1109/TEVC.2023.3318638. <https://gpbib.cs.ucl.ac.uk/gp-html/Hengzhe_Zhang_ieeeTEC.html> Code: <https://github.com/hengzhe-zhang/EvolutionaryForest>
- **What it does.** It is a "modular and hierarchical multitree GP" in which trees can use the outputs of other trees as features. It adds a macro parent-repair strategy, macro crossover and adaptive operator rates. On 98 regression datasets it beat seven bloat-control methods on test performance and model size, and it ranked well on SRBench against 22 methods.
- **Versus A5.** It is hierarchical, with reuse of tree outputs. It has no explicit level structure, no library separate from the individual, and no encapsulation or deletion operators. It reports fixed-budget comparisons, not compute scaling.
- **Versus A6.** The abstract does not say whether a tree may read any earlier tree (skip-like) or only the adjacent one. **This must be checked in the full text before building A4.** If any earlier tree is allowed, A4 is already an A6-like connectivity baseline.
- **Novelty.** It is already in the plan as A4. It overlaps A5 in substance: hierarchical reuse of evolved features, including on SRBench.

### 2.12 Tangled Program Graphs (TPG)

- **Citations.**
  - S. Kelly and M. I. Heywood (2017). Emergent Tangled Graph Representations for Atari Game Playing Agents. *EuroGP 2017*, LNCS 10196, pp. 64–79. doi:10.1007/978-3-319-55696-3_5. <https://gpbib.cs.ucl.ac.uk/gp-html/Kelly_2017_EuroGP.html>
  - S. Kelly and M. I. Heywood (2017). Multi-task Learning in Atari Video Games with Emergent Tangled Program Graphs. *GECCO 2017*, pp. 195–202. <https://gpbib.cs.ucl.ac.uk/gp-html/Kelly_2017_GECCO.html>
  - S. Kelly and M. I. Heywood (2018). Emergent Solutions to High-Dimensional Multitask Reinforcement Learning. *Evolutionary Computation* 26(3):347–380. doi:10.1162/evco_a_00232. Confirmed from search listings; DOI page not opened.
  - S. Kelly (2018). *Scaling Genetic Programming to Challenging Reinforcement Tasks through Emergent Modularity*. PhD thesis, Dalhousie University. <http://hdl.handle.net/10222/73979>
- **What it does.** Teams of linear programs are coevolved; teams can reference other teams as actions, so graphs of reusable teams emerge with depth and width growing over time. The EuroGP paper reports TPG beating deep RL on 15 of 20 Atari games with much sparser computation.
- **Versus A5 and A6.** TPG has emergent hierarchical reuse (a team can be referenced by many teams), and graphs need not be strictly layered. It targets policies, not symbolic regression, and has no explicit levels or skip ablation. Kelly's thesis title explicitly links modularity to "scaling" GP, though in task difficulty rather than compute-versus-generalisation curves.
- **Novelty.** The plan already adopts TPG for Stage 3. It is a precedent for "modularity lets GP scale". It does not anticipate a compute-matched generalisation study on symbolic tasks.

### 2.13 Layered learning GP

- **Citation.** S. M. Gustafson and W. H. Hsu (2001). Layered Learning in Genetic Programming for a Co-operative Robot Soccer Problem. *EuroGP 2001*, LNCS 2038, pp. 291–301. doi:10.1007/3-540-45355-5_23. <https://gpbib.cs.ucl.ac.uk/gp-html/gustafson_2001_EuroGP.html>
- **What it does.** It decomposes a task into hand-designed sub-problems solved in sequence, carrying the earlier population forward. It reaches better fitness faster than standard GP on keep-away soccer.
- **Versus A5 and A6.** Its layers are a human-designed curriculum, not evolved modules. No library, no skips.
- **Novelty.** Low threat.

### 2.14 Library learning outside GP: DreamCoder and LaSR

- **Citations.**
  - K. Ellis et al. (2021). DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning. *PLDI 2021*. doi:10.1145/3453483.3454080 (DOI resolves to ACM DL). Preprint: arXiv:2006.08381.
  - A. Grayeli, A. Sehgal, O. Costilla-Reyes, M. Cranmer and S. Chaudhuri (2024). Symbolic Regression with a Learned Concept Library. *NeurIPS 2024*. arXiv:2409.09359. <https://neurips.cc/virtual/2024/poster/96212>
- **What they do.** DreamCoder alternates program search with refactoring found programs into a growing library of abstractions (built on earlier abstractions), plus a neural recognition model. LaSR adds an LLM-maintained library of natural-language concepts to PySR's genetic search; it reports 66 of 100 Feynman equations recovered against 59 for the best baseline.
- **Versus A5 and A6.** DreamCoder's growing, layered library of reused abstractions is the closest conceptual neighbour to A5 outside GP. It relies on a neural guide and program refactoring, not evolution, and learns across many tasks. LaSR's library is textual and LLM-driven. Neither studies skips.
- **Novelty.** Neither is a direct threat, but A5 should be positioned against DreamCoder-style library learning: "evolutionary library learning within a single task, under a fixed compute budget". LaSR is a candidate reference point if the FunSearch-style fallback in the plan's risk register is used.

### 2.15 Surveys and reviews

- G. Gerules and C. Janikow (2016). A Survey of Modularity in Genetic Programming. *IEEE CEC 2016*, pp. 5034–5043. doi:10.1109/CEC.2016.7748328. <https://gpbib.cs.ucl.ac.uk/gp-html/Gerules_2016_CEC.html>
- L. Rodriguez-Coayahuitl, A. Y. Rodriguez-Gonzalez, D. Fajardo-Delgado and M. G. Sanchez Cervantes (2025). Problem Decomposition Strategies and Credit Distribution Mechanisms in Modular Genetic Programming for Supervised Learning. *IEEE Transactions on Evolutionary Computation* 29(6):2785–2803. doi:10.1109/TEVC.2025.3526581. <https://gpbib.cs.ucl.ac.uk/gp-html/Rodriguez-Coayahuitl_TEVC.html>
  - This reviews 70 papers on modular, hierarchical and coevolutionary tree GP for supervised learning, with a taxonomy (solution architecture, decomposition strategy, credit assignment) and a comparison with deep learning. **It should be read in full before WP2.** It is the most likely place to find any hierarchy-with-skips GP work this check missed, and its taxonomy is a ready-made frame for placing A5 and A6.

---

## 3. GP performance as compute grows

- **Long-term evolution (Langdon).**
  - W. B. Langdon (2017). Long-Term Evolution of Genetic Programming Populations. arXiv:1703.08481. <https://arxiv.org/abs/1703.08481> Runs Mux-6 tree GP for up to 100,000 generations, with trees of more than 10^8 nodes.
  - W. B. Langdon and W. Banzhaf (2019). Continuous Long-Term Evolution of Genetic Programming. *ALIFE 2019*. <https://gpbib.pmacs.upenn.edu/gp-html/Langdon_2019_alife.html>
  - W. B. Langdon and W. Banzhaf (2022). Long-Term Evolution Experiment with Genetic Programming. *Artificial Life* 28(2):173–204. doi:10.1162/artl_a_00360. <https://gpbib.cs.ucl.ac.uk/gp-html/Langdon_2022_ALJ.html> Up to a million generations: innovation continues "but this is limited by their depth". Deep expressions disperse information and impede evolvability. The authors argue for "open complexity" rather than ever-deeper nesting.
  - W. B. Langdon (2022). Measuring failed disruption propagation in genetic programming. *GECCO 2022*, pp. 964–972. <https://discovery.ucl.ac.uk/id/eprint/10154055/> On average up to 83.3% of disruptions from crossover fail to reach the root. Related: Failed Disruption Propagation in Integer GP, arXiv:2204.13997.
  - **Relevance.** This is the strongest existing evidence that deep tree GP plateaus because information from deep changes does not reach the output. It directly motivates H3: skips should keep deep modules "responsive to variation". The H3 mutation-sensitivity measure should be defined to match Langdon's disruption-propagation measure so results are comparable. It also motivates A5 and A6 as forms of "open complexity" (wide, reused structure rather than deep nesting).
- **Scaling laws in symbolic regression.** D. Otte, J. K. H. Franke, A. Zelā, F. Ferreira and F. Hutter (2025). Towards Scaling Laws for Symbolic Regression. arXiv:2510.26064 (NeurIPS 2025 Math-AI and EurIPS 2025 workshops). <https://arxiv.org/abs/2510.26064> It shows power-law scaling of loss and solved rate with compute for transformer-based SR. It is not GP, but it is the nearest precedent for the plan's fitted-error-floor analysis and should be cited for method.
- **Generalisation theory for GP symbolic regression.** M. Nomura, R. Hamano and I. Ono (2026). On the Generalization Bounds of Symbolic Regression with Genetic Programming. arXiv:2604.17402 (accepted at PPSN 2026). <https://arxiv.org/abs/2604.17402> It splits the generalisation gap into a structure-selection term and a constant-fitting term, linking depth limits and parsimony to the hypothesis-class size. It is relevant to arguing why shared modules (smaller effective structure) might generalise better at large compute (H1 and H2).
- **Finding.** No study was found that fits compute-scaling curves of test error for GP, or compares GP representations on compute-matched scaling curves. The H1 framing, plateau-then-continued-improvement under one shared compute counter, appears to be novel.

---

## 4. Conclusions

### (a) Novelty verdict

**A5 (hierarchical GP).** The representation and operators are largely anticipated:
- ADFs with architecture-altering operations: hierarchy, creation and deletion;
- Angeline & Pollack's module acquisition and Walker & Miller's ECGP: encapsulation into a shared library, module mutation;
- Rosca & Ballard's ARL: evolving library with deletion and hierarchy;
- Racine et al.'s DL'GP: levels of shared subroutines;
- Zhang et al. 2024 (A4) and Modular GP-GOMEA 2025: hierarchical reuse of evolved sub-expressions in symbolic regression.

What appears new is:
1. the strict level structure combined with a single evolving shared library and all four operators together;
2. a **compute-matched** test, with one node-evaluation counter, of whether hierarchy keeps improving *generalisation* after tree GP plateaus;
3. the controls that isolate reuse (A5-copy), learned versus random modules (A5-frozen) and capacity (A1-size);
4. structure recovery against known sub-functions (H4), which ECGP and Modular GP-GOMEA touch on only informally.

A5 is publishable as a rigorous scaling study, not as a new representation.

**A6 (with skips).** Skip-like connectivity is **not new** in GP:
- CGP's levels-back, linear GP registers and Koza's result-producing branch all allow it;
- Modular GP-GOMEA explicitly lets any tree read any earlier tree and the raw inputs;
- Fan et al. (2023) reuse internal-node features;
- neural SR (OccamNet) and CGP-based neural architecture search use explicit ResNet/DenseNet-style skips.

No work found isolates the effect of skips on top of a strictly layered, evolving module hierarchy, with a compute-matched comparison and a mechanism measure (mutation sensitivity or disruption propagation at depth). Sotto et al. (2021) come closest on the importance of levels-back, but for convergence, not generalisation as compute grows. A6's novelty therefore lies in the controlled test and the mechanism (H3), not the connectivity idea. This triggers the plan's own risk-register fallback.

### (b) Recommended framing changes

1. **Reframe A6 as "a test of an existing idea at scale".** Present skips as the module-level version of CGP's levels-back and of dense connectivity in Modular GP-GOMEA. Use the ResNet/DenseNet analogy as motivation only. Remove or correct the plan's sentence that searches "found no GP-specific work on skip connections".
2. **Present A5 as a consolidation, not a new representation.** Credit ADFs, module acquisition, ARL, DL'GP and ECGP for the ingredients, and put the claimed contribution on H1 and H2: compute-matched scaling plus reuse and capacity controls.
3. **Tie H3 to Langdon's findings.** Define the mutation-sensitivity measure in line with "failed disruption propagation" (Langdon 2022) and frame A5 and A6 as a test of Langdon and Banzhaf's "open complexity" argument.
4. **Position against library learning.** One paragraph contrasting A5 with DreamCoder (neural-guided, cross-task) and LaSR (LLM concepts): A5 is evolutionary, within a single task and compute-counted.
5. **Check A4's connectivity** in the Zhang et al. full text, and state clearly in the paper how A5 and A6 differ from A4.
6. **Read before WP2:** the Rodriguez-Coayahuitl et al. (2025) review, the full DL'GP paper and the full Modular GP-GOMEA paper. Any missed prior work on hierarchy with skips is most likely cited there.

### (c) Additional baselines to consider (each needs Michael's agreement; fairness implications noted)

1. **A3 with levels-back varied (A3-lb1 versus A3-lbmax).** A near-free skip ablation in an established representation. If skips help A3 the same way they help A5, H3 is about connectivity in general; if not, it is specific to modules. It uses the same counter and budget as other arms.
2. **A6-inputs-only.** Skips to raw inputs only, no cross-level skips. This separates "access to raw features" (as in Koza's result-producing branch and Modular GP-GOMEA's input terminals) from "access to any lower level".
3. **A2 with hierarchical ADFs** (an ADF may call earlier ADFs), as a config option of A2 rather than a new arm. It is the classic version of "levels plus top-level skips" and makes the A2 versus A5 comparison more informative.
4. **Modular GP-GOMEA (optional, external).** The closest recent symbolic-regression method with A6-like connectivity. It uses a different search engine (GOMEA), so it cannot share the selection scheme. It would be an external reference point, not a matched arm. Check code availability first.
5. **ECGP (optional, lower priority).** A module-acquisition graph GP. It could stand in for "explicit modules in CGP" if reviewers ask why A3 has no modules. It adds implementation cost; defer unless G2 results make it necessary.

### Caveats for G1

- Tag-based modularity in tree GP found no benefit on even-4-parity (Spector et al. 2012). Koza's ADF effort reductions on even-parity are the G1 target, but the size of the effect depends on settings, so the reproduction should report Koza's computational-effort measure with confidence intervals.
- Several older works (Kinnear 1994, the original 1995 architecture-altering operations paper, Brameier & Banzhaf 2007) are cited here without online verification and are marked **unverified**.
