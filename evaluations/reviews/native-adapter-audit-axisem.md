# Independent audit: DiffAxiSEM3D

Audit status: final numerical acceptance passed for the declared one/two-rank Nr1/m0 solid two-factor application, including actual repartition-history repair. Earlier provisional findings and checkpoints are retained below. No implementation or frozen skill edits.

## Evidence and scope

Read `adapter/DiffAxiSEM.cpp`, `adapter/interface.py`, `scripts/prepare_source.py`, `scripts/make_case.py`, `scripts/validate.py`, generated case inputs, and relevant public upstream Newmark, Domain, SolidPoint, Mass1D and axial boundary code. Applicable frozen skill: SeismicDiffSkill v0.1.0 and its derivative/numerical/application contracts. No private old adapters or Hybrid implementation consulted.

The declared application is solid-only, isotropic, one native MPI rank, Nr=1 (real m=0), native scalar Mass1D, no attenuation or absorbing boundary, free outer surfaces plus axial projection, two regional lambda/mu multipliers at fixed density. This is a restricted application of native AxiSEM3D, not general 3D/MPI physics.

## Derivation and route findings

Native `upstream/src/core/time/NewmarkTimeScheme.hpp:54-56` updates v'=v+dt/2*(a+a'), a'=B(u), u'=u+dt*v'+dt^2/2*a'. The reverse in `adapter/DiffAxiSEM.cpp:149-155` matches that actual sequence. Recording before applySources (`prepare_source.py:30`) samples u_n; injecting data cotangent after reversing its transition is correct for this pre-step observation convention.

For Nr1 scalar Mass1D, D=M^-1 is diagonal per point with identical component values, and P is the diagonal axial mask. Thus D and P commute. `z=invMass(project(b))` and `project(k(z))` implement the restricted-state transpose of D P (-K); extra projection is valid on the zero-initial-state invariant subspace. The native axial mask in upstream AxialBoundary.cpp:33-35 is real/orthogonal for m0. The native stiffness symmetry assumption needs the actual operator dot test to pass, especially near the axis.

The element force multiplier in `prepare_source.py:16-21` scales an element's linear elastic lambda/mu contribution at fixed mass; `factor` lines 128 and native regional actions lines 151-153 give exact derivatives with respect to those factors. No parameter FD or temporal AD exists in the production reverse. Native spatial operator recomputation is not time-step replay; full u_n history is saved. The main interface performs real objective data AD (`interface.py:44-49`) and processed target includes FIR, receiver mixing, tanh, FFT log spectral energy and temporal differences (`65-82`). Optional custom autograd wrapper only bridges native data/adjoint and permits static outer parameter AD.

Sampling uses actual C++ `sampleForward` and `sampleTranspose` (lines 83-94), with additive duplicate contributions and all SPZ components. The C++ operator check (115-124) exercises those methods and actual native stiffness/mass/projection. This data layout is custom two-nearest-node 0.7/0.3 sampling, not upstream station interpolation/output; documentation must distinguish them.

## Pending acceptance issue

P1 — `scripts/validate.py:10` declares native_forward='bitwise' but lines 12-55 never compute or aggregate a baseline forward comparison. A numerical pass of this script would currently not establish native forward preservation. Producer/root were notified: require a recoverable unpatched native baseline and patched factor=1 native station-output comparison under identical config, with nonzero trace norm and measured gate added to final acceptance. If a separate actual baseline gate is packaged and aggregated, this concern closes.

The current script otherwise includes separate lower/upper/mixed non-vacuous Taylor sweeps, actual objective data VJP checks, arbitrary-probe model dot check (lines 39-46), and outer exp model VJP comparison (48-52). These are useful tests, but no native run report was available locally at the initial audit. Numerical correctness is not yet established.

## Confidence limits

Read-only derivation review only; no heavy build/run/SSH performed. Verify final source/build provenance, precision, baseline raw data, model/data checks and all raw gradients, history counts/storage, failures/repairs, frozen packet integrity and exact reproduction commands when producer finishes. General Nr>1, complex representation, MPI, fluid/coupling, attenuation/absorbers and density derivatives are outside this review and this supported derivative.

## Source repair inspected (raw rerun pending)

New `scripts/forward_preservation.py` runs separately compiled original baseline, patched extension disabled, and unity-material full-history extension; compares native station ASCII bytes and retains executable/data hashes. `scripts/validate.py:11,55` now loads and aggregates that report. This is an appropriate structural repair to the missing-gate issue. Actual native station nonzero signals must be checked in the raw report: current aggregate checks `bool(files)` plus nonzero custom receivers, which alone could theoretically accept only a data_time file. The producer was asked to require nonzero non-time native waveform data and matching candidate file sets. No final report yet.

Root additionally requested a two-rank extension after serial checkpoint. MPI correctness is a new acceptance scope: shared probes must be replica-consistent, assembled native operators need unique-point or inverse-multiplicity metric, receiver transpose must exchange/assemble owner contributions into replicas, and global material gradients need consistent weighting. No MPI implementation or result is accepted by this provisional serial review.

## Serial-source checkpoint raw acceptance

The preserved serial source/report at `records/serial-source-checkpoint` and actual `records/validation` raw artifacts now pass independent review. This is a checkpoint with 4 custom receivers; final MPI-ready source adds a center/shared receiver and has separate rank-specific acceptance pending. Original baseline, extension-disabled and unity-factor/history native station files are bitwise identical; native waveform norms are 0.00202286, 0.000715450 and 0.000776045, closing both baseline and non-vacuity issues.

Independent audit metrics: `/private/tmp/native-adapter-audit-axisem-raw.json`. I independently implemented the NumPy objective/VJP for the full FIR -> receiver mix -> tanh plus rFFT log-energy and temporal-difference objective. Saved data cotangents match at relative 5.74e-17 for L2 and 5.89e-16 for processed; binary cotangents, native gradient files and data files match NPZs exactly. All 96 raw model-perturbation output objectives across both targets/three directions/eight plus-minus epsilons were recomputed and match reported objectives within 3.47e-18. The arbitrary-probe model dot error recomputes to relative 2.78e-7. Nonzero lower/upper/mixed gradients have observable order-2 Taylor intervals and consecutive central matches. The outer exp bridge gradient matches the native multiplier gradient times m exactly. A failed coarse data-VJP grid is retained before justified smaller-epsilon extension.

This establishes serial-checkpoint numerical correctness without a duplicate native run. Full checkpoint u-history was not synchronized locally (all perturbation histories total hundreds of MB); source/history metadata were reviewed, and only the final principal histories were requested for independent resampling.

## Final MPI-ready source audit (runtime pending)

The new `adapter/DiffAxiSEM.cpp` uses native global mesh tags (upstream GLLPoint constructs SolidPoint with mGlobalTag), gathers owners/replica counts, and forms inverse-multiplicity metric at lines 118-130. Probes at 155 are shared-consistent by tag. Native stiffness (`93-97`) and new sampling-source (`108-111`) actions use the original gather/send/receive/wait/scatter assembly; no existing replicated state is reassembled. Owner-only C contributions sum globally (`99-106`), with global-tag nearest-node selection and duplicates, including an explicit center receiver to exercise MPI-shared sampling.

`dot` (90-91) is the global unique-node metric. Therefore the regional gradients in line 193 correctly contract the globally assembled regional K_j output without duplicate interface weighting; this differs appropriately from SPECFEM's unique-element contractions. Global gradients are reduced inside `dot`, retained identically on each rank, and written only by root, avoiding an extra reduction. Boundary/mass ordering and Newmark transpose remain consistent with the native restricted-state operator. One/two rank limits are enforced in C++ (54-55) and Python (`interface.py:15`). Final validation requires nonzero shared-point and shared-sample counts at two ranks and a separate matched baseline for each rank count.

No source-equation blocker found; actual final rank1/rank2 raw numerical gates, full principal histories and final record/hash closures are still required for release of the expanded scope.

## Genuine repartition-history defect found by refined application

The producer's larger 16x16 case exposed a true defect after baseline/Taylor work: native cost-balanced MPI initialization can change local point count/order between the separately launched forward and adjoint. The original rank-local history read assumed identical local layouts. Different sizes caused a truncation failure; equal sizes with different ordering could have silently returned a wrong gradient. Earlier serial source acceptance remains a checkpoint, not acceptance of this unsafe general MPI history route. Failed refined artifacts must be retained.

The new source repair was read at `adapter/DiffAxiSEM.cpp:139-170,191-242`. It persists exact forward global sampling rows, forward history global tags per rank, and separate forward/adjoint layouts; reverse reads all forward-rank records and maps current native points by immutable global GLL tag before stiffness sensitivity actions. This is a mathematically appropriate adaptation to native repartitioning, without temporal replay or an AD wave solver. It reports doubled aggregate history read traffic for two adjoint ranks. Wrapper dot-check copies now include sampling/tag metadata.

Source repair is credible; final release still requires actual affected numerical reruns and a successful demonstrably changed forward/adjoint layout case. Producer was asked to identify such a run explicitly.

## Final release gate and independently checked evidence

**Numerical release gate: YES** for the declared real Nr1/m0 solid, fixed-density, two regional lambda/mu multipliers, one/two MPI rank application. There is no remaining numerical artifact blocker. Source/package post-integrity is reported complete with the original frozen packet SHA; final public package hygiene is a separate root review. No native solver rerun/build was performed by this reviewer.

The final matching-code `validation-ranks1` and `validation-ranks2` reports both pass ordinary and processed data VJP checks, native propagation/sampling checks, lower/upper/mixed model Taylor sweeps, arbitrary-probe model dot checks and the outer exp model bridge. Their routes explicitly use forward global GLL tags remapped to the current native partition. Pre-remap reports are preserved separately; they are not substituted for current acceptance. Refined 16x16 processed validation also passes separate family Taylor and data VJP/model dot gates after the actual history-identity repair.

Independent final metrics are saved in `/private/tmp/native-adapter-audit-axisem-final-raw.json` and `/private/tmp/native-adapter-audit-axisem-history.json`; the scratch independent NumPy objective/VJP checker is `/private/tmp/audit_axisem_final.py`. The final receiver data shape is [776,5,3]. Independent FIR/mixing/tanh/rFFT-log-energy/temporal-difference cotangents agree with the saved actual AD cotangents at relative approximately 6.1e-16 (base rank1), 6.0e-16 (base rank2) and 5.33e-16 (refined). Recomputing all synthetic data-space Taylor objectives matches stored values within 3.5e-18. Actual native model perturbation pairs match saved JVP arrays for the base case; arbitrary-probe model-dot discrepancies recompute to ~6.13e-8 for base and 1.61e-8 refined. Final native gradient text files equal the NPZ arrays exactly. Final native binary cotangent files were not synchronized locally at audit closure; production file writing/reading and actual data-VJP route were inspected, and prior serial-checkpoint binary cotangents were checked exactly. This absence is a minor artifact-completeness limit, not a numerical blocker.

Original native and extension native station ASCII bytes independently match for both base rank counts with nonzero waveform norms. For the refined case, the original timing-cost partition comparison failed only strict byte equality, with relative waveform differences ~1e-15; the failure/metrics are retained. A separately labelled matched native NR-weighted partition control passes exact station-byte comparison with nonzero signals. Derivative runs retain native COST partitioning. This is a legitimate controlled forward-preservation comparison and a clearly qualified separate partition policy; it is not a relaxation of the failed bitwise tolerance or proof of bitwise reproducibility under independent timing-based partitioning.

Full principal base histories were checked locally: 5,382,336 bytes for rank1; 2 x 2,849,472 bytes for rank2. Reapplying their actual owner sampling gives EXACT saved data equality. All 17 base shared MPI points have identical stored u for every sample.

One bounded read-only SSH NumPy memmap audit inspected the existing refined processed histories, avoiding their bulk download and performing no native propagation/replay. Forward dofs are [6459,6555] and adjoint dofs [6303,6699], with both local tag orders changed. The forward/adjoint global tag sets contain the same 4225 unique physical points, and persisted C global rows are exactly identical. Sampling the actual old-partition full histories gives EXACT saved data equality; all 113 shared old-partition points have identical u histories. The two history files have 40,097,472 / 40,693,440 bytes, and streamed hashes ebf5c5db...0ff1195 / 826a3001...9b409b3033 are recorded in the independent history JSON. This demonstrates the remap under a genuinely changed native partition, closing the earlier defect rather than merely checking identity mapping.

Native stiffness/mass/Newmark/MPI actions remain in C++, no parameter FD or temporal AD supplies production gradients, and reverse uses three native spatial stiffness actions per step per rank. Reads of all forward rank files on every adjoint rank are counted separately from physical storage; this is history reindexing, not time-step replay. The example and refined supplement remain discrete native derivative tests. Hardcoded Exodus minimum-period metadata does not certify frequency resolution or continuum accuracy; the package appropriately makes no convergence, general 3D/Fourier, density-gradient, arbitrary-boundary, GPU or universal skill-effort claim.

## Root artifact closure after final sync

All five principal native cotangent binaries were subsequently synchronized. Root compared each with its saved actual AD array using exact equality; all passed. See `axisem-final-cotangent-closure.json`. This closes the minor local completeness note above without a native or AD rerun.
