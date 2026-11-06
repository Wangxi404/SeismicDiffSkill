# Public native package source audit

Date: 2026-11-07 (Asia/Shanghai). Scope: read-only source/package review of `DiffAxiSEM3D` and `DiffSPECFEM`; no native build, native run, fetch, checkout, producer edit, or numerical acceptance rerun was performed. Python AST parsing and `bash -n` were the only syntax checks. Numerical acceptance is owned by the separate native-adapter audit and producer.

## Disposition

**DiffSPECFEM: source/package review passes.** I found no substantial standalone packaging, public-path, dependency-import, source-license, or generated-binary disclosure blocker in the published source tree. This is not an independent numerical acceptance decision.

**DiffAxiSEM3D: the two concrete packaging issues found were repaired by the producer during review and the repairs were inspected.** Publication must still use the producer's final post-remap reports and source patch. At initial inspection its repository had no staged/tracked files, which is expected for the in-progress release rather than evidence of a final missing package. Do not label the pre-remap validation reports as the final source result.

## Findings and repair evidence

1. **Resolved: Axi runtime GNU time was an undeclared hardcoded dependency.** Initially `adapter/interface.py:22`, `scripts/forward_preservation.py:12`, and refined scripts hardcoded `/usr/bin/time -v`, while README described generic dependencies. BSD/macOS time rejects `-v`. Current `adapter/interface.py:11,23-25` resolves `DIFFAXISEM_TIME`, `gtime`, or `time`; current `scripts/reproduce.sh:10-16` exports the chosen executable and checks for GNU time before building; README:40,52 lists GNU time and its override. Current wrapper/validation scripts contain no `/usr/bin/time` hardcoding. The producer also guarded the `--version` substitution with `|| true`, so unsupported time reaches the clear GNU-time diagnostic under `set -e`; this final source repair was inspected and the current shell/Python syntax checks passed.

2. **Resolved: Axi pin checkout did not reject carried local source edits.** Initially `scripts/reproduce.sh:15-16` checked out the revision but then copied the working tree, so a checkout could retain tracked edits and untracked source. Current `scripts/reproduce.sh:20-22` rejects tracked/untracked dirty upstream and verifies exact HEAD `bc67bb2d76deb2fb208d14dff7b7b25c57d9f35f` before the source copy at line34. It fails rather than resets/deletes user source. The inspected local upstream was clean and at that exact revision.

3. **Required qualification, not an additional numeric verdict: bitwise preservation under MPI.** `scripts/refined_check.py:13-29` uses the native deterministic NR-weighted partition in both original/patched preservation arms; the actual derivative checks use native cost-based partitioning. `scripts/refined_forward_control.py:22-24` explicitly retains the initial strict bitwise failure and attributes it to separate timing-based partitions changing floating summation order. `records/derivative-contract.md:71-78` explicitly documents forward/adjoint repartition and global-tag history remapping. Final README/release language should identify the matched-partition scope of refined bitwise control and avoid claiming universal bitwise equality across independently repartitioned starts. `scripts/forward_preservation.py:29` still strictly requires bitwise comparisons for the small base case; that is a declared acceptance gate, not proof of bitwise portability to arbitrary partitions/hosts. The producer/root were notified.

## Build, dependencies, and source behavior

### DiffAxiSEM3D

- CMake hint names were checked against the pinned public source, not guessed: `EIGEN3_INCLUDE_DIR` is consumed by `upstream/cmake/FindEigen3.cmake:66-99`; `BOOST_DIR` by `upstream/CMakeLists.txt:134-136`; `METIS_DIR` by `CMakeLists.txt:147-159` and `FindMETIS.cmake:91-96,155-161`; `FFTW3_DIR` by `CMakeLists.txt:139-143` and `FindFFTW3.cmake:14-20`; `NETCDF_DIR` by `CMakeLists.txt:183-187` and `FindNetCDF.cmake:28-36`. The public `DIFFAXISEM_*` wrappers map to these correct variables.
- CMake minimum 3.26 and C++17 agree with `upstream/CMakeLists.txt:51,272-275`. Public flags enable double precision, disable the debug executable and optional HDF5; baseline build is copied before derivative instrumentation. Only the stated Boost header floor changes in the baseline source.
- Python runtime imports outside the standard library are NumPy, SciPy (`scipy.io.netcdf_file` for mesh generation), and Torch; all are documented and preflight-imported. No hidden Python runtime package was found. No project install metadata is promised: the documented execution model is a source checkout and repository-local imports.
- Compiler/CMake/Python/time and dependency-prefix overrides are configurable; the historical `/home/xiwang/...` paths in `records/source-lock.json` are host provenance, not hardcoded recipe paths.
- Public recipe intentionally requires a fresh `native-src` directory and does not delete an existing checkout/build. `prepare_source.py` deterministically rewrites the two upstream hook files from the pinned `upstream` source and copies current extension files; it is suitable for repeated source preparation against that source. It does not itself replace the recipe's source-pin gate.
- The dependency lock currently records versions/tags and host details rather than immutable hashes for every dependency; user overrides intentionally select their supplied installations. Its `skill_post_integrity: planned` was an in-progress record at inspection, so finalization is still the producer's responsibility rather than a failed final audit.

### DiffSPECFEM

- README:110 explicitly scopes the recipe to Linux with GCC/Gfortran and OpenMPI for MPI. Build scripts pin `abf3196e3192af3343f2b402c82f34205577e672`, reject modified tracked upstream files, configure `--enable-double-precision`, preserve original executables before patching, then rebuild with one nice10 make worker. README accurately requires fresh unmodified source directories; rerunning after applying the patch deliberately fails the dirty-source guard.
- `requirements.txt` NumPy>=2,<3 and Torch>=2.6,<3 cover every nonstandard runtime Python import. Patching itself uses standard-library Python only. GNU time `-v` is used by the resource scripts, which is consistent with the stated Linux target; listing Git, make, Bash and GNU time explicitly would improve dependency completeness but is not evidence of a new platform promise.
- Compiler names are intentionally the Linux GCC names in the recipe; no historical Python absolute path is executed by the build/adapter. README:125-126 labels the recorded host Python as recorded-host guidance.
- `patch_native.py:8` refuses already-patched input with a nonzero exit. Its docstring says “Idempotently” although behavior is “apply once and refuse a repeated patch”; that wording could be made precise. This does not invalidate the documented fresh-source workflow. The caller pins and guards the upstream; the patch helper should not be treated as a general arbitrary-revision patcher.
- `scripts/mpi_rank_time.sh` is executable in the Git index (100755), as required by its direct mpirun invocation. Repository-relative imports, example source/data files, native extension source and patch are tracked.

## License and attribution review

- Axi `LICENSE.upstream` matches the fetched upstream MIT license text and retains the AxiSEM3D authors' copyright. Root `LICENSE` adds `Copyright (c) 2026 Sid` while retaining the upstream copyright and MIT permission/warranty text. MIT's concrete source-distribution condition is to retain the copyright and permission notices in copies or substantial portions; these notices must remain in the released package. No relicense conflict was found in the source package.
- Spec root `LICENSE` contains the full GPLv3 text; root `NOTICE` identifies the extension, Sid 2026, and the GPL-3.0-or-later choice while retaining upstream author attribution. Native extension files have `SPDX-License-Identifier: GPL-3.0-or-later`; the inserted patch follows the original upstream header rather than replacing it. For this modified source distribution, GPL sections 4-5 require retained copyright/license/warranty notices, a license copy, a prominent modification/date notice, and the modified whole under GPL; the root NOTICE/README identify the extension and 2026 date. Preserve both Sid and upstream notices. Conveying compiled object code would additionally engage section 6 corresponding-source conditions, but this audit found no native executable in the source candidate set and does not invent a binary release requirement.
- Fetched upstream source/third-party notices remain authoritative for fetched dependencies. Neither repository needs to import unrelated private adapters or manuscript material to build. No source notice was removed by this audit.

## Public file selection and privacy

`git ls-files --cached --others --exclude-standard` was inspected as the candidate public set. At inspection:

- Axi: 5004 source/evidence candidate files, 107472124 bytes; no selected full-history `.bin`, native ELF/Mach-O executable, symlink, or private-file-name candidate. `.gitignore` excludes upstream checkout, native-src, build, dependencies, bytecode/cache, archives, `records/**/history*.bin`, `/runs/`, and generated `/case-refined/`.
- Spec: 346 tracked source/evidence files, 28168061 bytes; no tracked full-history `.bin`, native executable, symlink, or private-file-name candidate. Its ignore rules exclude vendor/build scratch, native baseline binaries, run directories, Python caches, history binaries and selected native generated databases. Retained `.npz`, receiver traces, raw kernels and logs are intentional reviewable evidence, not accidental compiled build products.
- Bounded scans of these candidate files found no private-key block, common GitHub token, AWS access-ID, or bearer-token pattern. This is a pattern scan rather than a claim to prove absence of every possible secret. Recorded host usernames/paths are provenance and were not mistaken for authentication credentials.
- No private old adaptation or manuscript content was inspected. Final Axi staging should use an explicit source/evidence whitelist and recheck the actual staged set after numerical reports finalize.

## Checks performed

All 10 public-wrapper Python files per repository parsed successfully with Python `ast.parse`; every public shell script passed `bash -n`. The Axi dependency-hint mapping, upstream clean status/pin, source/license texts, current GNU-time/source-cleanliness repairs, tracked script execute bits, file-selection sizes/magic, artifact exclusions and credential-pattern filenames were inspected. No native computation was run, and this report makes no new numerical pass claim.
