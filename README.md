# Seismic differentiation skill

A reusable Codex skill for adapting an existing seismic forward **source** solver to differentiable inversion: data-domain AD supplies a cotangent, and a consistent native discrete adjoint supplies the propagation VJP. It is not tailored to named solvers and does not target a framework that is already fully differentiable.

The canonical package is `skills/differentiate-seismic-solver/`. It contains the instructions, derivative contract, numerical acceptance criteria and reusable numerical checks. Target-specific adapters and their tests remain independent software projects.

Development uses general numerical fixtures and lessons from the separate HybridADFWI comparison. Full packet snapshots are frozen before applications to independent solver repositories. A version/hash identifies the instructions and tools used; application retries and source changes are recorded separately. The first frozen packet is `releases/v0.1.0/skill` (SHA256 `b09e0ee57324f715b80178c2a30cb26a2140b1ea3d5545ad8b15dc5680dca49f`). See `evaluations/` for development evidence and subsequent application records.

This repository's synthetic compiled fixture tests the workflow itself. It is not a substitute for a real HPC solver validation or a performance benchmark. A published numerical claim requires its actual native build, configuration and raw verification record.

Install utility-test dependencies with `python -m pip install -r requirements-dev.txt`, then run `python -m pytest tests -q`. The compiled development application additionally needs a C++17 compiler and PyTorch; run `PYTHON=/path/to/torch-python evaluations/generic_forward_test/reproduce.sh`. Skill format is also checked with Codex's `skill-creator/scripts/quick_validate.py`. Freeze/verify tooling lives in `tools/` and never silently overwrites an existing frozen version.

## Install the frozen skill

Copy `releases/v0.1.0/skill` to `~/.codex/skills/differentiate-seismic-solver`, or link that directory. The frozen packet includes its own MIT license. Verify it with `python tools/freeze.py verify releases/v0.1.0 --expected-sha256 b09e0ee57324f715b80178c2a30cb26a2140b1ea3d5545ad8b15dc5680dca49f`.

## Validation status

The frozen generic packet passed 26 utility tests and an independent compiled scalar-wave source adaptation (43 numerical/API gates; original failed check retained). This is a development fixture.

[DiffSPECFEM](https://github.com/Wangxi404/DiffSPECFEM) is now validated and published: native SH/Newmark, four regional log-density/log-Vs coordinates, one/two MPI ranks, with an additional 32x32 mesh tested on two MPI ranks. Original and extended native outputs are bitwise identical; actual native sampling, processed AD sources, raw GLL kernels and multi-epsilon model checks passed independent raw-artifact audit.

[DiffAxiSEM3D](https://github.com/Wangxi404/DiffAxiSEM3D) is validated and published: native Nr=1/m=0 solid, two layer factors scaling lambda/mu together, one/two MPI ranks, plus a 16x16 refinement case. Actual changed forward/adjoint partitions are handled by global-tag history remapping. Full L2/processed objective, model/propagation/sampling dot/Taylor checks and an outer PyTorch parameter map passed independent raw/history audit. Refined strict native-forward preservation uses a separately labelled matched NR-partition control; production derivatives keep COST partitioning.

These are bounded native derivative validations, not all-physics or continuum-convergence claims. Both applications used the exact same frozen packet. Their failed attempts, target-specific reviews/repairs, source pins, parameter coordinates and complete reports are recorded in `evaluations/applications.json` and `evaluations/reviews/`. HybridADFWI is a separate full-AD comparison, not a skill target.

## Comparison implementation

[HybridADFWI](https://github.com/Wangxi404/HybridADFWI-Open) is published separately with correctness tests, matched full-AD/adjoint measurements and raw kernel evidence. It is a comparison for an already differentiable solver, not an application of this frozen skill. The original private development repository and unpublished manuscript are kept separate from the clean public export.
