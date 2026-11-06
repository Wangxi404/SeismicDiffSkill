# Seismic differentiation skill

A reusable Codex skill for adapting an existing seismic forward **source** solver to differentiable inversion: data-domain AD supplies a cotangent, and a consistent native discrete adjoint supplies the propagation VJP. It is not tailored to named solvers and does not target a framework that is already fully differentiable.

The canonical package is `skills/differentiate-seismic-solver/`. It contains the instructions, derivative contract, numerical acceptance criteria and reusable numerical checks. Target-specific adapters and their tests remain independent software projects.

Development uses general numerical fixtures and lessons from the separate HybridADFWI comparison. Full packet snapshots are frozen before applications to independent solver repositories. A version/hash identifies the instructions and tools used; application retries and source changes are recorded separately. See `releases/` once the first freeze is complete and `evaluations/` for application records.

This repository's synthetic compiled fixture tests the workflow itself. It is not a substitute for a real HPC solver validation or a performance benchmark. A published numerical claim requires its actual native build, configuration and raw verification record.

Run development checks with `python -m pytest -q`. Skill format is also checked with Codex's `skill-creator/scripts/quick_validate.py`. Freeze/verify tooling lives in `tools/` and never silently overwrites an existing frozen version.
