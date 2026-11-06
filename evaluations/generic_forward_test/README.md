# Native scalar-wave data-objective inversion interface

Runnable independent extension of the supplied C++ fixture. It preserves the original native displacement update byte-for-byte and adds a C++ native adjoint and Python/PyTorch objective interface. All work and build artifacts are in this directory; the source fixture and frozen skill packet were not edited.

Run:

```sh
PYTHON=/path/to/torch-python ./reproduce.sh
```

This uses `/Users/sid/anaconda3/envs/torch27-py39/bin/python`, NumPy 2.0.2, PyTorch 2.7.0, Apple clang 17, CPU only. `build.sh` contains exact compiler arguments. The native baseline is independently built as `libnative.dylib`; `libadjoint.dylib` compiles the unchanged native source plus `adjoint.cpp`. `numerics.py` is an exact copy of the frozen skill's reusable helper, bundled for reproducibility.

## Interface

`NativeWave.objective_gradient(model, objective, coordinate)` returns `(scalar_loss, raw_model_gradient, details)`. Supported coordinates are `c2`, `velocity` (c2=v²), and `log_velocity` (c2=exp(2l)). The objective is a real scalar callable on the float64 CPU receiver-data tensor. For integration with normal PyTorch training, `NativeWave.differentiable_data(model_tensor, coordinate)` provides a custom backward using the same native adjoint. Only first-order gradients are supported.

The production route is one native forward with full float64 displacement history, a fresh differentiable receiver leaf, the actual objective's AD data cotangent, native C++ reverse propagation, then an optional static material-map VJP. It uses no temporal AD, parameter finite differences, checkpointing or replay. `wave_tangent` and model central differences occur only in independent correctness checks.

Trace layout is `[nt,nr]`, observing displacement after each native update. The transpose scatter sums repeated receivers. Source increments are injected after damping as in the supplied source; no extra dt/dx factors are added. Model endpoints are fixed-boundary inactive coordinates with exactly zero sensitivity. Damping, source, geometry, dt/dx, initial states and boundaries remain fixed. Unsupported coordinates are rejected. See `CONTRACT.md` for the complete discrete derivative, signs and units.

The two demonstrated objectives are discrete sum-of-squares and a nonlinear chain with receiver mixing, differentiable trace RMS normalization, tanh, time convolution and decimation, and real-FFT smooth magnitudes. Both process observed and synthetic data identically. The actual chain's AD output drives the native adjoint; it is not replaced with a least-squares residual.

## Observed results

Final run: **43/43 gates passed**. Native versus extension traces and full histories are bitwise equal on the main heterogeneous/damped case and a source-near-boundary case. Duplicate and endpoint receivers are included. The largest analytic propagation/model dot relative discrepancy was 4.32e-14. Both data-VJP checks and 12 model Taylor sweeps (three coordinate families, two objectives, two informative directions) passed; all 15 step sizes in each sweep are retained. Exact tangent-column checks validate every c2 gradient entry. The native custom backward matches the explicit receiver-leaf interface.

Six native-adjoint log-velocity inversion steps reduced the L2 objective from 5.947819465134294 to 0.056981817291574824. Main history is 51,944 bytes; native adjoint live vectors are 1,032 bytes; no replays. Timings and process RSS are tiny-fixture diagnostics, not an HPC benchmark.

The first run had one independent FD-JVP gate failure: at velocity h=1e-4, truncation error 3.0307e-6 exceeded the unchanged 3e-6 tolerance. Errors decreased at the expected second-order rate. The sweep was extended with h=1e-6 and requires the two finest consecutive steps to pass; every old step and failure is retained. No production or tolerance change was made. See `attempts.json`, `results.run1.failed.json`, and `run1.*.log`.

## Artifacts

- `adapter.py`, `adjoint.cpp`, unchanged `native_wave.cpp`: runnable implementation.
- `run_tests.py`, `numerics.py`, `build.sh`, `reproduce.sh`: checks and exact reproduction.
- `results.json`: complete machine-readable final report, raw Taylor steps, dot products, routes, resources and provenance.
- `raw_arrays.npz`: original native/extension traces, histories, target/configuration, AD cotangents, raw c2 kernels, raw model gradients, directions and inversion result.
- `attempts.json`, `results.run1.failed.json`, `raw_arrays.run1.npz`, run stdout/stderr logs: retained attempts and repairs.
- `pre_integrity.json`, `post_integrity.json`: frozen packet verification, both SHA256 b09e0ee57324f715b80178c2a30cb26a2140b1ea3d5545ad8b15dc5680dca49f.

This is an independent source-and-skill application. No intended solution, other agent notes or target implementation was inspected. No hands-on time estimate or causal claim about skill effort is made.

This archived development test predates the real solver applications. Reproduction scripts were made repository-relative after the independent run; original provenance is retained in `original_artifact_manifest.json` and `commands.log`. The skill packet itself was unchanged. Requires NumPy, PyTorch and a C++17 compiler.
