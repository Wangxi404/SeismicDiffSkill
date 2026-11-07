# Numerical acceptance

Choose tolerances before the corresponding run, using the solver's arithmetic, linear-solver accuracy, conditioning and file precision. Record a change and its reason; retain prior failed data. There is no universal tolerance suitable for every solver.

## Required evidence

1. **Native forward:** reproduce the native example first. Compare the extension's traces against that baseline with identical model, sources, observation layout, boundaries, precision, build and history settings. Bitwise equality is ideal when the numerical execution is unchanged; otherwise retain measured absolute/relative discrepancy and explain the numerical source.
2. **Data VJP:** validate `AD(objective,d)` against data-space directional derivatives for a simple target and at least one nontrivial processing chain. Use the actual AD output, not a hard-coded least-squares residual while advertising arbitrary objectives.
3. **Transpose identities:** test sampling/coordinate conversion and available tangent/adjoint actions. In consistent coordinates, `<F'(m)p,v> = <p,g_m(v)>`. A central model difference can supply a test JVP if no tangent solver exists; it is not a production gradient.
4. **Model Taylor sweep:** evaluate `E2(h)=abs(J(m+h*p)-J(m)-h*dot(g,p))` over multiple step sizes. Seek an observable second-order interval before roundoff/solver/export noise dominates. Also report centered directional derivatives. Normalize or scale p in physical units and retain every step, not just the best one. Zero or nearly orthogonal directions are uninformative; add informative directions and parameter families.
5. **End-to-end route:** rerun objective -> AD data VJP -> native source -> adjoint -> raw kernel -> model gradient. Verify that the production route has no temporal-AD or parameter-FD fallback. Record the number of forward/adjoint/replay calls and state storage.

The helper `scripts/numerics.py` computes comparison, inner-product and Taylor diagnostics. An adapter may import it, or use its CLI for saved arrays. Store units, coordinate transforms and perturbation scales alongside its raw metrics. A helper returning `passed` is not sufficient when the test was posed in the wrong physical coordinates.

## Test design that exposes common mistakes

- Use nonuniform materials and a source/receiver geometry that samples informative paths. Where supported, test parameter families separately as well as together.
- Include the boundaries and auxiliary states advertised by the wrapper. A homogeneous interior-only test cannot validate a free surface or absorbing-boundary transpose.
- Test multiple components, source experiments and duplicate/interpolated receivers when supported; summing their adjoint contributions is not the same as overwriting a grid point.
- Compare raw unprocessed gradients. Illumination division, clipping, smoothing, fitted scaling, or matching only cosine can conceal an incorrect derivative.
- For exported single-precision or formatted text data, use an epsilon range above the output quantization floor. Report that floor instead of claiming a Taylor slope from zero residuals.
- A scalar objective is differentiated at the same model and with exactly the same fixed settings in every perturbed run. Do not let model initialization, auto-updated density, remeshing, output normalization, or a rebuilt PML silently change the function under test.

## Performance and capacity after correctness

Measure a declared complete-gradient scope and native phase baselines. Include source conversion, parameter VJPs, history writes and replay when used; state exclusions for setup, warmup and storage. Match history policy when isolating adapter overhead, and show separate policies when evaluating their trade-off.

Use the target's actual CPU/MPI/GPU execution. On a shared host, respect existing jobs and record resources and contention. Do not impose one particular host, a GPU requirement, or a fixed number of ranks as a property of the generic skill.

Report process RSS separately from device/process GPU memory and allocator allocated/reserved metrics. An OOM attempt has no completed runtime or gradient error. Keep it as a capacity result; any smaller matched fallback is a separately labelled case. Changed model size, source count or host load cannot serve as a controlled checkpoint ablation.
