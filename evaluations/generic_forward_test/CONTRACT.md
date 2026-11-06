# Pre-implementation numerical contract

Skill: differentiate-seismic-solver 0.1.0-rc1, frozen packet SHA256 b09e0ee57324f715b80178c2a30cb26a2140b1ea3d5545ad8b15dc5680dca49f. Pre verification passed. Source fixture SHA256 07bf9bfa81994d2f08febcfaf5a586210ddd1d9f01012e38a5b22eab77afe578; copy remains byte-identical.

Native 1-D scalar displacement update at interior nodes, fixed zero endpoints and initial states:
 u[t+1,i] = D[i]*(2*u[t,i]-u[t-1,i] + a*c2[i]*(u[t,i-1]-2*u[t,i]+u[t,i+1])) + e_s[i]*src[t], a=dt^2/dx^2.
Source is a direct displacement increment after damping, NOT a dt^2 force density. Trace[t,r]=u[t+1,receiver[r]], float64, time-major. Repeated receiver locations accumulate on scatter. Endpoint samples are supported and identically zero for actual propagation. Mesh units are a consistent arbitrary length/time scale; velocity length/time; c2 length^2/time^2. Losses use discrete unweighted sums/means explicitly stated in the Python code; there are no hidden dt, dx or integration weights.

Supported active coordinates: independent per-node positive c2; positive velocity v with c2=v^2; log-velocity l with c2=exp(2*l). All nx nodes are accepted, with endpoint model coordinates physically inactive and correctly zero derivative. Unsupported active settings: damping, source samples/amplitude/location, dt, dx, endpoints, initial states, receiver geometry. No PML or other physics is advertised.

Temporal route: unchanged native wave_forward (compiled C++) records full displacement history (nt+1)*nx float64; no replay. Native C++ reverse recurrence and sensitivity use that history; 3*nx float64 adjoint vectors. For each reverse step, add C^T*bar_d[t] to bar_u[t+1]; accumulate g_c2[i] += a*D[i]*lap(u[t])[i]*bar_u[t+1,i], then apply transpose interior update, discarding cotangents to fixed endpoint/initial states. The transpose uses neighbor coefficients D[i]*c2[i], not D[j]*c2[j]. Damping is held fixed. No gradient scaling, clipping, smoothing or illumination correction.

Data objective uses a fresh PyTorch receiver-data leaf and its actual AD VJP; the temporal state and history never enter PyTorch. Optional static material-map VJP returns the three supported coordinates. A torch.autograd.Function provides normal user-facing backward using the same native adjoint.

Acceptance thresholds declared before execution (float64 solver and exports): unchanged forward/history bitwise equality; sampling/tangent dot tests atol 1e-10 rtol 2e-10; model FD-JVP adjoint checks atol 1e-9 rtol 3e-6; objective/model Taylor checks helper default 1.8--2.2 order, atol 1e-10 rtol 2e-5, at least two consecutive observable second-order intervals and two consecutive central matches. Gradient comparisons atol 1e-10 rtol 2e-10. All raw steps retained. No threshold relaxation without logged justification.

Build/run CPU only, clang++ -O2 -std=c++17 -dynamiclib, float64, serial. No GPU, remote work or HPC performance claims. Native baseline with and without history is compared to disclose history overhead. Timings are small-fixture diagnostics only.
