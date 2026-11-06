// Copyright (c) 2026 Sid. MIT License; see the repository LICENSE.
// A small compiled development fixture, not a production HPC benchmark.
// c2 is a positive per-node squared velocity; damping is fixed per node.
// src has nt entries, receiver indices have nr entries, trace has nt*nr
// entries in time-major order. Optional history has (nt+1)*nx entries and
// records the displacement at every integer time, starting with zero.
// Both endpoint displacements and the initial two states are fixed at zero.
#include <algorithm>
#include <vector>

extern "C" int wave_forward(int nx, int nt, double dt, double dx,
                            const double* c2, const double* damping,
                            const double* src, int source_index,
                            const int* receiver_indices, int nr,
                            double* trace, double* history) {
    if (nx < 5 || nt < 1 || dx <= 0 || dt <= 0 || source_index < 1 ||
        source_index >= nx-1 || nr < 1) return 1;
    for (int r=0; r<nr; ++r)
        if (receiver_indices[r] < 0 || receiver_indices[r] >= nx) return 2;
    std::vector<double> previous(nx, 0.), current(nx, 0.), next(nx, 0.);
    if (history) std::fill(history, history+nx, 0.);
    const double a=dt*dt/(dx*dx);
    for (int t=0; t<nt; ++t) {
        std::fill(next.begin(), next.end(), 0.);
        for (int i=1; i<nx-1; ++i) {
            const double lap=current[i-1]-2.*current[i]+current[i+1];
            next[i]=damping[i]*(2.*current[i]-previous[i]+a*c2[i]*lap);
        }
        next[source_index] += src[t];
        for (int r=0; r<nr; ++r) trace[t*nr+r]=next[receiver_indices[r]];
        if (history) std::copy(next.begin(), next.end(), history+(t+1)*nx);
        previous.swap(current);
        current.swap(next);
    }
    return 0;
}
