# Derivative contract

Use this when deciding what the native adjoint source and kernel mean. Write down the contract for the selected configuration before coding.

## Coordinates and graph boundary

For real discrete variables, coefficients c=M(m), residual R(u,c;q)=0, sampling d=Cu, and J(m)=ell(d,d_obs)+r(m):

```
bar_d = grad_d ell
R_u.T lambda = C.T bar_d
bar_c = -R_c.T lambda
grad_m J = M'(m).T bar_c + grad_m r
```

The signs above use L=ell-lambda.T R. Equivalent residual signs change lambda conventions, not the final gradient. These equations request matrix-free transpose actions, not assembled space-time Jacobians. Full reverse AD also computes VJPs; the distinction is who supplies the propagation transpose and what graph is retained.

```
c = material_map(m)                         # optional static AD graph
with no_temporal_grad():
    d, history = native_forward(detach(c))
d_leaf = detached_receiver_leaf(d)
bar_d = AD_VJP(objective(d_leaf, d_obs), d_leaf, seed=1)
with no_temporal_grad():
    bar_c = native_adjoint_and_sensitivities(history, bar_d)
grad_m = static_VJP(c, m, bar_c)             # or native model gradient
```

This is an interface sketch, not a required API. For binary solvers, exchange files or native bindings may implement it. Do not put material-map construction inside an indiscriminate no-grad block if its VJP is needed later. The data objective's AD graph may depend on trace duration and processing-network size; removing the wavefield tape does not eliminate all time dependence.

## Source conversion

Record output components, receiver ordering, source experiments, sample times, units, masks/interpolation, and file precision. AD returns the derivative in these data coordinates. The native source may require transpose sampling, reverse-time indexing, signs, mass/integration factors, or a distinct injected component. Test this conversion rather than assuming that residual samples or their time reversal are a universal force source.

If using complex fields, declare a real/imaginary coordinate representation or an explicit complex cotangent convention. A casual replacement of transpose by Hermitian transpose can miss real-objective factors or conjugations.

## State and material sensitivity

Include PML/split states, memory variables, staggered substeps, projections and in-place boundary assignments in the discrete operator. Reversing the time loop or using the forward damping rule unchanged is not a derivation of their transposes. For

```
u[n+1] = Phi_n(u[n], c, q[n]); d[n] = C_n u[n]
p[N] = C_N.T bar_d[N]
p[n] = C_n.T bar_d[n] + Phi_u[n].T p[n+1]
bar_c = sum_n Phi_c[n].T p[n+1]
```

with fixed initial state, histories needed for Phi_c must be saved or reconstructed. The adjoint live state and the saved forward history need not contain identical variables. Choose the minimum justified history from the sensitivity equations, not from the number of names in the forward state.

Native sensitivity kernels may be densities under a spatial integral, derivatives with respect to log parameters, or coefficient-weighted quantities. Convert them to derivatives in the declared model coordinates, including quadrature/Jacobian factors, before applying a static or neural model VJP. Keep raw kernels and converted gradients as distinct artifacts.

## Fixed versus active settings

Specify sources, receiver geometry, initial state, boundaries, mesh and damping profiles that remain fixed. In a perturbation check, do not rebuild a model-dependent damping profile if the returned gradient held it fixed. If such a setting is active, include its derivative terms rather than hiding the mismatch in a tolerance or mask.

Record internal arithmetic precision separately from export dtype. Casting single-precision traces or kernels to float64 does not change the solver's precision. Do not change source scaling, material averaging, boundary rules, or precision in only one arm of a claimed same-forward comparison.
