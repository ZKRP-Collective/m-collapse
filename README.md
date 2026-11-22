# M-Collapse

M-Collapse (formerly OWIC) is an experimental discrete-log research prototype
for prime-order subgroups of `j=0` curves `y^2 = x^3 + b`. It was put together
over a couple of evenings: it is an exploratory prototype, not a solver.

## Algorithm

- **Factor base.** `m` points `B_i` with unknown logs `x_i`.
- **Weighted terms.** `c*B_i` with `c` in `{±1, ±2} x mu_3` (12 coefficients
  per anchor), using the GLV endomorphism `phi(x, y) = (beta*x, y)`.
- **Pair index.** All two-term sums stored once per `S_2 x mu_6` orbit
  (`R = 24*C(m,2) + 8*m + 1` classes); the query-oriented variant stores direct
  representatives and pays for the six phases at lookup.
- **Relations.** For public targets `T = u*G + v*Q`, every term `t` is probed
  as `T - t` in the pair index; each hit gives an exact linear equation
  `sum c*x_i = u + v*k (mod n)`, verified on the curve before use.
- **Circuit stopping.** Anchor logs are eliminated online; the first anchor
  dependency with a nonzero `Q` coefficient yields `k` directly, usually
  before full rank.

## Layout

- `mcollapse_sage/` — SageMath implementation: weighted relations, symmetry
  and y-quotient pair indices, higher-arity quotients, Circuit stopping, and
  their tests and experiment runners. `results/` holds the recorded summaries.
- `mcollapse/` — standard-library Python harness for the H1–H3 factor-base
  experiments, with manifests and tests.

## Run

```sh
python3 -m unittest discover -s mcollapse/tests -t .
sage mcollapse_sage/test_circuit_mcollapse.sage
sage mcollapse_sage/run_circuit_mcollapse_experiment.sage
```

Hash domain tags keep the historical `owic-` prefix so fixtures and seeds
stay reproducible. Summaries in `results/` predate the rename.

## Known limitations

- **Generic algorithm.** Only group operations and `mu_6` are used, so the
  generic-group bound `Omega(sqrt(n/6))` applies. A rough cost model
  (online work `~ n/(24m)`, preparation `~ 12m^2`) gives a total of order
  `sqrt(m*n)`, which is worse than rho with negation and GLV.
- **Toy regime.** Fixtures have `n ~ 2^8` and `m = 12`, so `12m` terms cover
  almost the whole group. Recorded ratios compare M-Collapse variants with
  each other, not with rho or BSGS.
- **Uncharged preparation.** `choose_factor_base` enumerates every `k*G`
  (`O(n)`) and this is not in the work ledger. It is infeasible for
  secp256k1.

## Open hypotheses

1. **Baseline gate.** Add rho/BSGS with `mu_6` to the same ledger, charging
   factor-base construction. Predicted: the M-Collapse/rho ratio grows as
   about `sqrt(m)`.
2. **Coverage artifact.** Fix `m` and grow `n`. Predicted: online work grows
   about `n/(24m)`.
3. **Multi-target.** With known anchor logs, each extra target costs about
   `n/(24m)`. Predicted: this loses to Bernstein–Lange precomputation rho.
4. **Summation polynomials.** Replace the pair table with `S_3` root
   finding over `F_p`. Predicted: the decomposition probability is unchanged,
   so there is no asymptotic gain.

## License

Apache License 2.0, see [`LICENSE`](LICENSE).
