# Application record

Keep a machine-readable record and the raw run artifacts in the adapter repository, outside any frozen skill packet. Do not infer hands-on development time from chat duration, lines of code, or a build timestamp.

Record these fields where applicable:

- skill version, complete packet/manifest hash, pre/post integrity check;
- target source URL/revision, source hashes and any pre-existing local changes;
- exact build/compiler/library/runtime settings and a runnable native baseline;
- supported physics, model/coefficients, active/fixed quantities, units and precision;
- data and adjoint-source layout/conversion, raw kernel and model-gradient coordinates;
- numerical source changes or instrumentation and their rationale;
- complete commands, failures, retries, adapter patches, human interventions and their provenance;
- AD data-objective checks, native-forward discrepancy, transpose tests, complete Taylor sweeps and raw gradients;
- temporal-AD/replay route, stored history and complete resource measurements;
- exact reproduction commands, package dependencies, scope and remaining failures.
- figure/animation inventory and plotting commands, input hashes, actual model/state/gradient coordinates and units, smoothing/display transforms, selected native times, clipping/aggregation policy, README gallery and rendered QA outcome.

Use statuses such as `planned`, `attempted`, `failed`, and `passed` for individual gates. A build or a zero exit code is not a numerical pass. Unsupported parameters should be rejected or clearly excluded, not returned as zero gradients.

For a frozen application, target-specific repairs do not modify the packet. Record each repair and rerun the affected checks. A generic packet correction creates a new version; preserve the failed version and keep versions separate in reported results. An independent agent context and a frozen packet do not by themselves make the experiment blinded or prove that the skill reduces effort; disclose prior familiarity and use an appropriate no-skill baseline before making a causal cost claim.
