# Render and fidelity review

This is a collaborator-facing report from a small native one-dimensional fixture. No publication-readiness or continuum-accuracy claim is made.

- Opened and visually inspected all seven generated PNG plates at useful size. Model coordinates, signs, units, colourbars, legends and panel titles are legible. Source/receiver overlays remain inside the native full domain.
- Rendered and inspected all seven exported PDFs in a contact sheet. Text extraction found selectable text in every single-page PDF and zero spans outside page bounds. No label collisions were observed in the reviewed images.
- The final canvas audit found zero rendered text objects outside the figure. The initial audit flagged 48 locator tick labels outside axis limits; those objects are suppressed by Matplotlib and were absent from PNGs/PDFs. The initial report is retained as `render_review-initial.json`, and the detector repair is recorded in `attempts.json`.
- The per-plate rendered axes metadata records plot-area bounds. Across each plate, maximum panel width/height discrepancies are below 10⁻¹⁰ points, within the 1.5-point review tolerance.
- The GIF contains 31 frames. Inspected display frames 0, 10, 20 and 30, corresponding to native state indices 0, 50, 100 and 150. All frames use the same displacement scale, all nodes, and fixed source/receiver overlays.
- Original arrays are copied byte-for-byte into `inputs/`; file SHA256 values are recorded. The native-source hash is taken from the supplied numerical provenance. No native source or skill packet was edited.
- Raw-array checks confirm bitwise equality of native/extension data, exact post-update sampling from the history, a zero initial state, zero fixed endpoints, and bitwise equality of L2 residual/cotangent. A direct audit of saved recurrence values gives maximum absolute mismatch 1.1102230246251565×10⁻¹⁶. This is an audit of stored samples, not a new simulation.
- All 15 saved model Taylor steps, two directions, three model coordinates and two objectives remain visible. Coarse central mismatches and roundoff-floor regions are retained. Numerical gates were not rerun or altered.

Automated details: [PDF/GIF audit](qa/audit.json), [final canvas audit](figures/render_review.json), and per-plate `*.alignment.json`. Visual inspection artifacts: [PDF contact sheet](qa/pdf-contact.png) and [GIF samples](qa/gif-samples.png).

The processed objective formula is absent from the allowed saved artifacts. Its actual saved AD cotangent and recorded verification are therefore shown and explicitly labelled, without inventing its processing graph. Target-model wavefields and timing uncertainty are also absent and are not fabricated.
