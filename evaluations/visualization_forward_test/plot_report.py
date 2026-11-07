#!/usr/bin/env python3
"""Plot only saved native outputs. No propagation, AD, smoothing or interpolation."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42,
                     'svg.fonttype': 'none', 'savefig.dpi': 180})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(x):
    x = np.asarray(x)
    return {'shape': list(x.shape), 'min': float(x.min()), 'max': float(x.max()),
            'l2': float(np.linalg.norm(x)), 'finite': bool(np.isfinite(x).all())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arrays', type=Path, default=Path('inputs/raw_arrays.npz'))
    ap.add_argument('--results', type=Path, default=Path('inputs/results.json'))
    ap.add_argument('--contract', type=Path, default=Path('inputs/CONTRACT.md'))
    ap.add_argument('--out', type=Path, default=Path('figures'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    a = dict(np.load(args.arrays, allow_pickle=False))
    r = json.loads(args.results.read_text())
    dt, dx = .30, 1.0
    nt, nx = a['main_history'].shape[0] - 1, a['model_velocity'].size
    x = np.arange(nx) * dx
    times = np.arange(nt + 1) * dt
    trace_times = times[1:]
    rec = a['receivers']
    residual = a['main_native_data'] - a['observed']
    source_nodes = np.flatnonzero(a['main_history'][1])
    assert source_nodes.size == 1
    source = int(source_nodes[0])
    assert a['main_history'][1, source] == a['source'][0]
    assert np.array_equal(a['main_history'][0], np.zeros(nx))
    assert np.array_equal(a['main_native_data'], a['main_history'][1:, rec])
    assert np.array_equal(residual, a['data_vjp_l2'])
    assert np.array_equal(a['main_native_data'], a['main_extended_data'])
    assert np.allclose(a['model_c2'], a['model_velocity'] ** 2, rtol=0, atol=1e-15)
    assert np.all(a['main_history'][:, [0, -1]] == 0)

    # Audit the saved recurrence; this does not integrate a new wavefield.
    u = a['main_history']
    prev = np.vstack([np.zeros((1, nx)), u[:-2]])
    audit_rhs = a['damping'][None, 1:-1] * (
        2 * u[:-1, 1:-1] - prev[:, 1:-1] +
        dt ** 2 / dx ** 2 * a['model_c2'][None, 1:-1] *
        (u[:-1, :-2] - 2 * u[:-1, 1:-1] + u[:-1, 2:]))
    audit_rhs[:, source - 1] += a['source']
    audit_max = float(np.max(np.abs(audit_rhs - u[1:, 1:-1])))
    manifest = {'schema': 1, 'test_scope': 'forward application of frozen visual skill',
        'candidate_packet_sha256': '51b19130b19625fb074f76db5165e6517d662a61079723e0381d62bfb96f947a',
        'input_files': {p.name: {'sha256': digest(p)} for p in [args.arrays, args.results, args.contract]},
        'historical_numerics_packet': r['application']['packet_sha256'],
        'source': r['source'], 'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                                           'matplotlib': matplotlib.__version__},
        'configuration': {'physics': r['supported']['physics'], 'nx': nx, 'nt': nt,
            'dt': dt, 'dx': dx, 'coordinates': 'x_i=i*dx; full node range 0..42',
            'units': 'nondimensional simulation units; L and T are arbitrary consistent length/time scales',
            'component': 'scalar displacement u', 'rank': 'serial CPU, no MPI replicas',
            'history_time': 'u[n] at n*dt; includes zero initial state',
            'trace_time': 'd[t,r]=u[t+1, receiver[r]] at (t+1)*dt',
            'source_index': source, 'source_index_basis': 'sole nonzero native state at first update equals source[0]',
            'source_convention': 'direct displacement increment after damping, no dt^2 factor',
            'receiver_indices': rec.tolist(), 'repeated_receiver_channels_retained': True,
            'parameter_vector': 'model_c2, model_velocity or log(model_velocity), as indicated',
            'fixed_parameters': r['supported']['fixed'],
            'initial_model_generation': 'not specified in supplied artifacts; no smoothing claimed'},
        'source_fidelity_checks': {'post_update_trace_bitwise_match': True,
            'native_extended_trace_bitwise_match': True, 'l2_cotangent_residual_bitwise_match': True,
            'initial_state_zero': True, 'fixed_endpoints_zero': True,
            'saved_recurrence_absolute_linf': audit_max,
            'l2_objective_from_raw': float(.5 * np.sum(residual ** 2)),
            'recorded_l2_objective': r['tests']['data_vjp_l2']['objective']},
        'transforms': ['x and times constructed using supplied dt/dx', 'target-start velocity difference',
            'prediction-observed residual', 'saved processed cotangent minus residual',
            'native-minus-extension forward difference', 'native-minus-independent-tangent-column c2 gradient',
            'Taylor/dot diagnostics extracted unchanged from results.json'],
        'scale_policy': 'full array ranges; signed maps centred at zero; shared scales for comparable fields; no clipping',
        'forbidden_transforms_not_used': ['smoothing', 'gradient normalization', 'spatial interpolation',
                                           'wavefield fabrication', 'native simulation', 'new AD'],
        'limitations': ['One-dimensional scalar fixture only; no 2D/3D or continuum-accuracy claim.',
            'Only starting-model native displacement history was saved; no target-model wavefield is invented.',
            'Processed objective formula/processing operator is absent from allowed artifacts; saved AD cotangent and its recorded verification are shown without reconstructing that objective.',
            'Original timing samples/uncertainty are unavailable; no controlled speedup chart is made.',
            'Numerical checks are historical saved evidence, not re-executed by this plotting-only application.'],
        'figures': [], 'animations': [], 'failures': []}
    review = []

    def save(fig, name, title, caption, keys, extra=None):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes = [ax for ax in fig.axes if ax.get_label() != '<colorbar>']
        # Record rendered coordinates and any text escaping the canvas.
        outside = []
        suppressed_tick_labels = set()
        for ax in fig.axes:
            for axis in (ax.xaxis, ax.yaxis):
                low, high = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if not low <= tick.get_loc() <= high:
                        suppressed_tick_labels.update([id(tick.label1), id(tick.label2)])
        for artist in fig.findobj(matplotlib.text.Text):
            if id(artist) not in suppressed_tick_labels and artist.get_visible() and artist.get_text().strip():
                bb = artist.get_window_extent(renderer)
                if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > fig.bbox.width + 1 or bb.y1 > fig.bbox.height + 1:
                    outside.append(artist.get_text())
        geometry = {'axes_bounds_fraction': [list(ax.get_position().bounds) for ax in axes],
                    'figure_inches': list(fig.get_size_inches()), 'text_outside_canvas': outside}
        boxes = geometry['axes_bounds_fraction']
        geometry['max_plot_width_difference_pt'] = (max(b[2] for b in boxes) - min(b[2] for b in boxes)) * fig.get_size_inches()[0] * 72
        geometry['max_plot_height_difference_pt'] = (max(b[3] for b in boxes) - min(b[3] for b in boxes)) * fig.get_size_inches()[1] * 72
        geometry['alignment_tolerance_pt'] = 1.5
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(args.out / f'{name}.{ext}')
        (args.out / f'{name}.alignment.json').write_text(json.dumps(geometry, indent=2) + '\n')
        record = {'name': name, 'title': title, 'caption': caption, 'array_keys': keys,
                  'outputs': [f'{name}.{ext}' for ext in ('png', 'pdf', 'svg')],
                  'statistics': {k: stats(a[k]) for k in keys if k in a}, 'geometry': geometry,
                  'clipped_count': 0, 'clipped_fraction': 0.0}
        if extra:
            record.update(extra)
        manifest['figures'].append(record)
        review.append({'figure': name, 'automated_text_canvas_check': 'pass' if not outside else 'review',
                       'text_outside_canvas': outside})
        plt.close(fig)

    def decorate(ax, xlabel='Position x (simulation L)', ylabel=None):
        ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        ax.grid(alpha=.16)

    # Actual target and native starting model in the same coordinate system.
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.1), layout='constrained')
    ax[0].plot(x, a['true_velocity'], label='Saved target', lw=2, color='#c44e52')
    ax[0].plot(x, a['model_velocity'], label='Native starting model', lw=1.8, color='#2878a4')
    ax[0].set_title('a  Actual velocity models', loc='left')
    ax[0].legend(frameon=False)
    decorate(ax[0], ylabel='Velocity v (simulation L/T)')
    delta = a['true_velocity'] - a['model_velocity']
    bound = float(np.max(np.abs(delta)))
    ax[1].plot(x, delta, color='#663399', lw=1.8)
    ax[1].axhline(0, color='.4', lw=.7)
    ax[1].set_ylim(-bound * 1.12, bound * 1.12)
    ax[1].set_title('b  Target minus starting model', loc='left')
    decorate(ax[1], ylabel='Velocity difference (simulation L/T)')
    fig.suptitle('1D native case | full 43-node domain', fontsize=14)
    save(fig, '01_models', 'Actual model and starting discrepancy',
         'Native starting and saved target velocities; no smoothing is inferred. Differences retain their physical coordinate and sign.',
         ['true_velocity', 'model_velocity', 'model_c2'], {'difference_statistics': stats(delta)})

    # 1D spatial curves, followed by a space-time image labelled as such.
    snapshot_indices = [20, 50, 90, 150]
    wave_bound = float(np.max(np.abs(u)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.0), layout='constrained')
    for n, ax in zip(snapshot_indices, axes.flat):
        ax.plot(x, u[n], color='#2878a4', lw=1.6)
        ax.axhline(0, color='.55', lw=.6)
        ax.axvline(x[source], color='#c44e52', ls='--', lw=.9, label='Source')
        ax.plot(x[np.unique(rec)], np.full(np.unique(rec).size, -wave_bound * 1.04), '|',
                ms=9, color='.15', label='Receiver nodes')
        ax.set_ylim(-wave_bound * 1.13, wave_bound * 1.13)
        ax.set_title(f'Native u[{n}] | t = {times[n]:g} simulation T', loc='left')
        decorate(ax, ylabel='Displacement u (simulation units)')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='upper right')
    ax = axes[1, 1]
    ax.plot(trace_times, a['source'], color='#c44e52')
    ax.axhline(0, color='.55', lw=.6)
    ax.set_title('Applied source increments at node 17', loc='left')
    decorate(ax, 'Post-update time (simulation T)', 'Displacement increment (simulation units)')
    ax = axes[1, 2]
    image = ax.imshow(u, origin='lower', aspect='auto', interpolation='nearest',
                      extent=[x[0] - dx/2, x[-1] + dx/2, -dt/2, times[-1] + dt/2],
                      cmap='RdBu_r', vmin=-wave_bound, vmax=wave_bound)
    ax.set_title('Saved 1D history: position × time', loc='left')
    ax.set_xlabel('Position x (simulation L)'); ax.set_ylabel('State time (simulation T)')
    fig.colorbar(image, ax=ax, label='Displacement u (simulation units)', shrink=.85)
    fig.suptitle('Native scalar displacement | no spatial interpolation or new simulation', fontsize=14)
    save(fig, '02_native_wavefield', 'Native displacement and source',
         'All spatial nodes are retained. Four native indices share one full-history amplitude scale. The history includes u[0]=0, with post-update traces at u[t+1]. Receiver nodes are shown as rug marks; channels 3 and 4 both sample node 12.',
         ['main_history', 'source', 'receivers'],
         {'native_indices': snapshot_indices, 'native_times': times[snapshot_indices].tolist(),
          'wavefield_display_limits': [-wave_bound, wave_bound], 'interpolation': 'nearest cell display only, no new samples'})

    # Receiver CHANNEL order preserves duplicate and endpoint receivers.
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout='constrained')
    obs_bound = max(float(np.abs(a['observed']).max()), float(np.abs(a['main_native_data']).max()))
    res_bound = float(np.abs(residual).max())
    processed_diff = a['data_vjp_processed'] - residual
    panels = [('Observed (saved target data)', a['observed'], obs_bound, 'Displacement'),
              ('Predicted (native starting model)', a['main_native_data'], obs_bound, 'Displacement'),
              ('Residual = predicted − observed', residual, res_bound, 'Displacement'),
              ('Saved AD cotangent: L2', a['data_vjp_l2'], res_bound, 'Objective / displacement'),
              ('Saved AD cotangent: processed', a['data_vjp_processed'], float(np.abs(a['data_vjp_processed']).max()), 'Objective / displacement'),
              ('Processed cotangent − residual', processed_diff, float(np.abs(processed_diff).max()), 'Numeric difference (simulation units)')]
    scales = []
    for i, (ax, (label, data, b, unit)) in enumerate(zip(axes.flat, panels)):
        im = ax.imshow(data, aspect='auto', interpolation='nearest', origin='upper',
                       extent=[-.5, len(rec)-.5, trace_times[-1]+dt/2, trace_times[0]-dt/2],
                       cmap='RdBu_r', vmin=-b, vmax=b)
        ax.set_title(f'{chr(97+i)}  {label}', loc='left', fontsize=10)
        ax.set_xticks(range(len(rec)), [str(v) for v in rec])
        ax.set_xlabel('Receiver channel order (tick = node index)')
        ax.set_ylabel('Post-update time (simulation T)')
        fig.colorbar(im, ax=ax, label=f'{unit} (simulation units)', shrink=.86)
        scales.append({'panel': label, 'limits': [-b, b], 'statistics': stats(data)})
    fig.suptitle('Data differentiation boundary | 150 times × 9 actual receiver channels', fontsize=14)
    save(fig, '03_data_boundary', 'Data, residual and actual AD cotangents',
         'Observed and predicted gathers share a scale. Residual and L2 cotangent share another and are bitwise equal for the saved half-sum-squared objective. The processed cotangent is the actual saved AD VJP and differs from the residual. No processed objective formula is inferred.',
         ['observed', 'main_native_data', 'data_vjp_l2', 'data_vjp_processed', 'receivers'], {'panel_scales': scales})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.1), layout='constrained')
    units = [('c2', 'c²', 'Objective / (simulation L/T)²'),
             ('velocity', 'v', 'Objective / (simulation L/T)'),
             ('log_velocity', 'log(v / 1 simulation L/T)', 'Objective (dimensionless coordinate)')]
    for ax, (coord, label, unit) in zip(axes.flat, units):
        for loss, color in [('l2', '#2878a4'), ('processed', '#c44e52')]:
            ax.plot(x, a[f'gradient_{coord}_{loss}'], label=loss, color=color, lw=1.7)
        if coord == 'c2':
            ax.plot(x, a['tangent_column_gradient_c2'], ':', color='.1', lw=1.2, label='Independent tangent columns, processed')
        bound = max(float(np.abs(a[f'gradient_{coord}_{loss}']).max()) for loss in ('l2', 'processed'))
        ax.set_ylim(-bound * 1.12, bound * 1.12)
        ax.axhline(0, color='.5', lw=.6)
        ax.set_title(f'Unprocessed derivative with respect to {label}', loc='left', fontsize=10)
        decorate(ax, ylabel=unit); ax.legend(frameon=False, fontsize=8)
    diff = a['gradient_c2_processed'] - a['tangent_column_gradient_c2']
    ax = axes[1, 1]
    ax.plot(x, diff, color='#663399', lw=1.5)
    ax.axhline(0, color='.5', lw=.6)
    b = max(float(np.max(np.abs(diff))), 1e-16)
    ax.set_ylim(-b * 1.2, b * 1.2)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_title('Raw processed c² gradient difference, own scale', loc='left', fontsize=10)
    decorate(ax, ylabel='Derivative difference (objective / (L/T)²)')
    fig.suptitle('Model-coordinate derivatives | full native node vectors, no normalization', fontsize=14)
    save(fig, '04_gradients', 'Actual model gradients and independent difference',
         'These are discrete per-node derivatives, not continuous kernel densities. Within each coordinate both objectives share a signed scale. Fixed endpoint derivatives are zero. The raw native-minus-independent comparison retains its own small scale.',
         [f'gradient_{c}_{loss}' for c, _, _ in units for loss in ('l2', 'processed')] + ['tangent_column_gradient_c2'],
         {'difference_statistics': stats(diff), 'gradient_scaling': 'none', 'smoothing': 'none'})

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout='constrained')
    for ax, loss in zip(axes[0], ['l2', 'processed']):
        test = r['tests'][f'data_vjp_{loss}']
        rows = test['steps']; h = [z['step'] for z in rows]
        ax.loglog(h, [z['first_order_residual'] for z in rows], 'o-', ms=3, label='Taylor remainder')
        ax.loglog(h, [z['roundoff_floor'] for z in rows], ':', label='Recorded roundoff floor')
        ax.set_title(f'Data AD VJP: {loss} | all {len(rows)} steps', loc='left')
        decorate(ax, 'Data perturbation step h', 'Absolute Taylor remainder (objective units)')
        ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    fdiff = a['main_extended_data'] - a['main_native_data']
    ax.plot(trace_times, np.max(np.abs(fdiff), axis=1), color='#2878a4')
    ax.set_ylim(-1e-16, 1e-16)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.text(.5, .75, 'All 1,350 differences are exactly zero', ha='center', transform=ax.transAxes)
    ax.set_title('Native versus extension forward discrepancy', loc='left')
    decorate(ax, 'Post-update time (simulation T)', 'max channel |extension − native|')
    ax = axes[1, 1]; ax.axis('off')
    tests = ['sampling_main', 'native_tangent_adjoint_main', 'sampling_boundary_small',
             'native_tangent_adjoint_boundary_small', 'coordinate_tangent_adjoint_c2',
             'coordinate_tangent_adjoint_velocity', 'coordinate_tangent_adjoint_log_velocity']
    table_rows = []
    for key in tests:
        q = r['tests'][key]
        table_rows.append([key.replace('native_', '').replace('coordinate_', ''), f"{q['absolute_error']:.2e}", 'pass' if q['passed'] else 'FAIL'])
    table = ax.table(cellText=table_rows, colLabels=['Saved dot test', 'Absolute error', 'Gate'],
                     colWidths=[.63, .23, .14], cellLoc='left', bbox=[0, .10, 1, .85])
    table.auto_set_font_size(False); table.set_fontsize(8)
    ax.set_title('Recorded sampling and model adjoint checks', loc='left')
    fig.suptitle('Quantitative evidence | saved tests, no rerun or threshold changes', fontsize=14)
    save(fig, '05_verification', 'Data VJP, forward and dot verification',
         'All data Taylor steps, including floor behavior, are retained. Forward discrepancy is shown on a dedicated zero-resolving scale. Saved dot tests use the predeclared tolerances in results.json. The plotting application independently checks trace/history sampling and raw-array equality.',
         ['main_native_data', 'main_extended_data'], {'result_tests': ['data_vjp_l2', 'data_vjp_processed'] + tests,
         'forward_difference_statistics': stats(fdiff)})

    # Full model sweeps: both directions, every step, both diagnostic types.
    for loss, number in [('l2', '06'), ('processed', '07')]:
        fig, axes = plt.subplots(3, 2, figsize=(12, 11.5), layout='constrained')
        names = []
        for row, (coord, label, _) in enumerate(units):
            for direction, color, mark in [('aligned', '#2878a4', 'o'), ('mixed', '#c44e52', 's')]:
                key = f'model_taylor_{coord}_{loss}_{direction}'; names.append(key)
                test = r['tests'][key]; rows = test['steps']; h = np.array([z['step'] for z in rows])
                first = np.array([z['first_order_residual'] for z in rows])
                central = np.array([z['central_directional_error'] for z in rows])
                if np.any(first <= 0) or np.any(central <= 0):
                    raise ValueError(f'{key}: zero/nonpositive log diagnostic; handle explicitly')
                axes[row, 0].loglog(h, first, mark+'-', color=color, ms=3, label=direction)
                axes[row, 0].loglog(h, [z['roundoff_floor'] for z in rows], ':', color=color, alpha=.65,
                                    label=f'{direction} recorded floor')
                axes[row, 1].loglog(h, central, mark+'-', color=color, ms=3, label=direction)
                axes[row, 1].loglog(h, [z['central_roundoff_floor'] for z in rows], ':', color=color, alpha=.65,
                                    label=f'{direction} recorded floor')
                tolerance = r['thresholds']['taylor_atol'] + r['thresholds']['taylor_rtol'] * abs(test['directional_derivative'])
                axes[row, 1].axhline(tolerance, color=color, ls='--', alpha=.6, lw=.8,
                                     label=f'{direction} central tolerance')
            for col in range(2):
                axes[row, col].set_title(f'{label}: ' + ('Taylor remainder' if col == 0 else 'Central directional derivative error'),
                                         loc='left', fontsize=10)
                decorate(axes[row, col], 'Step h in declared model coordinate',
                         'Absolute remainder (objective units)' if col == 0 else 'Absolute directional derivative error')
                axes[row, col].legend(frameon=False, fontsize=7, ncol=2)
        fig.suptitle(f'Full model Taylor sweeps: {loss} objective | all 15 steps per direction', fontsize=14)
        save(fig, f'{number}_model_taylor_{loss}', f'Full model Taylor sweeps: {loss}',
             'Every saved step for all three model coordinates and both aligned/mixed directions is shown. Remainders, central errors, declared central tolerances and recorded roundoff floors remain visible; coarse nonmatching steps are retained. Complete raw values and Boolean gates remain in inputs/results.json.',
             [], {'result_tests': names, 'all_steps_retained': True, 'display_floor_added': False})

    # Animation uses actual 1D native samples. All frames use full-history limits.
    frames = list(range(0, nt + 1, 5))
    if frames[-1] != nt: frames.append(nt)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    line, = ax.plot(x, u[0], color='#2878a4', lw=1.8)
    ax.set_ylim(-wave_bound * 1.13, wave_bound * 1.13)
    ax.axhline(0, color='.55', lw=.6)
    ax.axvline(x[source], color='#c44e52', ls='--', lw=1, label='Source at node 17')
    ax.plot(x[np.unique(rec)], np.full(np.unique(rec).size, -wave_bound * 1.04), '|', ms=12,
            color='.2', label='Receiver nodes (9 channels, 8 unique)')
    ax.legend(frameon=False, loc='upper right', fontsize=8)
    decorate(ax, ylabel='Native displacement u (simulation units)')
    def update(index):
        line.set_ydata(u[index])
        ax.set_title(f'Actual 1D native state u[{index}] | t = {times[index]:.1f} simulation T\n'
                     'Uniform sampling every 5 native states; playback is not physical real time', fontsize=11)
        return line,
    update(frames[0])
    movie = FuncAnimation(fig, update, frames=frames, blit=False)
    movie.save(args.out / 'native_wavefield.gif', writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)
    manifest['animations'].append({'output': 'native_wavefield.gif', 'array_key': 'main_history',
        'native_indices': frames, 'times': times[frames].tolist(), 'time_units': 'simulation T',
        'fps': 10, 'temporal_selection_rule': 'every fifth saved state including initial and final',
        'spatial_selection': 'all 43 native nodes', 'display_limits': [-wave_bound * 1.13, wave_bound * 1.13],
        'clipped_count': 0, 'interpolation': 'none; line connects native node samples',
        'source_and_receivers_overlay': True})

    manifest['plotting_source'] = {'path': Path(__file__).name, 'sha256': digest(__file__)}
    manifest['output_hashes'] = {p.name: digest(p) for p in sorted(args.out.iterdir())
                                 if p.suffix in ('.png', '.pdf', '.svg', '.gif')}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    (args.out / 'render_review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({'figures': len(manifest['figures']), 'animations': len(manifest['animations']),
                      'saved_recurrence_absolute_linf': audit_max,
                      'canvas_outside_text_count': sum(len(z['text_outside_canvas']) for z in review)}, indent=2))


if __name__ == '__main__':
    main()
