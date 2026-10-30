from pathlib import Path
import json
import numpy as np


def draw_adaptive_summary(g):
    betas = (0, .5, 1, 2)
    alone = g.load_json('adaptive_costcurve.json')['cells']
    fused = g.load_json('combined_grid.json')['cells']
    beacon = g.load_json('unpredictable_eval.json')['adaptive']
    utility = g.load_json('utility_seed_spread.json')
    def a(b):
        return alone[f'suspicion/mimic|peer_current|phi=0|beta={b:g}']['stats']['suspicion']['auc']
    def f(b, rule):
        return fused[f'max|mimic|flat|peer_current|phi=0|beta={b:g}']['stats'][rule]
    rows = {
        'Payload auditor alone': [a(b) for b in betas],
        'Fused, maximum of seven': [f(b, 'combined_max')['auc'] for b in betas],
        'Fused, weighted sum': [f(b, 'combined_sum')['auc'] for b in betas],
        'Fused, two-of-seven': [f(b, 'combined_two')['auc'] for b in betas],
        'Beacon, three of five': [a(0)] + [beacon[f'sub3|worst|peer_current|phi=0|beta={b:g}']['auc'] for b in betas[1:]],
        'TPR': [f(b, 'combined_max')['tpr_client_round_TEST'] for b in betas],
    }
    assert np.isfinite(np.array(list(rows.values()))).all()
    printed = {k: [(f'{100*v:.0f}\\%' if k == 'TPR' else f'{v:.3f}') for v in vs]
               for k, vs in rows.items()}
    fig = g.plt.figure(figsize=(g.COL1, 5.0))
    grid = fig.add_axes([.31, .58, .65, .33])
    cmap = g.plt.get_cmap('RdBu')
    from matplotlib.patches import Rectangle
    for y, (label, vals) in enumerate(rows.items()):
        for x, val in enumerate(vals):
            color = cmap(val) if label != 'TPR' else '#E6E6E6'
            grid.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor=color,
                                     edgecolor='white', linewidth=1))
            text = printed[label][x].replace('\\%', '%')
            grid.text(x, y, text, ha='center', va='center', fontsize=g.FS_LABEL,
                      color=('white' if label != 'TPR' and (val < .16 or val > .84) else '#161616'))
    grid.set_xlim(-.5, 3.5)
    grid.set_ylim(5.5, -.5)
    grid.set_xticks(range(4), [f'{b:g}' for b in betas])
    grid.xaxis.tick_top()
    grid.set_yticks(range(6), ['Payload auditor', 'Fused maximum', 'Fused mean',
                             'Two-of-seven', 'Beacon: three of five', 'Test TPR: fused maximum'])
    grid.tick_params(axis='both', length=0, labelsize=g.FS_LABEL, pad=5)
    grid.set_title(r'Perturbation budget $\beta$', fontsize=g.FS_LABEL, pad=20)
    for spine in grid.spines.values():
        spine.set_visible(False)
    fig.text(.035, .955, '(a)', fontsize=g.FS_PANEL, fontweight='bold')

    curve = fig.add_axes([.105, .12, .385, .33])
    colors = [g.C_OURS, g.C_CROSS, '#7A5195', g.C_MAXALL, g.C_BREAK]
    markers = ['s', '^', 'o', 'D', 'v']
    names = ['Auditor', 'Maximum', 'Mean', 'Two-of-seven', 'Beacon']
    styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
    curve.axhline(.5, color='#777777', lw=.7, ls='--')
    for (key, vals), col, marker, name, style in zip(rows.items(), colors, markers, names, styles):
        curve.plot(betas, vals, color=col, marker=marker, ms=3, lw=1,
                   mfc='white', mew=.7, ls=style, label=name)
    curve.set(xlim=(-.06, 2.06), ylim=(-.03, 1.07), xlabel=r'Budget $\beta$', ylabel='Detection AUC')
    curve.set_xticks(betas)
    curve.legend(loc='lower left', fontsize=g.FS_MIN, frameon=False, ncol=2,
                 handlelength=1.3, columnspacing=.5, labelspacing=.25)
    curve.tick_params(labelsize=g.FS_TICK, length=2)
    fig.text(.035, .485, '(b)', fontsize=g.FS_PANEL, fontweight='bold')

    util = fig.add_axes([.635, .12, .335, .33])
    band = utility['honest_range']
    util.axhspan(-band, band, color='#DADADA', lw=0)
    util.axhline(0, color='#666666', lw=.7)
    for x, key in enumerate(('fused_b1', 'fused_b2', 'fused_b1_x3')):
        r = utility['scenarios'][key]
        vals = r['per_seed_drop']
        col = g.C_OURS if r['consistent_sign'] else g.C_BREAK
        util.scatter(x + np.linspace(-.15, .15, len(vals)), vals, s=16,
                     facecolors='white', edgecolors=col, linewidth=.8, zorder=4)
        util.plot([x-.27, x+.27], [r['mean_drop']]*2, color=col, lw=1.5)
    util.set_xticks(range(3), [r'$\beta=1$', r'$\beta=2$', r'$\beta=1$'+'\nthree clients'])
    util.set(xlim=(-.5, 2.5), ylabel='Honest accuracy lost')
    util.tick_params(labelsize=g.FS_TICK, length=2)
    fig.text(.555, .485, '(c)', fontsize=g.FS_PANEL, fontweight='bold')
    for ax in (curve, util):
        ax.grid(axis='y', color='#DEDEDE', lw=.4)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    result = g.save_figure(fig, 'fig_adaptive_summary')
    (Path(g.FIGDIR) / 'fig_adaptive_summary.values.json').write_text(
        json.dumps({'betas': betas, 'rows': printed}, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    import make_figures as g
    g.install_guards()
    g.assert_fonts()
    g.set_style()
    draw_adaptive_summary(g)
    g.assert_clean_log()
    g.assert_no_captions()
