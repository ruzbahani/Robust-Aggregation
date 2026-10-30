from pathlib import Path
import argparse
import csv
import hashlib
import html
import io
import json

if __package__:
    from .result_tables import expected_tables, RESULTS, ROOT, SOURCES
else:
    from result_tables import expected_tables, RESULTS, ROOT, SOURCES

HEADERS = {
    'inversion': ['global echo', 'peer copy', 'lagged self', 'AR(2)', 'noise', 'shrink', 'echo: delta 0.1', 'echo: CIFAR-10'],
    'passport': ['value'], 'peercopy': ['rate'],
    'operating': ['worst TPR (clean FPR), nominal 10%', 'nominal 5%', 'nominal 1%', 'nominal 0.5%',
                  'worst AUC', 'AUC excluding copy source'],
    'combination': ['max raw', 'per-statistic', 'max robust z'],
    'combiners': ['honest FPR', 'worst TPR', 'worst AUC'],
    'effort': ['steps', 'innovation ratio', 'auditor AUC', 'auditor TPR', 'norm-bound AUC', 'Krum AUC', 'median AUC'],
    'effortdecomp': ['movement', 'alignment', 'innovation'],
    'effortsplit': ['parameter share', 'statistics-only displacement', 'alignment, recalibrated statistics',
                    'forward-only movement', 'forward-only innovation'],
    'effortdrivers': ['forward passes only', 'no computation', 'trains a tenth', 'trains half', 'no attack'],
    'adaptive': ['beta 0', 'beta 0.5', 'beta 1', 'beta 2'],
}
TITLES = {
    'inversion': 'Robust contribution rankings', 'passport': 'Passport binding',
    'operating': 'Frozen operating point', 'combination': 'Standardization',
    'combiners': 'Alternative combiners', 'effort': 'Real training effort',
    'effortdecomp': 'Fixed-readout decomposition', 'adaptive': 'Adaptive grid',
    'effortsplit': 'Parameter and normalization-statistics split',
    'effortdrivers': 'What flags each reduced-effort client',
    'peercopy': 'Copy attribution',
}


def render_table(key, rows):
    head ='<tr><th>measure / detector</th>' + ''.join(
        '<th>' + html.escape(x) + '</th>' for x in HEADERS[key]) + '</tr>'
    body = ''.join('<tr><th>' + html.escape(label) + '</th>' + ''.join(
        '<td>' + html.escape(v) + '</td>' for v in values) + '</tr>'
        for label, values in rows.items())
    return f'<h2>{html.escape(TITLES[key])}</h2><table>{head}{body}</table>'


def export_content():
    tables = expected_tables()
    files = {}
    for name, rows in tables.items():
        buf = io.StringIO(newline='')
        writer = csv.writer(buf, lineterminator='\n')
        writer.writerow(['measure / detector', *HEADERS[name]])
        for label, values in rows.items():
            if len(values) != len(HEADERS[name]):
                raise ValueError(f'{name}/{label}: inconsistent column count')
            writer.writerow([label, *values])
        files[name + '.csv'] = buf.getvalue()
    document = {'format': 1, 'tables': {k: {'title': TITLES[k], 'columns': HEADERS[k], 'rows': v}
                                     for k, v in tables.items()},
                'source_sha256': {n: hashlib.sha256((RESULTS / n).read_bytes()).hexdigest()
                                  for n in sorted(SOURCES)}}
    files['tables.json'] = json.dumps(document, ensure_ascii=False, indent=2) + '\n'
    return tables, files


def check_figures(tables, path):
    figure = json.loads(path.read_text(encoding='utf-8'))
    for label, values in tables['adaptive'].items():
        actual = [v.replace(r'\%', '%') for v in figure['rows'][label]]
        if actual != values:
            raise ValueError(f'Adaptive figure / {label}: {actual} != {values}')
    return sum(map(len, tables['adaptive'].values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'tables')
    ap.add_argument('--check', action='store_true', help='Verify exports without modifying them')
    ap.add_argument('--check-figures', action='store_true')
    args = ap.parse_args()
    tables, files = export_content()
    if not args.check:
        args.out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        path = args.out / name
        if args.check:
            if path.read_text(encoding='utf-8') != text:
                raise ValueError(f'Stale or changed result export: {path}')
        else:
            path.write_text(text, encoding='utf-8', newline='\n')
    cells = sum(len(v) for rows in tables.values() for v in rows.values())
    print(f'{len(tables)} table panels, {cells} result cells, {len(SOURCES)} source JSON files; '
          f'{"verified" if args.check else "exported"} to {args.out}')
    if args.check_figures:
        count = check_figures(tables, ROOT / 'figures/fig_adaptive_summary.values.json')
        print(f'{count} adaptive figure values agree with result tables')


if __name__ == '__main__':
    main()
