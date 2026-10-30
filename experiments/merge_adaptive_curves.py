from pathlib import Path
import argparse
import json


def merge_curves(paths):
    out = {'thresholds': None, 'honest_fpr_TEST': None, 'cells': {}, 'sources': []}
    for path in paths:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        for key in ('thresholds', 'honest_fpr_TEST'):
            if out[key] is None:
                out[key] = data[key]
            elif out[key] != data[key]:
                raise ValueError(f'Inconsistent {key}: {path}')
        overlap = set(out['cells']) & set(data['cells'])
        if overlap:
            raise ValueError(f'Duplicate adaptive cells: {sorted(overlap)}')
        out['cells'].update(data['cells'])
        out['sources'].append(str(path))
    if not out['cells']:
        raise ValueError('No adaptive cells')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sources', nargs='+', type=Path)
    ap.add_argument('--out', type=Path, default=Path('results/adaptive_costcurve.json'))
    args = ap.parse_args()
    args.out.write_text(json.dumps(merge_curves(args.sources), indent=2) + '\n', encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
