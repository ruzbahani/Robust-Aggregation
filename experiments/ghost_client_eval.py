from pathlib import Path
import argparse
import json
from negative_results import DEFAULT_ENTROPY, ghost_client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('results/negative_results.json'))
    args = ap.parse_args()
    result = json.loads(args.out.read_text(encoding='utf-8')) if args.out.exists() else {}
    entropy = result.get('meta', {}).get('entropy', DEFAULT_ENTROPY)
    result['N3_ghost_prover'] = ghost_client(warmup=5, entropy=bytes.fromhex(entropy))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['N3_ghost_prover'], indent=2))


if __name__ == '__main__':
    main()
