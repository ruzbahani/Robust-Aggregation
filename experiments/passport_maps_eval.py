from pathlib import Path
import argparse
import hashlib
import json
import platform
import numpy as np
import torch
from torch.utils.data import DataLoader
from negative_results import (DEFAULT_ENTROPY, build_model, load_cifar100, rank_report,
                              exact_forgery, sybil_across_clients, ghost_client, jsonable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, default=Path('checkpoints/cifar100_client0.pt'))
    ap.add_argument('--data-root', default='data')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--entropy', default=DEFAULT_ENTROPY)
    ap.add_argument('--across-clients', action='store_true')
    ap.add_argument('--out', type=Path, default=Path('results/n1_with_rescale.json'))
    ap.add_argument('--negative-out', type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    np.random.seed(0)
    entropy = bytes.fromhex(args.entropy)
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = build_model(100, 128, 64, passport=True, alpha=0.0).to(args.device)
    model.load_state_dict(ck['state'])
    theta = {k: v.to(args.device) for k, v in ck['passport'].items()}
    model.install_passport(theta)
    model.install_dynamic(None)
    _, test = load_cifar100(root=args.data_root)
    batches = [(x.to(args.device), y.to(args.device)) for x, y in DataLoader(test, batch_size=1000)]
    rank = rank_report(model, args.device)
    result = exact_forgery(model, theta, batches, ck['global_C'].to(args.device),
                             [0.0, 1.0, 5.0, 50.0], args.device, entropy, sybil_clones=0)
    if args.across_clients:
        paths = [args.checkpoint.with_name(f'cifar100_client{i}.pt') for i in (1, 2, 3)]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        result['across_clients'] = sybil_across_clients(
            [str(p) for p in paths], batches, args.device, 100, 128, 64, entropy)
    meta = {'device': args.device, 'torch': torch.__version__, 'python': platform.python_version(),
            'checkpoint': str(args.checkpoint), 'entropy': args.entropy,
            'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'rank_deficiency': rank, 'N1': result, 'meta': meta},
                                  indent=2, default=jsonable) + '\n', encoding='utf-8')
    if args.negative_out:
        negative = json.loads(args.negative_out.read_text()) if args.negative_out.exists() else {}
        negative.update(meta=meta, rank_deficiency=rank, N1_exact_forgery=result,
                        N3_ghost_prover=ghost_client(warmup=5, entropy=entropy))
        args.negative_out.write_text(json.dumps(negative, indent=2, default=jsonable) + '\n', encoding='utf-8')
    print(args.out)


if __name__ == '__main__':
    main()
