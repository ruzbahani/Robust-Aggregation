import copy
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from passportfl.passport import build_model, passport_bytes, make_passport_tensors
from passportfl import identity as idm
from passportfl.protocol import PassportHolder, Receiver, Contribution
from passportfl.probe import make_probes, select_subset, MerkleTree, leaf_hash

torch.manual_seed(0)
NC, D, W, M, T = 10, 16, 8, 32, 8


def make_holder(entropy, scheme="Ed25519", init_seed=0):
    torch.manual_seed(init_seed)
    m = build_model(NC, D, W, passport=True)
    return PassportHolder(m, scheme, M, T, "cpu", entropy)


def random_prototypes():
    return np.random.default_rng(1).normal(size=(NC, D))


def test_key_is_function_of_theta():
    a = make_holder(b"A" * 64); b = make_holder(b"A" * 64); c = make_holder(b"B" * 64); d = make_holder(b"A" * 64, init_seed=1)
    assert a.pk == b.pk == d.pk and a.pk != c.pk
    assert a.G == b.G and a.G != c.G
    assert a.G != d.G


def test_signature_schemes():
    for scheme in ("Ed25519", "ML-DSA-44"):
        h = make_holder(b"S" * 64, scheme)
        s = h._signer()
        assert idm.SCHEMES[scheme].verify(s.public_key, b"m", s.sign(b"m"))
        assert not idm.SCHEMES[scheme].verify(s.public_key, b"m2", s.sign(b"m"))


def test_honest_chain_accepted():
    h = make_holder(b"H" * 64)
    rx = Receiver(M, T, delta_per_round=1e9)
    for r in range(3):
        c = h.issue(random_prototypes() + r * 0.01, dict(n_samples=10, train_loss=1.0, local_acc=0.5, intra_compactness=1.0, inter_separation=2.0))
        res = rx.verify(c)
        assert res.accepted, res.reason
    assert rx.peers[h.G].round == 3


def record_fields():
    return dict(n_samples=10, train_loss=1.0, local_acc=0.5, intra_compactness=1.0, inter_separation=2.0)


def test_tampered_prototype_rejected():
    h = make_holder(b"T" * 64); rx = Receiver(M, T, 1e9, warmup_rounds=0)
    c = h.issue(random_prototypes(), record_fields())
    c2 = copy.deepcopy(c); c2.C[0, 0] += 1.0
    assert not rx.verify(c2).accepted
    assert rx.verify(c).accepted


def test_tampered_record_rejected():
    h = make_holder(b"R" * 64); rx = Receiver(M, T, 1e9, warmup_rounds=0)
    c = h.issue(random_prototypes(), record_fields())
    c2 = copy.deepcopy(c); c2.record.local_acc = 0.94
    assert "signature" in rx.verify(c2).reason


def test_foreign_passport_rejected():
    a = make_holder(b"a" * 64); b = make_holder(b"b" * 64); rx = Receiver(M, T, 1e9, warmup_rounds=0)
    ca = a.issue(random_prototypes(), record_fields())
    assert rx.verify(ca).accepted
    cb = b.issue(random_prototypes(), record_fields())
    cb.client_G = a.G; cb.genesis = None; cb.E_prev = ca.E_new; cb.round = 2
    assert not rx.verify(cb).accepted


def test_stale_and_replayed_rejected():
    h = make_holder(b"z" * 64); rx = Receiver(M, T, 1e9, warmup_rounds=0)
    c1 = h.issue(random_prototypes(), record_fields()); assert rx.verify(c1).accepted
    c2 = h.issue(random_prototypes(), record_fields()); assert rx.verify(c2).accepted
    assert "round" in rx.verify(copy.deepcopy(c1)).reason


def test_prototype_substitution_caught_by_probes():
    a = make_holder(b"p" * 64); b = make_holder(b"q" * 64, init_seed=7); rx = Receiver(M, T, 1e9, warmup_rounds=0)
    x = torch.randn(200, 3, 32, 32)

    def protos(h):
        with torch.no_grad():
            z = h.model.features(x).numpy()
        return np.stack([z[i::NC].mean(0) for i in range(NC)])

    Ca, Cb = protos(a), protos(b)
    for _ in range(2):
        assert rx.verify(a.issue(Ca, record_fields())).accepted
        assert rx.verify(b.issue(Cb, record_fields())).accepted
    res = rx.verify(b.issue(Ca, record_fields()))
    assert not res.accepted and res.chain_ok and "proto" in res.reason, res
    b2 = make_holder(b"q" * 64, init_seed=7); rx2 = Receiver(M, T, 1e9, warmup_rounds=0)
    for _ in range(2):
        assert rx2.verify(b2.issue(Cb, record_fields())).accepted
    sep = np.linalg.norm(Cb[:, None] - Cb[None], axis=-1).mean()
    Cb2 = Cb + 0.05 * sep * np.random.default_rng(3).normal(size=Cb.shape)
    assert rx2.verify(b2.issue(Cb2, record_fields())).accepted


def test_merkle_and_subset():
    leaves = [leaf_hash(np.random.default_rng(i).normal(size=D)) for i in range(M)]
    t = MerkleTree(leaves)
    for i in (0, 5, M - 1):
        assert MerkleTree.verify(t.root, leaves[i], t.proof(i))
    assert not MerkleTree.verify(t.root, leaves[0], t.proof(1))
    s = select_subset(b"\x01" * 32, M, T)
    assert len(set(s)) == T and s == select_subset(b"\x01" * 32, M, T)
    assert s != select_subset(b"\x02" * 32, M, T)


def test_probes_public_and_deterministic():
    p1 = make_probes(b"G" * 32, 4); p2 = make_probes(b"G" * 32, 4)
    assert torch.equal(p1, p2) and p1.shape == (4, 3, 32, 32)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
