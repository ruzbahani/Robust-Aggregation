from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from . import identity as idm
from .identity import LearningRecord
from .passport import PassportResNet18, passport_bytes
from .probe import (ProbeReveal, ProbeCheckResult, check_reveals, commit_and_reveal,
                    embed_probes, make_probes, MerkleTree, leaf_hash)


@dataclass
class GenesisCert:
    G: bytes
    pk: bytes
    arch: bytes
    F0: bytes
    scheme: str


@dataclass
class Contribution:
    client_G: bytes
    round: int
    E_prev: bytes
    E_new: bytes
    C: np.ndarray
    record: LearningRecord
    F: bytes
    reveals: List[ProbeReveal]
    sig: bytes
    genesis: Optional[GenesisCert] = None

    def wire_size(self) -> int:
        return len(pickle.dumps(self))


class PassportHolder:
    def __init__(self, model: PassportResNet18, scheme: str = "ML-DSA-44", m_probes: int = 256, t_reveal: int = 32,
                 device: str = "cpu", entropy: Optional[bytes] = None, external_key: Optional[bytes] = None):
        self.model = model
        self.scheme = scheme
        self.m = m_probes
        self.t = t_reveal
        self.device = device
        if not model.passport_tensors:
            model.genesis(entropy)
        self.theta_bytes = passport_bytes(model.passport_tensors)
        self.model_bound = len(self.theta_bytes) >= idm.MIN_KEY_MATERIAL
        if not self.model_bound:
            self.theta_bytes = external_key if external_key is not None else os.urandom(64)
        signer = self._signer()
        self.pk = signer.public_key
        self.arch = idm.architecture_descriptor(model)
        pre_G = idm.H(b"preG|" + self.arch + b"|" + self.pk)
        self.probes = make_probes(pre_G, m_probes).to(device)
        embs0 = embed_probes(model, self.probes, device)
        self.F0 = MerkleTree([leaf_hash(e) for e in embs0]).root
        self.G = idm.genesis_id(self.arch, self.pk, self.F0)
        self.E = idm.GENESIS_E
        self.round = 0
        self.prev_C: Optional[np.ndarray] = None
        model.install_dynamic(None)

    def _signer(self):
        return idm.derive_signer(self.theta_bytes, self.scheme)

    def genesis_cert(self) -> GenesisCert:
        return GenesisCert(self.G, self.pk, self.arch, self.F0, self.scheme)

    def issue(self, C: np.ndarray, record_fields: dict) -> Contribution:
        self.round += 1
        r = self.round
        C = idm.dequantize(C)
        drift = 0.0 if self.prev_C is None else float(np.linalg.norm(C - self.prev_C, axis=1).mean())
        record = LearningRecord(round=r, proto_drift=drift, **record_fields)
        embs = embed_probes(self.model, self.probes, self.device)
        F, reveals = commit_and_reveal(embs, self.E, self.t)
        KC = idm.hash_prototypes(C)
        Rd = record.digest()
        msg = idm.signed_message(r, self.E, KC, Rd, F)
        sig = self._signer().sign(msg)
        E_new = idm.evolving_id(r, self.E, KC, Rd, F, sig)
        contrib = Contribution(self.G, r, self.E, E_new, C, record, F, reveals, sig,
                               genesis=self.genesis_cert() if r == 1 else None)
        self.E = E_new
        self.prev_C = C
        self.model.install_dynamic(E_new)
        return contrib


@dataclass
class PeerState:
    cert: GenesisCert
    E: bytes = idm.GENESIS_E
    round: int = 0
    probe_history: Dict[int, Tuple[int, np.ndarray]] = field(default_factory=dict)
    C_prev: Optional[np.ndarray] = None
    C_prev_round: int = 0


@dataclass
class VerifyResult:
    accepted: bool
    reason: str
    chain_ok: bool = False
    probe: Optional[ProbeCheckResult] = None


class Receiver:
    def __init__(self, m_probes: int = 256, t_reveal: int = 32, delta_per_round: float = 1.0,
                 max_proto_ratio: float = 1.0, min_label_stability: float = 0.5, warmup_rounds: int = 5):
        self.peers: Dict[bytes, PeerState] = {}
        self.m = m_probes
        self.t = t_reveal
        self.delta = delta_per_round
        self.max_ratio = max_proto_ratio
        self.min_stab = min_label_stability
        self.warmup = warmup_rounds

    def _register(self, cert: GenesisCert) -> Optional[str]:
        if idm.genesis_id(cert.arch, cert.pk, cert.F0) != cert.G:
            return "genesis id does not bind (arch, pk, F0)"
        if cert.scheme not in idm.SCHEMES:
            return "unknown signature scheme"
        self.peers[cert.G] = PeerState(cert)
        return None

    def verify(self, c: Contribution) -> VerifyResult:
        if c.client_G not in self.peers:
            if c.genesis is None or c.genesis.G != c.client_G:
                return VerifyResult(False, "unknown peer and no genesis certificate")
            err = self._register(c.genesis)
            if err:
                return VerifyResult(False, err)
        st = self.peers[c.client_G]
        if c.round != st.round + 1:
            return VerifyResult(False, f"round mismatch: expected {st.round + 1}, got {c.round}")
        if c.E_prev != st.E:
            return VerifyResult(False, "E_prev does not match stored chain head")
        KC = idm.hash_prototypes(c.C)
        Rd = c.record.digest()
        if c.record.round != c.round:
            return VerifyResult(False, "record round mismatch")
        msg = idm.signed_message(c.round, c.E_prev, KC, Rd, c.F)
        if not idm.SCHEMES[st.cert.scheme].verify(st.cert.pk, msg, c.sig):
            return VerifyResult(False, "signature invalid")
        if idm.evolving_id(c.round, c.E_prev, KC, Rd, c.F, c.sig) != c.E_new:
            return VerifyResult(False, "E_new does not recompute")
        st.E = c.E_new
        st.round = c.round
        pr = check_reveals(c.F, c.reveals, c.C, st.C_prev, st.C_prev_round, c.E_prev, self.m, self.t,
                           st.probe_history, c.round, self.delta, self.max_ratio, self.min_stab, self.warmup)
        if pr.merkle_ok and pr.subset_ok:
            st.C_prev = c.C
            st.C_prev_round = c.round
        if not pr.ok:
            why = [k for k, v in (("merkle", pr.merkle_ok), ("subset", pr.subset_ok), ("proto", pr.proto_ok),
                                  ("stability", pr.stability_ok), ("drift", pr.drift_ok)) if not v]
            return VerifyResult(False, "probe check failed: " + ",".join(why), True, pr)
        return VerifyResult(True, "ok", True, pr)
