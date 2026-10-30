from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np

H = lambda b: hashlib.sha256(b).digest()
NAN_SENTINEL = -1.0e9
DOMAIN_KEY = b"passportfl/v1/signing-key"
DOMAIN_PROBE = b"passportfl/v1/probe-seed"
DOMAIN_SUBSET = b"passportfl/v1/probe-subset"
FIXED_POINT_SCALE = 1 << 16


class Signer:
    name: str = "none"

    def __init__(self, seed32: bytes):
        raise NotImplementedError

    @property
    def public_key(self) -> bytes:
        raise NotImplementedError

    def sign(self, msg: bytes) -> bytes:
        raise NotImplementedError

    @staticmethod
    def verify(pk: bytes, msg: bytes, sig: bytes) -> bool:
        raise NotImplementedError


class MLDSA44Signer(Signer):
    name = "ML-DSA-44"

    def __init__(self, seed32: bytes):
        from dilithium_py.ml_dsa import ML_DSA_44
        self._impl = ML_DSA_44
        self._pk, self._sk = ML_DSA_44.key_derive(seed32)

    @property
    def public_key(self) -> bytes:
        return self._pk

    def sign(self, msg: bytes) -> bytes:
        return self._impl.sign(self._sk, msg, deterministic=True)

    @staticmethod
    def verify(pk: bytes, msg: bytes, sig: bytes) -> bool:
        from dilithium_py.ml_dsa import ML_DSA_44
        try:
            return bool(ML_DSA_44.verify(pk, msg, sig))
        except Exception:
            return False


class Ed25519Signer(Signer):
    name = "Ed25519"

    def __init__(self, seed32: bytes):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        self._sk = Ed25519PrivateKey.from_private_bytes(seed32)
        self._pk = self._sk.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw)

    @property
    def public_key(self) -> bytes:
        return self._pk

    def sign(self, msg: bytes) -> bytes:
        return self._sk.sign(msg)

    @staticmethod
    def verify(pk: bytes, msg: bytes, sig: bytes) -> bool:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        try:
            Ed25519PublicKey.from_public_bytes(pk).verify(sig, msg)
            return True
        except Exception:
            return False


SCHEMES = {"ML-DSA-44": MLDSA44Signer, "Ed25519": Ed25519Signer}


MIN_KEY_MATERIAL = 32


def derive_signer(theta_bytes: bytes, scheme: str = "ML-DSA-44") -> Signer:
    if len(theta_bytes) < MIN_KEY_MATERIAL:
        raise ValueError(
            f"refusing to derive a signing key from {len(theta_bytes)} bytes of passport "
            f"material (need >= {MIN_KEY_MATERIAL}). A model with no passport-bearing "
            f"layers yields b'' here, which would collapse every participant onto one key."
        )
    seed = hmac.new(DOMAIN_KEY, theta_bytes, hashlib.sha256).digest()
    return SCHEMES[scheme](seed)


def canonical_vectors(x: np.ndarray) -> bytes:
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    q = np.clip(np.round(x * FIXED_POINT_SCALE), -2**31, 2**31 - 1).astype("<i4")
    shape = np.asarray(x.shape, dtype="<i4").tobytes()
    return len(x.shape).to_bytes(1, "little") + shape + q.tobytes()


def dequantize(x: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(x, dtype=np.float64) * FIXED_POINT_SCALE) / FIXED_POINT_SCALE


def hash_prototypes(C: np.ndarray) -> bytes:
    return H(canonical_vectors(C))


def canonical_json(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass
class LearningRecord:
    round: int
    n_samples: int
    train_loss: float
    local_acc: float
    proto_drift: float
    intra_compactness: float
    inter_separation: float

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float):
                d[k] = NAN_SENTINEL if not math.isfinite(v) else round(v, 6)
        return d

    def has_nonfinite(self) -> bool:
        return any(v == NAN_SENTINEL for v in self.to_dict().values() if isinstance(v, float))

    def digest(self) -> bytes:
        return H(canonical_json(self.to_dict()))


def architecture_descriptor(model) -> bytes:
    spec = model.passport_spec()
    d = {"family": "PassportResNet18", "embed_dim": model.embed_dim, "num_classes": model.num_classes,
         "passport_in_channels": spec.in_channels}
    return canonical_json(d)


def genesis_id(arch: bytes, pk: bytes, F0: bytes) -> bytes:
    return H(b"G|" + arch + b"|" + pk + b"|" + F0)


def signed_message(round_: int, E_prev: bytes, KC: bytes, R_digest: bytes, F: bytes) -> bytes:
    return b"M|" + round_.to_bytes(4, "little") + E_prev + KC + R_digest + F


def evolving_id(round_: int, E_prev: bytes, KC: bytes, R_digest: bytes, F: bytes, sig: bytes) -> bytes:
    return H(b"E|" + round_.to_bytes(4, "little") + E_prev + KC + R_digest + F + H(sig))


GENESIS_E = bytes(32)
