import hashlib
import platform


def experiment_entropy(seed: int, client: int, domain: str) -> bytes:
    label = f"passportfl/experiment/v1/{domain}/{seed}/{client}".encode()
    return hashlib.sha512(label).digest()


def environment():
    import numpy
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": numpy.__version__, "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
