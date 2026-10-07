"""Minimal stand-in for torch, enough for _scale_lora_weights.

Real ComfyUI has torch; the sandbox here does not. Only `.to()` and tensor
multiplication are exercised, so numpy backs it.
"""

import numpy as np


class _T:
    def __init__(self, data):
        self.data = (np.asarray(data, dtype=np.float32) if not isinstance(data, _T)
                     else data.data)

    def __getitem__(self, i):
        return _T(self.data[i])

    def __len__(self):
        return len(self.data)

    def __mul__(self, other):
        return _T(self.data * (other.data if isinstance(other, _T) else other))

    def __rmul__(self, other):
        return _T(other * self.data)

    def to(self, dtype=None, *a, **kw):
        return self

    def numpy(self):
        return self.data

    @property
    def dtype(self):
        return "float32"


float32 = "float32"
float16 = "float16"
bfloat16 = "bfloat16"
