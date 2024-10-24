from inspect import isfunction
from matplotlib import pyplot as plt
import torch
import numpy as np
import random
import weakref
import torch.nn.functional as F

def exists(x):
    return x is not None

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class EMA:
    """
    exponential moving average
    inspired by:
    [1] https://github.com/fadel/pytorch_ema
    [2] https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/training/moving_averages.py#L281-L685
    """
    def __init__(self, model, decay=0.9999):
        shadow = []
        refs = []
        for k, v in model.named_parameters():
            if v.requires_grad:
                shadow.append((k, v.detach().clone()))
                refs.append((k, weakref.ref(v)))
        self.shadow = dict(shadow)
        self._refs = dict(refs)
        self.decay = decay
        self.num_updates = -1
        self.backup = None

    def update(self):
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for k, _ref in self._refs.items():
            assert _ref() is not None, "referenced object no longer exists!"
            self.shadow[k] += (1 - decay) * (_ref().data - self.shadow[k])

    def apply(self):
        self.backup = dict([
            (k, _ref().detach().clone()) for k, _ref in self._refs.items()])
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.shadow[k])

    def restore(self):
        for k, _ref in self._refs.items():
            _ref().data.copy_(self.backup[k])
        self.backup = None

    def __enter__(self):
        self.apply()

    def __exit__(self, *exc):
        self.restore()

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "num_updates": self.num_updates
        }

    @property
    def extra_states(self):
        return {"decay", "num_updates"}

    def load_state_dict(self, state_dict, strict=True):
        _dict_keys = set(self.__dict__["shadow"]).union(self.extra_states)
        dict_keys = set(state_dict["shadow"]).union(self.extra_states)
        incompatible_keys = set.symmetric_difference(_dict_keys, dict_keys) \
            if strict else set.difference(_dict_keys, dict_keys)
        if incompatible_keys:
            raise RuntimeError(
                "Key mismatch!\n"
                f"Missing key(s): {', '.join(set.difference(_dict_keys, dict_keys))}."
                f"Unexpected key(s): {', '.join(set.difference(dict_keys, _dict_keys))}"
            )
        self.__dict__.update(state_dict)

if __name__ == '__main__':
    setup_seed(0)
    a = np.random.random()
    print(a)