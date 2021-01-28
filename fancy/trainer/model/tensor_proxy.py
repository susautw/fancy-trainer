from typing import Tuple, Callable

import torch


class TensorProxy(torch.Tensor):
    _tensors: Tuple[torch.Tensor] = None
    _current_tensor_idx: int = None

    def __init__(self, *tensors: torch.Tensor, tensor_idx: int = 0):
        self._tensors = tensors
        if len(self._tensors) > 0:
            if tensor_idx < len(self._tensors):
                self._current_tensor_idx = tensor_idx
            else:
                raise IndexError(f"tensor_idx out of bound: {tensor_idx} >= {len(self._tensors)}")
        else:
            raise ValueError("Need at least one tensor.")

    def __new__(cls, *args, **kwargs):
        return super(TensorProxy, cls).__new__(cls)

    def __dir__(self):
        return object.__dir__(self)

    def __repr__(self):
        return repr(self.current_tensor)

    def __str__(self):
        return str(self.current_tensor)

    @property
    def current_tensor_idx(self) -> int:
        return self._current_tensor_idx

    @current_tensor_idx.setter
    def current_tensor_idx(self, val: int) -> None:
        self._current_tensor_idx = val

    @property
    def num_tensors(self) -> int:
        return len(self._tensors)

    @property
    def current_tensor(self) -> torch.Tensor:
        return self._tensors[self._current_tensor_idx]

    def __getattribute__(self, item: str):
        if item not in dir(torch.Tensor):
            return super(TensorProxy, self).__getattribute__(item)
        else:
            return getattr(self.current_tensor, item)


def _wrapper(method_name):
    def _forward(self, *args, **kwargs):
        return self.__getattribute__(method_name)(*args, **kwargs)

    return _forward

for name in set(dir(torch.Tensor)).difference(dir(object)):
    if isinstance(type(getattr(torch.Tensor, name)), Callable):
        try:
            setattr(TensorProxy, name, _wrapper(name))
        except AttributeError:
            pass
