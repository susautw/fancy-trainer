import pytest
import torch

from fancy.trainer.model import TensorProxy


class TestTensorProxy:
    tensors = [
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, 0, 2, 3]),
        torch.tensor([0, 1, 0, 3]),
        torch.tensor([0, 1, 2, 0]),
    ]

    @pytest.fixture()
    def tensor(self):
        return TensorProxy(*self.tensors)

    def test_initialize_no_tensor_in(self):
        with pytest.raises(ValueError):
            TensorProxy()

    def test_initialize_tensor_idx_out_of_bound(self):
        with pytest.raises(IndexError):
            TensorProxy(torch.tensor([0, 1, 2, 3]), tensor_idx=1)

    @pytest.mark.parametrize("tensor_idx,expected_tensor", [(i, tensor) for i, tensor in enumerate(tensors)])
    def test_tensor_method(self, tensor_idx, expected_tensor, tensor: TensorProxy):
        tensor.current_tensor_idx = tensor_idx
        assert torch.equal(tensor + 2, expected_tensor + 2)
        assert torch.equal(tensor - 2, expected_tensor - 2)
        assert torch.equal(tensor * 2, expected_tensor * 2)
        assert torch.equal(tensor / 2, expected_tensor / 2)
