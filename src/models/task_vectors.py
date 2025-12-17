import abc
from typing import OrderedDict, Union

import torch

from src.models.modeling import ImageEncoder
from src.utils.variables_and_paths import MODELS

_Checkpoint = Union[str, dict, torch.nn.Module]


def symmetric_difference(A, B):
    """Returns the symmetric difference between two lists."""
    return list(set(A) ^ set(B))


class _TaskVector(abc.ABC):
    def __init__(
        self,
        model_name,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        assert (
            model_name in MODELS
        ), f"Invalid model name: {model_name}. Valid models are: {MODELS}"
        self.model_name = model_name
        if vector is not None:
            self.vector = vector
        else:
            assert (
                pretrained_checkpoint is not None and finetuned_checkpoint is not None
            )
            with torch.no_grad():
                # parse pretrained_checkpoint
                pretrained_state_dict = self._safe_load(pretrained_checkpoint)

                # parse finetuned_checkpoint
                finetuned_state_dict = self._safe_load(finetuned_checkpoint)

                if "model_name" in finetuned_state_dict.keys():
                    finetuned_state_dict.pop("model_name")

                # the final task vector is the difference between finetuned and pretrained vectors.
                assert (
                    pretrained_state_dict.keys() == finetuned_state_dict.keys()
                ), f"State dicts have different keys: {symmetric_difference(pretrained_state_dict.keys(), finetuned_state_dict.keys())}."
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype == torch.int64:
                        continue
                    if pretrained_state_dict[key].dtype == torch.uint8:
                        continue
                    self.vector[key] = (
                        finetuned_state_dict[key] - pretrained_state_dict[key]
                    )

    def _safe_load(self, checkpoint):
        if isinstance(checkpoint, str):
            return self._load_checkpoint(checkpoint).state_dict()
        elif isinstance(checkpoint, dict):
            return checkpoint
        elif isinstance(checkpoint, torch.nn.Module):
            return checkpoint.state_dict()
        else:
            raise ValueError(f"Invalid type for checkpoint: {type(checkpoint)}")

    @abc.abstractmethod
    def _load_checkpoint(self, checkpoint) -> torch.nn.Module:
        """Load a checkpoint into a model."""
        raise NotImplementedError

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return self.__class__(vector=new_vector)

    def __sub__(self, other):
        """Subtract two task vectors."""
        return self.__add__(-other)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return self.__class__(vector=new_vector)

    def __pow__(self, power):
        """Power of a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] ** power
        return self.__class__(vector=new_vector)

    def __mul__(self, other):
        """Multiply a task vector by a scalar."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = other * self.vector[key]
        return self.__class__(vector=new_vector)

    def dot(self, other):
        """Dot product of two task vectors."""
        with torch.no_grad():
            dot_product = 0.0
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                dot_product += torch.sum(self.vector[key] * other.vector[key])
        return dot_product

    def norm(self):
        """Norm of a task vector."""
        return torch.sqrt(self.dot(self))

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, args=None):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = self._load_checkpoint(pretrained_checkpoint)
            pretrained_model = pretrained_model.to(args.device)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"  # noqa: E501
                    )
                    continue
                dvc = self.vector[key].device
                new_state_dict[key] = pretrained_state_dict[key].to(
                    dvc
                ) + scaling_coef * self.vector[key].to(dvc)
        pretrained_model.load_state_dict(new_state_dict)
        return pretrained_model


class NonLinearTaskVector(_TaskVector):
    """A task vector for nonlinear models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return ImageEncoder.load(self.model_name, checkpoint)

    def apply_to_nonlinear(self, pretrained_nonlinear_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a nonlinear pretrained model."""
        return self.apply_to(pretrained_nonlinear_checkpoint, scaling_coef)
