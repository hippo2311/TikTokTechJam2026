from __future__ import annotations

from contextlib import contextmanager

import torch


class TrainableParameterEMA:
    """EMA only for trainable parameters, avoiding a duplicate frozen backbone."""

    def __init__(self, model: torch.nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()}}

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        self.decay = float(state["decay"])
        self.shadow = {name: value.to(device) for name, value in state["shadow"].items()}

    @contextmanager
    def average_parameters(self, model: torch.nn.Module):
        parameters = dict(model.named_parameters())
        originals = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in originals.items():
                    parameters[name].copy_(value)
