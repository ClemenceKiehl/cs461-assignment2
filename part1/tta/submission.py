from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from tta.base import TTAMethod


class Submission(TTAMethod):
    """
    TENT-style Test-Time Adaptation:
    - We update only BatchNorm affine parameters (gamma/beta)
      by minimizing prediction entropy on the current batch.
    - The wrapped model is a pretrained ResNet-50.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        steps_per_batch: int = 1,
        eps: float = 1e-8,
    ):
        """
        Args come from configs/submission.yaml under `tta.args`.
        """
        # store wrapped model in self.model (TTAMethod takes care of registration)
        super().__init__(model)

        self.eps = eps
        self.steps_per_batch = steps_per_batch

        # We will adapt only BN affine parameters
        self._configure_bn_params()

        # Optimizer on BN affine params only
        self.optimizer = torch.optim.Adam(
            self.bn_params, lr=lr, weight_decay=weight_decay
        )

        # Save initial model state so we can reset between corruptions
        self.model_state = deepcopy(self.model.state_dict())


    # Helper to choose which params are trainable

    def _configure_bn_params(self):
        # first freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        self.bn_params = []
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d) and m.affine:
                m.weight.requires_grad = True
                m.bias.requires_grad = True
                self.bn_params.append(m.weight)
                self.bn_params.append(m.bias)

        if not self.bn_params:
            raise RuntimeError("No BatchNorm2d affine parameters found to adapt.")




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Called for each batch in eval.py.

        We:
          1. Run a few entropy-minimization steps on BN params.
          2. Return logits from the adapted model.
        """
        # make sure gradients are enabled & model in train mode for BN
        self.train()

        for _ in range(self.steps_per_batch):
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)

            # entropy per sample: -sum p log p
            entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=1).mean()

            self.optimizer.zero_grad(set_to_none=True)
            entropy.backward()
            self.optimizer.step()

        # After adaptation, we use the last logits for prediction

        return logits


    # Reset between different corruption types

    def reset(self) -> None:
        """
        Called in eval.py before each new corruption type.
        We restore the original model weights and reset optimizer state.
        """
        self.model.load_state_dict(self.model_state, strict=True)
        self._configure_bn_params()
        # re-create optimizer to clear state
        self.optimizer = torch.optim.Adam(self.bn_params, lr=self.optimizer.param_groups[0]["lr"])
