"""Generic FLA models."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, asdict

import warnings

import torch
from torch import Tensor, nn

try:
    from fla import (
        GatedDeltaProductForCausalLM, GLAForCausalLM, TransformerForCausalLM)
except ImportError:
    warnings.warn(
        "[model] FLA models not available", RuntimeWarning)
    TransformerForCausalLM = None
    GLAForCausalLM = None
    GatedDeltaProductForCausalLM = None

# Use Grazzi/Siems' Negative-Mamba
try:
    from mamba_ssm import MambaLMHeadModel
    from mamba_ssm.models.config_mamba import MambaConfig
except ImportError:
    warnings.warn(
        "[model] Mamba models not available", RuntimeWarning)
    MambaLMHeadModel = None
    MambaConfig = None

try:
    from wavesAI.model.aussm import SSMSeq2Seq
except ImportError:
    warnings.warn(
        "wavesAI.model.aussm not available — SSMSeq2Seq disabled",
        RuntimeWarning,
    )
    SSMSeq2Seq = None


class CausalLMMixin:
    """
    Drop-in mixin for *ForCausalLM models.
    Assumes the parent class implements forward(**kwargs) returning an object with .logits
    """

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        # Keep signature consistent across your wrappers: accept x, pass through extras.
        out = super().forward(input_ids=x, attention_mask=attention_mask, **kwargs)
        return out.logits

    @torch.no_grad()
    def get_useful_stats(self) -> Dict[str, Any]:
        # Default stub; override in specific models if you want richer stats
        return {}

if GatedDeltaProductForCausalLM is not None:
    class GatedDeltaProduct(CausalLMMixin, GatedDeltaProductForCausalLM):
        pass

if GLAForCausalLM is not None:
    class GLA(CausalLMMixin, GLAForCausalLM):
        pass

# class MambaLSTM(CausalLMMixin, MambaForCausalLM):
#     pass

if TransformerForCausalLM is not None:
    class VanillaTransformer(CausalLMMixin, TransformerForCausalLM):
        pass


class ConfigWithToDict:
    def __init__(self, cfg: Any):
        self._cfg = cfg

    def to_dict(self) -> Dict[str, Any]:
        d = dict(vars(self._cfg))

        # Make sure nested objects become serializable
        return d


class MambaLSTM(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self._raw_config = config
        self.config = ConfigWithToDict(config)
        self.model = MambaLMHeadModel(config)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        _ = attention_mask  # ignored
        out = self.model(x, **kwargs)
        return out.logits

    @torch.no_grad()
    def get_useful_stats(self) -> Dict[str, Any]:
        return {}


@dataclass
class AussmConfig:
    d_model: int = 2560
    vocab_size: int = 9
    output_vocab_size: int = 9
    layers: Union[str, List[int]] = "a|a"
    d_state: int = 16
    mamba_expand: int = 2
    dt_rank: Union[str, int] = "auto"
    d_conv: int = 4
    conv_bias: bool = True
    bias: bool = False
    ssmau_conv_1d: bool = False
    embedding_decay: bool = False
    ssmau_cuda: bool = True
    verbose: bool = True


class AUSSM(nn.Module):
    def __init__(self, config: AussmConfig):
        super().__init__()
        self._raw_config = config
        self.config = ConfigWithToDict(config)
        self.model = SSMSeq2Seq(**asdict(config))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Tensor:
        _ = attention_mask  # ignored
        out = self.model(x, **kwargs)
        return out

    @torch.no_grad()
    def get_useful_stats(self) -> Dict[str, Any]:
        return {}