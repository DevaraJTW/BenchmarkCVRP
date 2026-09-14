import math
import torch
import torch.nn as nn
from torch import Tensor

from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy
from torch.nn.functional import scaled_dot_product_attention

class LearnableSDPAScalingFactor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling_factor_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        
        nn.init.zeros_(self.scaling_factor_net[-1].weight)
        nn.init.zeros_(self.scaling_factor_net[-1].bias)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            attn_mask: Tensor | None = None,
            dropout_p: float = 0,
            is_causal: bool = False,
            scale: float | None = None,
            enable_gqa: bool = False
    ) -> Tensor:

        n = key.size(-2) - 1
        log_n = torch.log(torch.tensor(
            [[n]],
            device=query.device,
            dtype=query.dtype,
        ))
        log_scaling = self.scaling_factor_net(log_n).squeeze()
        scaling_factor = torch.exp(log_scaling)
        query = query * scaling_factor
        scale = 1 / math.sqrt(query.size(-1))

        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )


def getLearnableScalingFactorPolicy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    #problem_size: int,
    ):

    learnable_sdpa_fn = LearnableSDPAScalingFactor()
    
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=embeddingDimension,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        sdpa_fn=learnable_sdpa_fn,
        )
    
    return policy
