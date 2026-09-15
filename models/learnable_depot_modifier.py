from typing import Tuple
import math

import torch
import torch.nn as nn

from einops import rearrange
from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import CVRPEnv
from rl4co.utils.ops import unbatchify
from rl4co.models import AttentionModelPolicy

from rl4co.models.zoo.am.decoder import AttentionModelDecoder, PrecomputedCache

class LearnableDepotModifierDecoder(AttentionModelDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.depot_modifier = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

        nn.init.zeros_(self.depot_modifier[-1].weight)
        nn.init.zeros_(self.depot_modifier[-1].bias)

    def forward(
        self,
        td: TensorDict,
        cached: PrecomputedCache,
        num_starts: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the logits of the next actions given the current state

        Args:
            cache: Precomputed embeddings
            td: TensorDict with the current environment state
            num_starts: Number of starts for the multi-start decoding
        """

        has_dyn_emb_multi_start = self.is_dynamic_embedding and num_starts > 1

        # Handle efficient multi-start decoding
        if has_dyn_emb_multi_start:
            # if num_starts > 0 and we have some dynamic embeddings, we need to reshape them to [B*S, ...]
            # since keys and values are not shared across starts (i.e. the episodes modify these embeddings at each step)
            cached = cached.batchify(num_starts=num_starts)

        elif num_starts > 1:
            td = unbatchify(td, num_starts)

        glimpse_q = self._compute_q(cached, td)
        glimpse_k, glimpse_v, logit_k = self._compute_kvl(cached, td)

        # Compute logits
        mask = td["action_mask"]
        logits = self.pointer(glimpse_q, glimpse_k, glimpse_v, logit_k, mask)

        # Now we need to reshape the logits and mask to [B*S,N,...] is num_starts > 1 without dynamic embeddings
        # note that rearranging order is important here
        if num_starts > 1 and not has_dyn_emb_multi_start:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        problem_size = td["visited"].shape[-1] - 1
        log_problem_size_tensor = torch.tensor(
            [[math.log(problem_size)]],
            device=logits.device,
            dtype=logits.dtype,
        )
        depot_modifier = self.depot_modifier(log_problem_size_tensor)

        logits = torch.cat(
            [logits[:, :1] * torch.exp(depot_modifier), logits[:, 1:]], dim=-1
        )

        return logits, mask

def getLearnableDepotProbsPolicy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    ):

    learnableDepotModifierDecoder = LearnableDepotModifierDecoder(
        embed_dim=embeddingDimension,
        num_heads=num_heads,
        env_name=env.name,
        mask_inner=True,
        out_bias_pointer_attn=False,
        linear_bias=False,
        use_graph_context=True,
        check_nan=True,
        )
    
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=embeddingDimension,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        decoder=learnableDepotModifierDecoder,
        )
    
    return policy
