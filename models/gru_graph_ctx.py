import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from tensordict import TensorDict

from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy
from rl4co.envs import RL4COEnvBase
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.zoo.am.decoder import PrecomputedCache

class GRUGraphCTXModelDecoder(AttentionModelDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.gru = nn.GRU(self.embed_dim, self.embed_dim, batch_first=True)

    def train(self, mode: bool = True):
        # Let Lightning set the overall model mode
        super().train(mode)
        # Force Set2Set (and its internal LSTM) to stay in training mode
        self.gru.train(True)
        return self

    def eval(self):
        # Lightning calls this for validation
        super().eval()
        # Keep Set2Set in training mode to avoid cuDNN RNN backward error
        self.gru.train(True)
        return self

    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> PrecomputedCache:
        """Compute the cached embeddings for the pointer attention.

        Args:
            embeddings: Precomputed embeddings for the nodes
            num_starts: Number of starts for the multi-start decoding
        """
        # The projection of the node embeddings for the attention is calculated once up front
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        # Optionally disable the graph context from the initial embedding as done in POMO
        if self.use_graph_context:
            _, final_hs = self.gru(embeddings)  # [B, N, D], [1, B, D]
            graph_context = final_hs[0]         # [B, D]
        else:
            graph_context = 0

        # Organize in a dataclass for easy access
        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

def getGRUGraphCTXPolicy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    ):
    
    weightedAverageDecoder = GRUGraphCTXModelDecoder(
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
        decoder=weightedAverageDecoder,
        )
    
    return policy