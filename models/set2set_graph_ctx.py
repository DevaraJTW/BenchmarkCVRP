import torch
import torch.nn as nn

from torch_geometric.nn import Set2Set

from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy

from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.zoo.am.decoder import PrecomputedCache

class Set2SetModelDecoder(AttentionModelDecoder):
    def __init__(self, *args, processing_steps: int = 3, num_layers: int = 1, **kwargs):
        super().__init__(*args, **kwargs)

        self.s2s = Set2Set(
            in_channels=self.embed_dim,
            processing_steps=processing_steps,
            num_layers=num_layers,
        )
        self.proj = nn.Linear(2 * self.embed_dim, self.embed_dim)

    def train(self, mode: bool = True):
        # Let Lightning set the overall model mode
        super().train(mode)
        # Force Set2Set (and its internal LSTM) to stay in training mode
        self.s2s.train(True)
        return self

    def eval(self):
        # Lightning calls this for validation
        super().eval()
        # Keep Set2Set in training mode to avoid cuDNN RNN backward error
        self.s2s.train(True)
        return self
    
    def _precompute_cache(
        self, embeddings: torch.Tensor, num_starts: int = 0
    ) -> PrecomputedCache:
        #print("decoder.training:", self.training)
        #print("set2set.training:", self.s2s.training)
        (
            glimpse_key_fixed,
            glimpse_val_fixed,
            logit_key_fixed,
        ) = self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        if self.use_graph_context:
            if embeddings.ndim != 3:
                raise ValueError(f"Expected embeddings shape [B,N,D], got {tuple(embeddings.shape)}")
            
            B, N, D = embeddings.shape
            x = embeddings.reshape(B * N, D)
            
            index = torch.arange(B, device=embeddings.device).repeat_interleave(N)

            s2s_out = self.s2s(x, index=index, dim_size=B)  # [B, 2D]
            graph_context = self.proj(s2s_out)  # [B, D]
            
        else:
            graph_context = 0

        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

def getSet2SetPolicy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    ):
    
    set2setDecoder = Set2SetModelDecoder(
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
        decoder=set2setDecoder,
        )
    
    return policy