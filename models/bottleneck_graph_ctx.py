import torch
import torch.nn as nn

from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy

from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.zoo.am.decoder import PrecomputedCache

class BottleneckModelDecoder(AttentionModelDecoder):
    def __init__(self, *args, noise_std: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)

        D = self.embed_dim
        self.to_mu = nn.Linear(D, D, bias=True)

        # simplest: fixed std (hyperparameter) so you don't even learn sigma yet
        self.noise_std = noise_std

    def _precompute_cache(self, embeddings: torch.Tensor, num_starts: int = 0) -> PrecomputedCache:
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings).chunk(3, dim=-1)

        if self.use_graph_context:
            pooled = embeddings.mean(1)      # [B, D]
            mu = self.to_mu(pooled)          # [B, D]

            if self.training and self.noise_std > 0:
                z = mu + self.noise_std * torch.randn_like(mu)
            else:
                z = mu

            graph_context = self.project_fixed_context(z)  # [B, D]
        else:
            graph_context = 0

        return PrecomputedCache(
            node_embeddings=embeddings,
            graph_context=graph_context,
            glimpse_key=glimpse_key_fixed,
            glimpse_val=glimpse_val_fixed,
            logit_key=logit_key_fixed,
        )

def getBottleneckPolicy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    ):
    
    bottleneckDecoder = BottleneckModelDecoder(
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
        decoder=bottleneckDecoder,
        )
    
    return policy