import math
from torch import Tensor

from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy
from torch.nn.functional import scaled_dot_product_attention

def buildSDPAFn(train_size, test_size):

    esf = math.log(test_size, train_size)
    
    def scaled_dot_product_attention_esf(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Tensor | None = None,
        dropout_p: float = 0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False
    ) -> Tensor :

        scale = esf / math.sqrt(query.size(-1))
    
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

    return scaled_dot_product_attention_esf

def getBaselineESFPolicy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    train_size: int,
    test_size: int,
    ):
    
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=embeddingDimension,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        sdpa_fn=buildSDPAFn(train_size, test_size),
        )
    
    return policy
