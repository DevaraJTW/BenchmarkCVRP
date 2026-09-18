from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy

from models.set2set_graph_ctx import Set2SetModelDecoder
from models.attention_sage_encoder import AttentionSAGEEncoder
from models.learnable_scaling_factor import LearnableSDPAScalingFactor

def getCombine00Policy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    ):

    sageEncoder = AttentionSAGEEncoder(
        env_name=env.name,
        embed_dim=embeddingDimension,
        num_layers=num_encoder_layers,
    )

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

    learnable_sdpa_fn = LearnableSDPAScalingFactor()
    
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=embeddingDimension,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        encoder=sageEncoder,
        decoder=set2setDecoder,
        sdpa_fn=learnable_sdpa_fn,
        )
    
    return policy
