from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy

from models.weighted_average import WeightedAverageModelDecoder
from models.attention_sage_encoder import AttentionSAGEEncoder
from models.learnable_scaling_factor import LearnableSDPAScalingFactor

def getCombine01Policy(
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

    weightedAverageDecoder = WeightedAverageModelDecoder(
        embed_dim=embeddingDimension,
        num_heads=num_heads,
        env_name=env.name,
        )

    learnable_sdpa_fn = LearnableSDPAScalingFactor()
    
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=embeddingDimension,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        encoder=sageEncoder,
        decoder=weightedAverageDecoder,
        sdpa_fn=learnable_sdpa_fn,
        )
    
    return policy
