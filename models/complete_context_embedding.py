import torch

from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy
from rl4co.models.nn.env_embeddings.context import EnvContext

class AggregateVRPContext(EnvContext):
    """Context embedding for the Capacitated Vehicle Routing Problem (CVRP).
    Project the following to the embedding space:
        - current node embedding
        - remaining capacity (vehicle_capacity - used_capacity)
    """

    def __init__(self, embed_dim):
        super(AggregateVRPContext, self).__init__(
            embed_dim=embed_dim, step_context_dim=embed_dim + 1
        )
        
    def _cur_node_embedding(self, embeddings, td):
        """Get embedding of current node"""
        cur_node_embedding = self.aggregate_embedding(embeddings, td)
        return cur_node_embedding

    def _state_embedding(self, embeddings, td):
        state_embedding = td["vehicle_capacity"] - td["used_capacity"]
        return state_embedding
    
    def aggregate_embedding(self, embeddings, td) -> torch.Tensor:
        mask = td["action_mask"].float()                # [B, N]
        mask = mask.unsqueeze(-1)                       # [B, N, 1]
        maskedSum = (embeddings * mask).sum(dim=1)     # sum([B, N, E] * [B, N, 1]), along N dimension) = sum([B, N, E]) = [B, E]
        numOfUnmasked = mask.sum(dim=1)
        
        aggEmbedding = maskedSum / numOfUnmasked
        
        return aggEmbedding
    
def getAggregatePolicy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    ):
    
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=embeddingDimension,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        context_embedding=AggregateVRPContext(embed_dim=embeddingDimension),
        )
    
    return policy
    