from typing import Callable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensordict import TensorDict
from torch import Tensor

try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    SAGEConv = None
    
from models.sage_encoder import SAGEEncoder
    
from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.utils.ops import get_full_graph_edge_index
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

EdgeIndexFnSignature = Callable[[TensorDict, int], Tensor]

def edge_idx_fn_wrapper(td: TensorDict, num_nodes: int):
    return get_full_graph_edge_index(num_nodes, self_loop=False).to(td.device)


class AttentionSAGEEncoder(nn.Module):

    def __init__(
        self,
        env_name: str,
        embed_dim: int,
        num_layers: int,
        num_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        encoder_network: nn.Module = None,
        init_embedding: nn.Module = None,
    ):
        super().__init__()

        self.attention = AttentionModelEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            env_name=env_name,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            net=encoder_network,
            init_embedding=init_embedding,
        )
        
        self.sage = SAGEEncoder(
            env_name=env_name,
            embed_dim=embed_dim,
        )
        
        self.alpha_logit = nn.Parameter(torch.tensor(2.0))
        
    def forward(
        self, td: TensorDict, mask: Tensor | None = None
    ) -> Tuple[Tensor, Tensor]:
        
        hidden_Attention, init_embeds_Attention = self.attention(td)
        hidden_SAGE, init_embeds_SAGE = self.sage(td)
        
        alpha = torch.sigmoid(self.alpha_logit)
        #print("AttentionSAGEEncoder alpha is:", alpha)
        
        hidden = alpha * hidden_Attention + (1.0 - alpha) * hidden_SAGE
        init_embeds = alpha * init_embeds_Attention + (1.0 - alpha) * init_embeds_SAGE

        return hidden, init_embeds

def getAttentionSAGEEncoderPolicy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    ):
    
    customEncoder = AttentionSAGEEncoder(
        env_name=env.name,
        embed_dim=embeddingDimension,
        num_layers=num_encoder_layers,
    )
    
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=embeddingDimension,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        encoder=customEncoder,
        )
    
    return policy