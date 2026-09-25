import torch
import torch.nn as nn

from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModelPolicy
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.utils.pylogger import get_pylogger
from rl4co.models.nn.attention import MultiHeadCrossAttention

log = get_pylogger(__name__)

class MAB(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
    ):
        super().__init__()

        self.attention = MultiHeadCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.rff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, X, Y):
        attn_out = self.attention(
            q_input=X,
            kv_input=Y,
        )

        H = self.norm1(X + attn_out)
        out = self.norm2(
            H + self.rff(H)
        )

        return out

class ISAB(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_inducing_points: int = 16,
    ):
        super().__init__()

        self.num_inducing_points = num_inducing_points

        # I in the Set Transformer paper
        self.inducing_points = nn.Parameter(
            torch.randn(
                1,
                num_inducing_points,
                embed_dim,
            )
        )

        self.mab1 = MAB(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

        self.mab2 = MAB(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    def forward(self, X):
        batch_size = X.size(0)

        I = self.inducing_points.expand(batch_size,-1,-1)
        H = self.mab1(I, X)
        X = self.mab2(X, H)

        return X

class ISABEncoder(nn.Module):
    def __init__(
        self,
        env_name: str = "cvrp",
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        num_inducing_points: int = 16,
    ):
        super().__init__()

        self.init_embedding = env_init_embedding(
            env_name,
            {"embed_dim": embed_dim},
        )

        self.layers = nn.ModuleList([
            ISAB(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_inducing_points=num_inducing_points,
            )
            for _ in range(num_layers)
        ])

    def forward(self, td, mask=None):
        # Same initial embedding as normal AM
        init_h = self.init_embedding(td)

        # [B, N, D]
        h = init_h

        for layer in self.layers:
            h = layer(h)

        return h, init_h

def getSetTransformerEncoderPolicy(
    env: CVRPEnv, 
    embeddingDimension: int,
    num_encoder_layers: int,
    num_heads: int,
    ):
    
    customEncoder = ISABEncoder(
        env_name=env.name,
        embed_dim=embeddingDimension,
        num_layers=num_encoder_layers,
        num_heads=num_heads,
    )
    
    policy = AttentionModelPolicy(
        env_name=env.name, 
        embed_dim=embeddingDimension,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
        encoder=customEncoder,
        )
    
    return policy