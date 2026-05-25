import torch
import matplotlib.pyplot as plt

from rl4co.envs import CVRPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.utils.trainer import RL4COTrainer

from tensordict.tensordict import TensorDict
import rl4co.envs.routing.cvrp.render as cvrp_render

from models.complete_context_embedding import AggregateVRPContext
from models.bottleneck_graph_ctx import BottleneckModelDecoder
from models.set2set_graph_ctx import Set2SetModelDecoder

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

_orig_render = cvrp_render.render

@staticmethod
def render_with_depot(td: TensorDict, actions: torch.Tensor = None, ax=None):
    return _orig_render(td, actions, ax, skip_depot=False)

CVRPEnv.render = render_with_depot

locationQuantity = 50
#checkpointPath = "checkpoints/lightning_logs/CVRP50E/checkpoints/epoch=149-step=29400.ckpt"
checkpointPath = "checkpoints/lightning_logs/CVRP50S2S/checkpoints/epoch=149-step=29400.ckpt"
#checkpointPath = "checkpoints/lightning_logs/CVRP10S2S/checkpoints/epoch=249-step=49000.ckpt"
#checkpointPath = "checkpoints/lightning_logs/CVRP100S2S/checkpoints/epoch=149-step=29400.ckpt"
#checkpointPath = "checkpoints/lightning_logs/CVRP10BN/checkpoints/epoch=249-step=49000.ckpt"
#checkpointPath = "checkpoints/lightning_logs/CVRP100BN/checkpoints/epoch=149-step=29400.ckpt"

# RL4CO env based on TorchRL
env = CVRPEnv(generator_params={
    "num_loc": locationQuantity,        # customers
    "capacity": 30,        # vehicle capacity (must be consistent with demand scale)
    "demand_min": 1,       # optional (if supported by your RL4CO version)
    "demand_max": 9,       # optional
    "seed": 123
})

embeddingDimension = 128
numberOfHeads = 8

bottleneckDecoder = BottleneckModelDecoder(
    embed_dim=embeddingDimension,
    num_heads=numberOfHeads,
    env_name=env.name,
    mask_inner=True,
    out_bias_pointer_attn=False,
    linear_bias=False,
    use_graph_context=True,
    check_nan=True,
)

set2setDecoder = Set2SetModelDecoder(
    embed_dim=embeddingDimension,
    num_heads=numberOfHeads,
    env_name=env.name,
    mask_inner=True,
    out_bias_pointer_attn=False,
    linear_bias=False,
    use_graph_context=True,
    check_nan=True,
)

# Policy: neural network, in this case with encoder-decoder architecture
embeddingDimension = 128
policy = AttentionModelPolicy(env_name=env.name, 
                              embed_dim=embeddingDimension,
                              num_encoder_layers=3,
                              num_heads=numberOfHeads,
                              decoder=set2setDecoder,
                              #context_embedding=AggregateVRPContext(embed_dim=embeddingDimension),
                            )

# RL Model: REINFORCE and greedy rollout baseline
model = REINFORCE(env,
                    policy,
                    baseline="rollout",
                    batch_size=512,
                    train_data_size=100_000,
                    val_data_size=10_000,
                    optimizer_kwargs={"lr": 1e-4},
                    )

# Greedy rollouts over untrained policy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
td_init = env.reset(batch_size=[3]).to(device)
policy = policy.to(device)
policy.eval()
out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
actions_untrained = out['actions'].cpu().detach()
rewards_untrained = out['reward'].cpu().detach()

#for i in range(3):
#    print(f"Problem {i+1} | Cost: {-rewards_untrained[i]:.3f}")
#    env.render(td_init[i], actions_untrained[i])

trainer = RL4COTrainer(
    max_epochs=200,
    accelerator="gpu",
    devices=1,
    logger=None,
    default_root_dir="checkpoints/",
)

trainer.test(model, ckpt_path=checkpointPath, weights_only=False)

# Greedy rollouts over trained model (same states as previous plot)
policy = model.policy.to(device)
out = policy(td_init.clone(), phase="test", decode_type="greedy", return_actions=True)
actions_trained = out['actions'].cpu().detach()

# Get average
costAvg = 0
for i, td in enumerate(td_init):
    costAvg += -out['reward'][i].item()

print("Cost average is:", costAvg/len(td_init))
#quit()

# Plotting
for i, td in enumerate(td_init):
    fig, axs = plt.subplots(1,2, figsize=(11,5))
    env.render(td, actions_untrained[i], ax=axs[0]) 
    env.render(td, actions_trained[i], ax=axs[1])
    axs[0].set_title(f"Untrained | Cost = {-rewards_untrained[i].item():.3f}")
    axs[1].set_title(r"Trained $\pi_\theta$" + f"| Cost = {-out['reward'][i].item():.3f}")
    plt.show()











