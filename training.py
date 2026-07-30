import torch
import matplotlib.pyplot as plt

from rl4co.envs import CVRPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.utils.trainer import RL4COTrainer

from tensordict.tensordict import TensorDict
import rl4co.envs.routing.cvrp.render as cvrp_render
from lightning.pytorch.callbacks import Timer
from utilities.timer_logger import TimeLogger

from models.default_baseline import getBaselinePolicy
from models.complete_context_embedding import getAggregatePolicy
from models.set2set_graph_ctx import getSet2SetPolicy
from models.weighted_average import getWeightedAveragePolicy
from models.weighted_average_simple import getWeightedAverageSimplePolicy
from models.gru_graph_ctx import getGRUGraphCTXPolicy
from models.attention_sage_encoder import getAttentionSAGEEncoderPolicy
#from models.bottleneck_graph_ctx import getBottleneckPolicy

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader

_orig_render = cvrp_render.render

builtin_timer = Timer()
custom_timer = TimeLogger(builtin_timer)

@staticmethod
def render_with_depot(td: TensorDict, actions: torch.Tensor = None, ax=None):
    return _orig_render(td, actions, ax, skip_depot=False)

CVRPEnv.render = render_with_depot

numOfLocations = 20
valFilePath = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_val.npz"
testFilePath = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_test.npz"

env = CVRPEnv(
    generator_params={
        "num_loc": numOfLocations,
        "capacity": 30,
        "demand_min": 1,
        "demand_max": 9,
        },
    val_file = valFilePath,
    test_file = testFilePath,
)

embeddingDimension = 128
numEncoderLayers = 3
numberOfHeads = 8

architectureList = [
    "Baseline",                 # 0
    "Aggregate",                # 1
    "Set2Set",                  # 2
    "WeightedAverage",          # 3
    "WeightedAverageSimple",    # 4
    "GRUGraphCTX",              # 5
    "AttentionSAGEEncoder"      # 6
]
modelArchitecture = architectureList[0]

match modelArchitecture:
    case "Baseline":
        policy = getBaselinePolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
    case "Aggregate":
        policy = getAggregatePolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
    case "Set2Set":
        policy = getSet2SetPolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
    case "WeightedAverage":
        policy = getWeightedAveragePolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
    case "WeightedAverageSimple":
        policy = getWeightedAverageSimplePolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
    case "GRUGraphCTX":
        policy = getGRUGraphCTXPolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
    case "AttentionSAGEEncoder":
        policy = getAttentionSAGEEncoderPolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)

model = REINFORCE(
    env,
    policy,
    baseline="rollout",
    batch_size=512,
    train_data_size=100_000,
    optimizer_kwargs={"lr": 1e-4},
)

trainer = RL4COTrainer(
    max_epochs=3,
    accelerator="gpu",
    devices=1,
    logger=None,
    callbacks=[builtin_timer, custom_timer],
    default_root_dir=str(modelArchitecture)+"/CVRP"+str(numOfLocations)+"/",
)

trainer.fit(
    model,
    )

