import torch
import matplotlib.pyplot as plt
import sys

from rl4co.envs import CVRPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.utils.trainer import RL4COTrainer

from tensordict.tensordict import TensorDict
import rl4co.envs.routing.cvrp.render as cvrp_render
from lightning.pytorch.callbacks import Timer
from utilities.timer_logger import TimeLogger

from models.architectures import architectureList, getPolicyByModelArchitecture

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

numOfLocations = 20 if len(sys.argv) < 3 else int(sys.argv[2])
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

modelArchitecture = architectureList[0] if len(sys.argv) < 2 else architectureList[int(sys.argv[1])]
embeddingDimension = 128
numEncoderLayers = 3
numberOfHeads = 8

policy = getPolicyByModelArchitecture(modelArchitecture, env, embeddingDimension, numEncoderLayers, numberOfHeads)

model = REINFORCE(
    env,
    policy,
    baseline="rollout",
    batch_size=512,
    train_data_size=100_000,
    optimizer_kwargs={"lr": 1e-4},
)

trainer = RL4COTrainer(
    max_epochs=1,
    accelerator="gpu",
    devices=1,
    logger=None,
    callbacks=[builtin_timer, custom_timer],
    default_root_dir="trained_models/"+str(modelArchitecture)+"/CVRP"+str(numOfLocations)+"/",
)

trainer.fit(
    model,
    )

