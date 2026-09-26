import torch
import matplotlib.pyplot as plt
import sys
import os

from rl4co.envs import CVRPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.utils.trainer import RL4COTrainer

from tensordict.tensordict import TensorDict
import rl4co.envs.routing.cvrp.render as cvrp_render
import lightning as L
from lightning.pytorch.callbacks import Timer
from utilities.timer_logger import TimeLogger

from models.architectures import architectureList, getPolicyByModelArchitecture

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader

from utilities.get_model_file import getModelFile

from lightning.pytorch.loggers import CSVLogger

#seed = 6262
#seed = 1098
#seed = 8765
#L.seed_everything(seed, workers=True)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

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

num_epochs = 150 if len(sys.argv) < 4 else int(sys.argv[3])

csv_logger = CSVLogger(
    save_dir="mix_trained_models/"+str(modelArchitecture)+"/",
    name="logs",
    version="epoch_"+str(num_epochs),
)

trainer = RL4COTrainer(
    max_epochs=num_epochs,
    accelerator="gpu",
    devices=1,
    logger=csv_logger,
    callbacks=[builtin_timer, custom_timer],
    enable_checkpointing = False,
    default_root_dir="mix_trained_models/"+str(modelArchitecture)+"/",
)

checkpoint_path = "mix_trained_models/"+str(modelArchitecture)+"/logs/mixed_training/checkpoints/model_checkpoint.ckpt"
prev_checkpoint_path = checkpoint_path
if num_epochs == 75:
    print("Beginning of training, training without checkpoint.")
    prev_checkpoint_path = None

trainer.fit(
    model,
    ckpt_path=prev_checkpoint_path,
    )
trainer.save_checkpoint(checkpoint_path)
