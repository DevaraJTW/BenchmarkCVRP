import torch
import matplotlib.pyplot as plt
import sys

from rl4co.envs import CVRPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.models import AttentionModelPolicy, REINFORCE
from rl4co.utils.trainer import RL4COTrainer

from tensordict.tensordict import TensorDict
import rl4co.envs.routing.cvrp.render as cvrp_render

from utilities.get_model_file import getModelFile

from models.architectures import architectureList, getPolicyByModelArchitecture

from rl4co.data.utils import load_npz_to_tensordict
from rl4co.data.dataset import TensorDictDataset
from torch.utils.data import DataLoader

_orig_render = cvrp_render.render

@staticmethod
def render_with_depot(td: TensorDict, actions: torch.Tensor = None, ax=None):
    return _orig_render(td, actions, ax, skip_depot=False)

CVRPEnv.render = render_with_depot

def TestMixedModel(model_index, test_problem_size):

    numOfLocations = test_problem_size
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

    modelArchitecture = architectureList[model_index]
    embeddingDimension = 128
    numEncoderLayers = 3
    numberOfHeads = 8

    policy = getPolicyByModelArchitecture(modelArchitecture, env, embeddingDimension, numEncoderLayers, numberOfHeads, None, test_problem_size)

    modelFilePath = "mix_trained_models/"+str(modelArchitecture)+"/logs/mixed_training/checkpoints/model_checkpoint.ckpt"

    model = REINFORCE(
        env,
        policy,
        baseline="rollout",
        batch_size=512,
        train_data_size=100_000,
        optimizer_kwargs={"lr": 1e-4},
    )

    trainer = RL4COTrainer(
        max_epochs=150,
        accelerator="gpu",
        devices=1,
        logger=None,
        default_root_dir="OLD_mix_trained_models/"+str(modelArchitecture)+"/",
    )

    results = trainer.test(
        model,
        ckpt_path=modelFilePath,
        weights_only=False,
        )

    return results[0]["test/reward"]

def main():
    trainingList = {
        #0: None,
        #8: None,
        13:  None,
        14:  None,
        15:  None,
        16:  None,
        17:  None,
        18:  None,
        #1: [20,50,100],
        #2: [20,50,100],
        #3: [20,50,100],
        #4: [20,50,100],
        #5: [20,50,100],
        #6: [20,50,100],
        #7: [20,50,100],
        #8: [20,50,100],
    }

    results = {}
    for enumArch, _ in trainingList.items():
        results[architectureList[enumArch]] = [
            TestMixedModel(enumArch, 20),
            TestMixedModel(enumArch, 50),
            TestMixedModel(enumArch, 100),
        ]

    print(results)

if __name__ == "__main__":
    main()



