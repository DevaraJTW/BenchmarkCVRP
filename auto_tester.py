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

def TestModel(model_index, train_problem_size, test_problem_size):

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

    policy = getPolicyByModelArchitecture(modelArchitecture, env, embeddingDimension, numEncoderLayers, numberOfHeads, train_problem_size, test_problem_size)

    modelCVRPNumber = train_problem_size
    modelFilePath = getModelFile(modelCVRPNumber, modelArchitecture)

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
        default_root_dir="trained_models/"+str(modelArchitecture)+"/CVRP"+str(numOfLocations)+"/test/",
    )

    results = trainer.test(
        model,
        ckpt_path=modelFilePath,
        weights_only=False,
        )

    return results[0]["test/reward"]

def main():
    trainingList = {
        #0: [20,50,100],
        #1: [20,50,100],
        #2: [20,50,100],
        #3: [20,50,100],
        #4: [20,50,100],
        #5: [20,50,100],
        #6: [20,50,100],
        #7: [20,50,100],
        #8: [20,50,100],
        #9: [20,50,100],
        10: [20,50,100],
        #11: [20,50,100],
    }

    results = {}
    for enumArch, numLocList in trainingList.items():
        for numLoc in numLocList:
            results[architectureList[enumArch]+"CVRP"+str(numLoc)] = [
                TestModel(enumArch, numLoc, 20),
                TestModel(enumArch, numLoc, 50),
                TestModel(enumArch, numLoc, 100),
            ]

    print(results)

if __name__ == "__main__":
    main()



