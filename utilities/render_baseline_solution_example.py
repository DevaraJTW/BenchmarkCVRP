import torch
import matplotlib.pyplot as plt
import sys

from rl4co.envs import CVRPEnv
import rl4co.envs.routing.cvrp.render as cvrp_render

from utilities.get_model_file import getModelFile
from models.architectures import (
    architectureList,
    getPolicyByModelArchitecture
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

modelArchitecture = (
    architectureList[0]
    if len(sys.argv) < 2
    else architectureList[int(sys.argv[1])]
)

numOfLocations = (
    20 if len(sys.argv) < 3
    else int(sys.argv[2])
)

instance_index = (
    0 if len(sys.argv) < 4
    else int(sys.argv[3])
)

modelCVRPNumber = 20

valFilePath = "data/CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_val.npz"
testFilePath = "data/CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_test.npz"


# --------------------------------------------------
# Environment
# --------------------------------------------------

env = CVRPEnv(
    generator_params={
        "num_loc": numOfLocations,
        "capacity": 30,
        "demand_min": 1,
        "demand_max": 9,
    },
    val_file=valFilePath,
    test_file=testFilePath,
)


# --------------------------------------------------
# Initialize policy
# --------------------------------------------------

embeddingDimension = 128
numEncoderLayers = 3
numberOfHeads = 8

policy = getPolicyByModelArchitecture(
    modelArchitecture,
    env,
    embeddingDimension,
    numEncoderLayers,
    numberOfHeads
)


# --------------------------------------------------
# Load trained policy weights
# --------------------------------------------------

modelFilePath = getModelFile(
    modelCVRPNumber,
    modelArchitecture
)

checkpoint = torch.load(
    modelFilePath,
    map_location="cpu",
    weights_only=False
)

policy_state_dict = {
    key.removeprefix("policy."): value
    for key, value in checkpoint["state_dict"].items()
    if key.startswith("policy.")
}

if not policy_state_dict:
    raise ValueError(
        "No policy weights found in checkpoint. "
        "Check the checkpoint state_dict keys."
    )

policy.load_state_dict(
    policy_state_dict,
    strict=True
)

policy = policy.to(device)
policy.eval()


# --------------------------------------------------
# Load one test instance
# --------------------------------------------------

dataset = env.load_data(testFilePath)

if not 0 <= instance_index < len(dataset):
    raise IndexError(
        f"Instance index must be between "
        f"0 and {len(dataset) - 1}"
    )

# Keep batch dimension
instance = dataset[
    instance_index:instance_index + 1
].to(device)

td = env.reset(instance)


# --------------------------------------------------
# Run inference
# --------------------------------------------------

with torch.inference_mode():

    output = policy(
        td.clone(),
        env,
        phase="test",
        decode_type="greedy",
        return_actions=True
    )

actions = output["actions"][0].detach().cpu()
cost = -output["reward"][0].item()


# --------------------------------------------------
# Plot comparison
# --------------------------------------------------

plt.rcParams.update({
    "font.size": 14
})

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(12, 5)
)


# Unsolved instance
depot = td["depot"][0].detach().cpu().numpy()
locs = td["locs"][0].detach().cpu().numpy()

ax1.scatter(
    locs[:, 0],
    locs[:, 1],
    s=20,
    color="tab:blue",
    label="Customers",
    zorder=2
)

ax1.scatter(
    depot[0],
    depot[1],
    s=100,
    marker="s",
    color="tab:orange",
    edgecolor="black",
    label="Depot",
    zorder=5
)

ax1.set_title("Unsolved Instance")


# Solved instance
cvrp_render.render(
    td[0].cpu(),
    actions,
    ax=ax2,
    skip_depot=False
)

ax2.set_title(
    f"Baseline Solution (Cost: {cost:.3f})"
)


# --------------------------------------------------
# Formatting
# --------------------------------------------------

for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.15)

plt.tight_layout()


# --------------------------------------------------
# Save and show
# --------------------------------------------------

output_path = (
    f"CVRP{numOfLocations}_"
    f"BaselineCVRP{modelCVRPNumber}_"
    f"instance{instance_index}.svg"
)

plt.savefig(
    output_path,
    bbox_inches="tight",
    format="svg",
)

plt.show()

print(f"Architecture: {modelArchitecture}")
print(f"Train size: CVRP{modelCVRPNumber}")
print(f"Test size: CVRP{numOfLocations}")
print(f"Instance index: {instance_index}")
print(f"Tour length: {cost:.4f}")
print(f"Saved to: {output_path}")