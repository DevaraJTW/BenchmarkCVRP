import torch
import matplotlib.pyplot as plt

from rl4co.envs import CVRPEnv

from tensordict.tensordict import TensorDict
import rl4co.envs.routing.cvrp.render as cvrp_render

from rl4co.data.utils import save_tensordict_to_npz

isTest = True
theSeed = 0

if isTest:
    theSeed = 1234
else:
    theSeed = 4321
    
#torch.manual_seed(theSeed)
#torch.cuda.manual_seed_all(theSeed)

_orig_render = cvrp_render.render

@staticmethod
def render_with_depot(td: TensorDict, actions: torch.Tensor = None, ax=None):
    return _orig_render(td, actions, ax, skip_depot=False)

CVRPEnv.render = render_with_depot

locationQuantity = 20

# RL4CO env based on TorchRL
env = CVRPEnv(
    generator_params={
        "num_loc": locationQuantity,        # customers
        "capacity": 30,        # vehicle capacity (must be consistent with demand scale)
        "demand_min": 1,       # optional (if supported by your RL4CO version)
        "demand_max": 9,       # optional
        },
    seed = theSeed,
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if isTest:
    tdTest = env.generator(10000)
    tdTest.set("demand", tdTest["demand"] * tdTest["capacity"])
    tdTest.set("capacity", tdTest["capacity"].squeeze(-1))
    save_tensordict_to_npz(tdTest, "cvrp"+str(locationQuantity)+"_test.npz")
    
else:
    tdVal = env.generator(10000)
    tdVal.set("demand", tdVal["demand"] * tdVal["capacity"])
    tdVal.set("capacity", tdVal["capacity"].squeeze(-1))
    save_tensordict_to_npz(tdVal, "cvrp"+str(locationQuantity)+"_val.npz")
    
"""
TensorDict(
    fields={
        capacity: Tensor(shape=torch.Size([100000, 1]), device=cpu, dtype=torch.int64, is_shared=False),
        demand: Tensor(shape=torch.Size([100000, 20]), device=cpu, dtype=torch.float32, is_shared=False),
        depot: Tensor(shape=torch.Size([100000, 2]), device=cpu, dtype=torch.float32, is_shared=False),
        locs: Tensor(shape=torch.Size([100000, 20, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
    batch_size=torch.Size([100000]),
    device=None,
    is_shared=False)
"""