import sys
import torch
from rl4co.envs import CVRPEnv

# test_dataset[0].keys() => dict_keys(['locs', 'depot', 'demand', 'capacity'])

numOfLocations = 20
env20 = CVRPEnv(generator_params={"num_loc": numOfLocations, "capacity": 30, "demand_min": 1, "demand_max": 9, }, val_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_val.npz", test_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_test.npz")
dataset20 = env20.dataset(phase="test")

numOfLocations = 50
env50 = CVRPEnv(generator_params={"num_loc": numOfLocations, "capacity": 30, "demand_min": 1, "demand_max": 9, }, val_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_val.npz", test_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_test.npz")
dataset50 = env50.dataset(phase="test")

numOfLocations = 100
env100 = CVRPEnv(generator_params={"num_loc": numOfLocations, "capacity": 30, "demand_min": 1, "demand_max": 9, }, val_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_val.npz", test_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_test.npz")
dataset100 = env100.dataset(phase="test")

def getTotalDemand(dataset):
    totalDemand = []
    for i in dataset:
        totalDemand.append(i["demand"].sum().item())

    totalDemand = torch.tensor(totalDemand)

    return totalDemand.mean().item()

def getMeanDemand(dataset):
    meanDemand = []
    for i in dataset:
        meanDemand.append(i["demand"].mean().item())

    meanDemand = torch.tensor(meanDemand)

    return meanDemand.mean().item(), meanDemand.std().item(), meanDemand.var().item()

print(getMeanDemand(dataset20), getTotalDemand(dataset20))
print(getMeanDemand(dataset50), getTotalDemand(dataset50))
print(getMeanDemand(dataset100), getTotalDemand(dataset100))



