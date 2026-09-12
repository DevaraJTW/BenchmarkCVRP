import sys
import torch
from rl4co.envs import CVRPEnv

numOfLocations = 20
env20 = CVRPEnv(generator_params={"num_loc": numOfLocations, "capacity": 30, "demand_min": 1, "demand_max": 9, }, val_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_val.npz", test_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_test.npz")
dataset20 = env20.dataset(phase="test")

numOfLocations = 50
env50 = CVRPEnv(generator_params={"num_loc": numOfLocations, "capacity": 30, "demand_min": 1, "demand_max": 9, }, val_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_val.npz", test_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_test.npz")
dataset50 = env50.dataset(phase="test")

numOfLocations = 100
env100 = CVRPEnv(generator_params={"num_loc": numOfLocations, "capacity": 30, "demand_min": 1, "demand_max": 9, }, val_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_val.npz", test_file = "CVRP"+str(numOfLocations)+"/cvrp"+str(numOfLocations)+"_test.npz")
dataset100 = env100.dataset(phase="test")

def getMeanNearestNeighborDistance(instance):
    locs = instance["locs"]

    distances = torch.cdist(locs, locs)
    distances.fill_diagonal_(float("inf"))
    nearestDistances = distances.min(dim=1).values
    meanNearestDistance = nearestDistances.mean().item()

    return meanNearestDistance

def getMeanDepotDistance(instance):
    locs = instance["locs"]      # [n, 2]
    depot = instance["depot"]    # [2]

    distances = torch.norm(locs - depot, dim=1)

    return distances.mean().item()

def getMeanPairwiseDistance(instance):
    locs = instance["locs"]   # [n, 2]

    distances = torch.cdist(locs, locs)   # [n, n]

    indices = torch.triu_indices(
        distances.size(0),
        distances.size(1),
        offset=1
    )

    pairwiseDistances = distances[indices[0], indices[1]]

    return pairwiseDistances.mean().item()

def getMetrics(dataset, metricFunction):
    values = []

    for instance in dataset:
        values.append(metricFunction(instance))

    values = torch.tensor(values)

    mean = values.mean().item()
    std = values.std(correction=0).item()

    return mean, std

print(
    getMetrics(dataset20, getMeanNearestNeighborDistance),
    getMetrics(dataset20, getMeanDepotDistance),
    getMetrics(dataset20, getMeanPairwiseDistance),
    )
print(
    getMetrics(dataset50, getMeanNearestNeighborDistance),
    getMetrics(dataset50, getMeanDepotDistance),
    getMetrics(dataset50, getMeanPairwiseDistance),
    )
print(
    getMetrics(dataset100, getMeanNearestNeighborDistance),
    getMetrics(dataset100, getMeanDepotDistance),
    getMetrics(dataset100, getMeanPairwiseDistance),
    )







