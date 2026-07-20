import numpy as np

data = np.load("data/CVRP20/cvrp20_test.npz")
print(data.files)
print(data)
print("locs", data["locs"])
print("locs len", len(data["locs"]))
print("depot", data["depot"])
print("depot len", len(data["depot"]))
print("demand", data["demand"])
print("demand len", len(data["demand"]))
print("capacity", data["capacity"])
print("capacity len", len(data["capacity"]))
