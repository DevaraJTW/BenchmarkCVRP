import torch

def calculateVariance(n_train):
    gc_20 = torch.load("graph_context_CVRP"+n_train+"_on_20.pt")
    gc_50 = torch.load("graph_context_CVRP"+n_train+"_on_50.pt")
    gc_100 = torch.load("graph_context_CVRP"+n_train+"_on_100.pt")

    variance = [
        gc_20.var(dim=0, unbiased=False).mean(),
        gc_50.var(dim=0, unbiased=False).mean(),
        gc_100.var(dim=0, unbiased=False).mean(),
    ]

    return variance

print(calculateVariance("20"))
print(calculateVariance("50"))
print(calculateVariance("100"))




