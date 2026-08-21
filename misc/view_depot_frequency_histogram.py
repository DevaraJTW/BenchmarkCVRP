import torch
import matplotlib.pyplot as plt
import numpy as np

def depot_tourroute_length_histogram(logprobs_list, actions_list, num_customers=20):
    logp = torch.tensor(logprobs_list)
    actions = torch.tensor(actions_list)

    depot_prob = logp[:, :, 0].exp()

    customer_action = (actions != 0).int()
    served_before = torch.cumsum(customer_action, dim=0) - customer_action
    valid = served_before < num_customers

    route_lengths = []
    route_len = torch.zeros(actions.size(1), dtype=torch.long)
    for step in range(actions.size(0)):
        for inst in range(actions.size(1)):
            if not valid[step, inst]:
                continue

            a = actions[step, inst].item()

            if a == 0:
                route_lengths.append(route_len[inst].item())
                route_len[inst] = 0
            else:
                route_len[inst] += 1

    return route_lengths

testProblemSize = "100"

dataM20 = torch.load(
    "misc/depot_probs_acts_M20_TS"+testProblemSize+".pt",
    weights_only=False
)

dataM100 = torch.load(
    "misc/depot_probs_acts_M100_TS"+testProblemSize+".pt",
    weights_only=False
)

curve_train20 = depot_tourroute_length_histogram(
    dataM20["log_probs"],
    dataM20["actions"],
)

curve_train100 = depot_tourroute_length_histogram(
    dataM100["log_probs"],
    dataM100["actions"],
)

max_len = 20
x = np.arange(1, max_len + 1)

counts20 = np.bincount(curve_train20, minlength=max_len + 1)[1:max_len+1]
counts100 = np.bincount(curve_train100, minlength=max_len + 1)[1:max_len+1]

perc20 = counts20 / counts20.sum() * 100
perc100 = counts100 / counts100.sum() * 100

width = 0.4

plt.figure()

plt.bar(
    x - width/2,
    perc20,
    width=width,
    label="Train20"
)

plt.bar(
    x + width/2,
    perc100,
    width=width,
    label="Train100"
)


plt.xlabel("Customers served per tour")
plt.ylabel("Count")
plt.title("Route length before depot return tested on CVRP"+testProblemSize)
plt.xticks(x)
plt.legend()
plt.savefig("depot_tour_length_CVRP"+testProblemSize+".svg", format="svg")
plt.show()


