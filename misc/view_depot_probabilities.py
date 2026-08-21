import torch
import matplotlib.pyplot as plt

def mean_depot_prob_curve(logprobs_list, actions_list, num_customers=20):
    logp = torch.tensor(logprobs_list)
    actions = torch.tensor(actions_list)

    depot_prob = logp[:, :, 0].exp()

    customer_action = (actions != 0).int()
    served_before = torch.cumsum(customer_action, dim=0) - customer_action
    valid = served_before < num_customers

    curve = []
    for step in range(logp.size(0)):
        vals = depot_prob[step][valid[step]]
        curve.append(vals.mean().item())

    return curve



dataM20 = torch.load(
    "misc/depot_probs_acts_M20_TS100.pt",
    weights_only=False
)

dataM100 = torch.load(
    "misc/depot_probs_acts_M100_TS100.pt",
    weights_only=False
)

curve_train20 = mean_depot_prob_curve(
    dataM20["log_probs"],
    dataM20["actions"],
)

curve_train100 = mean_depot_prob_curve(
    dataM100["log_probs"],
    dataM100["actions"],
)

plt.figure()
plt.plot(range(len(curve_train20)), curve_train20, marker="o", label="Trained on CVRP20")
plt.plot(range(len(curve_train100)), curve_train100, marker="o", label="Trained on CVRP100")
plt.xlabel("Decoding step")
plt.ylabel("Mean depot probability")
plt.title("Depot probability over steps tested on CVRP100")
plt.legend()
plt.savefig("depot_probs_CVRP.svg", format="svg")
plt.show()



