import torch
import matplotlib.pyplot as plt

def select_depot_rate_curve(logprobs_list, actions_list, num_customers=20):
    logp = torch.tensor(logprobs_list)
    actions = torch.tensor(actions_list)

    depot_prob = logp[:, :, 0].exp()

    customer_action = (actions != 0).int()
    served_before = torch.cumsum(customer_action, dim=0) - customer_action
    valid = served_before < num_customers
    selected_depot = (actions == 0)

    curve = []
    for step in range(actions.size(0)):
        vals = selected_depot[step][valid[step]].float()
        curve.append(vals.mean().item())

    return curve

dataM20 = torch.load(
    "misc/depot_probs_acts_M20_TS20.pt",
    weights_only=False
)

dataM100 = torch.load(
    "misc/depot_probs_acts_M100_TS20.pt",
    weights_only=False
)

curve_train20 = select_depot_rate_curve(
    dataM20["log_probs"],
    dataM20["actions"],
)

curve_train100 = select_depot_rate_curve(
    dataM100["log_probs"],
    dataM100["actions"],
)

plt.figure()
plt.plot(range(len(curve_train20)), curve_train20, marker="o", label="Trained on CVRP20")
plt.plot(range(len(curve_train100)), curve_train100, marker="o", label="Trained on CVRP100")
plt.xlabel("Decoding step")
plt.ylabel("Mean depot probability")
plt.title("Depot selection rate over steps tested on CVRP20")
plt.legend()
plt.savefig("depot_select_rate_CVRP.svg", format="svg")
plt.show()



