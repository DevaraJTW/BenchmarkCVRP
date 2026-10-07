import torch
import matplotlib.pyplot as plt

def select_depot_rate_curve(logprobs_list, actions_list, num_customers):
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

def drawDepotSelectionRate(ax, test_size, first):
    dataM20 = torch.load(
        "misc/depot_probs_acts_M20_TS"+str(test_size)+".pt",
        weights_only=False
    )

    dataM100 = torch.load(
        "misc/depot_probs_acts_M100_TS"+str(test_size)+".pt",
        weights_only=False
    )

    curve_train20 = select_depot_rate_curve(
        dataM20["log_probs"],
        dataM20["actions"],
        test_size,
    )

    curve_train100 = select_depot_rate_curve(
        dataM100["log_probs"],
        dataM100["actions"],
        test_size,
    )

    ax.grid()
    ax.set_xlim(0, 130)
    markersize = 3
    ax.plot(range(len(curve_train20)), curve_train20, marker="o", markersize=markersize, label="Trained on CVRP20")
    ax.plot(range(len(curve_train100)), curve_train100, marker="o", markersize=markersize, label="Trained on CVRP100")
    if first:
        ax.legend()


fig, axes = plt.subplots(2, 1, figsize=(9, 5))
nList = [20, 100]
first = True
for i, ax in enumerate(axes):
    drawDepotSelectionRate(ax, nList[i], first)
    first = False

plt.rcParams.update({"font.size": 14})
plt.figtext(0.002, 0.5, "Depot selection rate", va="center", rotation="vertical", fontsize=14)
plt.figtext(0.5, 0.03, "Steps", ha="center", fontsize=14)
plt.subplots_adjust(
    left=0.06,
    bottom=0.11,
    right=0.995,
    top=0.99,
    hspace=0.15,
)
#plt.legend()
fig.savefig("depot_selection_rate.svg", format="svg")
plt.show()

