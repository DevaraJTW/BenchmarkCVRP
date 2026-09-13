import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def attention_dispersion(logprobs_list, actions_list, num_customers=20):
    logp = torch.tensor(logprobs_list)
    actions = torch.tensor(actions_list)

    customer_selected = actions != 0
    customers_visited = customer_selected.cumsum(dim=0)

    active = torch.cat(
        [
            torch.ones(
                (1, actions.shape[1]),
                dtype=torch.bool
            ),
            customers_visited[:-1] < num_customers
        ],
        dim=0
    )

    probs = torch.exp(logp)

    entropy = -(
        probs * logp
    ).nan_to_num(0.0).sum(dim=-1)

    # --------------------------------
    # Mean entropy PER INSTANCE
    # --------------------------------

    active_float = active.float()

    entropy_per_instance = (
        (entropy * active_float).sum(dim=0)
        / active_float.sum(dim=0)
    )

    # [10000]
    print("entropy per instance:", entropy_per_instance.shape)

    # --------------------------------
    # Overall mean across instances
    # --------------------------------

    overall_mean_entropy = entropy_per_instance.mean().item()
    overall_std_entropy = entropy_per_instance.std().item()

    return overall_mean_entropy, overall_std_entropy

def loadAndRunAttentionDispersion(train_size, test_size):
    dataM = torch.load(
        "misc/attention_data_M"+str(train_size)+"_TS"+str(test_size)+".pt",
        weights_only=False
    )

    mean, std = attention_dispersion(
        dataM["logprobs"],
        dataM["actions"],
        test_size
    )

    return mean, std

def loadByTestSize(test_size):
    result = [
        loadAndRunAttentionDispersion(20, test_size)[0],
        loadAndRunAttentionDispersion(50, test_size)[0],
        loadAndRunAttentionDispersion(100, test_size)[0],
    ]

    return result



cvrp20 = loadByTestSize(20)
cvrp50 = loadByTestSize(50)
cvrp100 = loadByTestSize(100)

print(cvrp20)
print(cvrp50)
print(cvrp100)

# Positions with gaps between test-size groups
x20  = [0, 1, 2]
x50  = [4, 5, 6]
x100 = [8, 9, 10]

plt.figure(figsize=(9, 5))
plt.rcParams.update({"font.size": 14})
colors = ["tab:blue", "tab:orange", "tab:grey"]
plt.bar(x20, cvrp20, color=colors)
plt.bar(x50, cvrp50, color=colors)
plt.bar(x100, cvrp100, color=colors)
# Label every individual bar by training size
#plt.xticks(
#    x20 + x50 + x100,
#    [
#        "Train20", "Train50", "Train100",
#        "Train20", "Train50", "Train100",
#        "Train20", "Train50", "Train100"
#    ],
#    fontsize=11
#)
plt.xticks(
    [1, 5, 9],
    ["Test20", "Test50", "Test100"],
)
plt.ylabel("Average Entropy")
legend_elements = [
    Patch(facecolor=colors[0], label="Train20"),
    Patch(facecolor=colors[1], label="Train50"),
    Patch(facecolor=colors[2], label="Train100"),
]
plt.legend(handles=legend_elements, fontsize=11)
plt.title("Average entropy over all steps of 10,000 instances")
plt.tight_layout()
plt.savefig("average_entropy.svg", format="svg")
plt.show()
