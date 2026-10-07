import torch
import matplotlib.pyplot as plt

def attention_dispersion(logprobs_list, actions_list, num_customers=20):
    logp = torch.tensor(logprobs_list)
    actions = torch.tensor(actions_list)

    print(logp.shape)
    print(actions.shape)


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

    # entropy shape:
    # [steps, instances]

    print("entropy shape:", entropy.shape)

    mean_entropy_per_step = []
    for step in range(logp.shape[0]):
        step_entropy = entropy[step][active[step]]
        mean_entropy = step_entropy.mean().item()
        mean_entropy_per_step.append(mean_entropy)

    return mean_entropy_per_step

def createPlotByN(subplot_ax, train_size, render_legend):
    dataTS20 = torch.load(
        "misc/attention_data_M"+str(train_size)+"_TS20.pt",
        weights_only=False
    )

    entropyTS20 = attention_dispersion(
        dataTS20["logprobs"],
        dataTS20["actions"],
        20
    )

    dataTS50 = torch.load(
        "misc/attention_data_M"+str(train_size)+"_TS50.pt",
        weights_only=False
    )

    entropyTS50 = attention_dispersion(
        dataTS50["logprobs"],
        dataTS50["actions"],
        50
    )

    dataTS100 = torch.load(
        "misc/attention_data_M"+str(train_size)+"_TS100.pt",
        weights_only=False
    )

    entropyTS100 = attention_dispersion(
        dataTS100["logprobs"],
        dataTS100["actions"],
        100
    )

    markersize = 1.8
    linewidth = 1
    subplot_ax.plot(range(len(entropyTS20)), entropyTS20, linewidth=linewidth, marker="o", markersize=markersize, label="Tested on CVRP20", color="brown")
    subplot_ax.plot(range(len(entropyTS50)), entropyTS50, linewidth=linewidth, marker="o", markersize=markersize, label="Tested on CVRP50", color="orange")
    subplot_ax.plot(range(len(entropyTS100)), entropyTS100, linewidth=linewidth, marker="o", markersize=markersize, label="Tested on CVRP100", color="blue")
    subplot_ax.grid()
    if render_legend:
        subplot_ax.legend()

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3))
nList = [20, 50, 100]
for i, ax in enumerate(axes):
    render_legend = False
    if i == len(nList) - 1:
        render_legend = True
    createPlotByN(ax, nList[i], render_legend)

plt.rcParams.update({"font.size": 14})
plt.figtext(0.002, 0.5, "Mean entropy", va="center", rotation="vertical", fontsize=14)
plt.figtext(0.5, 0.03, "Steps", ha="center", fontsize=14)
plt.subplots_adjust(
    left=0.055,
    bottom=0.155,
    right=0.99,
    top=0.92,
    wspace=0.125
)
#plt.legend()
plt.show()
fig.savefig("attentionDispersion.svg", format="svg")

