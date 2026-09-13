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

train_size = 100

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

plt.figure()
plt.rcParams.update({"font.size": 14})
plt.plot(range(len(entropyTS20)), entropyTS20, marker="o", label="Tested on CVRP20")
plt.plot(range(len(entropyTS50)), entropyTS50, marker="o", label="Tested on CVRP50")
plt.plot(range(len(entropyTS100)), entropyTS100, marker="o", label="Tested on CVRP100")
plt.xlabel("Decoding step")
plt.ylabel("Mean entropy")
plt.legend()
plt.title("Attention dispersion of model trained on CVRP"+str(train_size))
plt.grid()
plt.savefig("attentionDispersionM"+str(train_size)+".svg", format="svg")
plt.show()