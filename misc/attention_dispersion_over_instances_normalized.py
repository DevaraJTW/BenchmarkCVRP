import torch
import matplotlib.pyplot as plt

def attention_dispersion(logprobs_list, actions_list, num_customers):
    logp = torch.as_tensor(logprobs_list)
    actions = torch.as_tensor(actions_list)

    # Active instances BEFORE each decoding step
    customer_selected = actions != 0
    customers_visited = customer_selected.cumsum(dim=0)

    active = torch.cat(
        [
            torch.ones(
                (1, actions.shape[1]),
                dtype=torch.bool,
                device=actions.device
            ),
            customers_visited[:-1] < num_customers
        ],
        dim=0
    )

    # Convert log-probabilities to probabilities
    probs = torch.exp(logp)

    # Raw entropy
    entropy = -(
        probs * logp
    ).nan_to_num(0.0).sum(dim=-1)

    # Number of feasible actions
    num_feasible = torch.isfinite(logp).sum(dim=-1)

    # Normalized entropy is only defined when >1 action is feasible
    valid = active & (num_feasible > 1)

    normalized_entropy = torch.zeros_like(entropy)

    normalized_entropy[valid] = (
        entropy[valid]
        / torch.log(num_feasible[valid].float())
    )

    # Average over valid decoding steps for each instance
    valid_float = valid.float()

    normalized_entropy_per_instance = (
        (normalized_entropy * valid_float).sum(dim=0)
        / valid_float.sum(dim=0).clamp_min(1)
    )

    mean_entropy = normalized_entropy_per_instance.mean().item()
    std_entropy = normalized_entropy_per_instance.std().item()

    return mean_entropy, std_entropy



dataM20TS20 = torch.load(
    "misc/attention_data_M20_TS20.pt",
    weights_only=False
)

entropyM20TS20, stdM20TS20 = attention_dispersion(
    dataM20TS20["logprobs"],
    dataM20TS20["actions"],
    20
)

dataM20TS100 = torch.load(
    "misc/attention_data_M20_TS100.pt",
    weights_only=False
)

entropyM20TS100, stdM20TS100 = attention_dispersion(
    dataM20TS100["logprobs"],
    dataM20TS100["actions"],
    100
)

print("Test20 :", entropyM20TS20, "+/-", stdM20TS20)
print("Test100:", entropyM20TS100, "+/-", stdM20TS100)


plt.figure()
plt.plot(range(len(entropyM20TS20)), entropyM20TS20, marker="o", label="Trained on CVRP20")
plt.plot(range(len(entropyM20TS100)), entropyM20TS100, marker="o", label="Trained on CVRP20")

plt.xlabel("Decoding step")
plt.ylabel("Mean entropy")
plt.title("Attention dispersion")
plt.grid()
plt.show()