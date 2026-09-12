import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import LogNorm

n_train = "100"

gc_20 = torch.load("graph_context_CVRP"+n_train+"_on_20.pt").cpu().numpy()
gc_50 = torch.load("graph_context_CVRP"+n_train+"_on_50.pt").cpu().numpy()
gc_100 = torch.load("graph_context_CVRP"+n_train+"_on_100.pt").cpu().numpy()

# =========================================================
# 1. Center EACH distribution separately
# =========================================================

X20 = gc_20 - gc_20.mean(axis=0, keepdims=True)
X50 = gc_50 - gc_50.mean(axis=0, keepdims=True)
X100 = gc_100 - gc_100.mean(axis=0, keepdims=True)


# =========================================================
# 2. Fit ONE PCA using all centered distributions
# =========================================================

X_all = np.concatenate(
    [X20, X50, X100],
    axis=0
)

pca = PCA(n_components=2)
pca.fit(X_all)

Z20 = pca.transform(X20)
Z50 = pca.transform(X50)
Z100 = pca.transform(X100)

print("Explained variance ratio:",
      pca.explained_variance_ratio_)

print(
    "PC1 + PC2 explained variance:",
    pca.explained_variance_ratio_.sum()
)


# =========================================================
# 3. Common plotting limits
# =========================================================

Z_all = np.concatenate(
    [Z20, Z50, Z100],
    axis=0
)

x_limit = np.max(np.abs(Z_all[:, 0])) * 1.05
y_limit = np.max(np.abs(Z_all[:, 1])) * 1.05

extent = (
    -x_limit,
    x_limit,
    -y_limit,
    y_limit
)


# =========================================================
# 4. Plot
# =========================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5),
    sharex=True,
    sharey=True
)

datasets = [
    (Z20, "CVRP"+n_train+" model on CVRP20"),
    (Z50, "CVRP"+n_train+" model on CVRP50"),
    (Z100, "CVRP"+n_train+" model on CVRP100")
]

for ax, (Z, title) in zip(axes, datasets):

    hb = ax.hexbin(
        Z[:, 0],
        Z[:, 1],
        gridsize=60,

        # VERY IMPORTANT:
        # same bins / area for all three plots
        extent=extent,

        mincnt=1,
        cmap="Greys",
        norm=LogNorm()
    )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)

plt.tight_layout()
plt.savefig("pca_density_projected_CVRP"+n_train+"_model.svg", dpi=300, bbox_inches="tight", format="svg")
plt.show()




