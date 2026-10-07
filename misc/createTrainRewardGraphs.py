import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    if x == 0:
        return "0"
    return f"{x/1000:g}k"

def set_scaled_ylim(ax, data, margin=0.05):
    ymin = data.min()
    ymax = data.max()
    yrange = ymax - ymin

    ax.set_ylim(
        ymin - yrange * margin,
        ymax + yrange * margin
    )

# ======================
# Load CSV
# ======================
df20 = pd.read_csv("metrics20.csv")
df50 = pd.read_csv("metrics50.csv")
df100 = pd.read_csv("metrics100.csv")

# Convert columns to numeric
cols = ["epoch", "step", "train/loss", "train/reward", "val/reward"]
for c in cols:
    df20[c] = pd.to_numeric(df20[c], errors="coerce")
    df50[c] = pd.to_numeric(df50[c], errors="coerce")
    df100[c] = pd.to_numeric(df100[c], errors="coerce")

# ======================
# Prepare data
# ======================
train_df20 = df20.dropna(subset=["train/reward"])
train_df50 = df50.dropna(subset=["train/reward"])
train_df100 = df100.dropna(subset=["train/reward"])

# ======================
# Subplots: 1 row, 3 columns
# ======================
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3))

datasets = [
    (train_df20, "CVRP20"),
    (train_df50, "CVRP50"),
    (train_df100, "CVRP100")
]

for ax, (df, title) in zip(axes, datasets):
    set_scaled_ylim(ax, df["train/reward"])
    ax.plot(df["step"], df["train/reward"], linewidth=1)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    #ax.set_xlabel("Steps", fontsize=14)
    #ax.set_ylabel("Reward", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True)

plt.figtext(0.002, 0.5, "Reward", va="center", rotation="vertical", fontsize=14)
plt.figtext(0.5, 0.03, "Steps", ha="center", fontsize=14)

plt.subplots_adjust(
    left=0.076,
    bottom=0.155,
    right=0.99,
    top=0.92,
    wspace=0.155
)

plt.show()