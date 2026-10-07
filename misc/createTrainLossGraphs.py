import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

def thousands_formatter(x, pos):
    if x == 0:
        return "0"
    return f"{x/1000:g}k"

# ======================
# Load CSV
# ======================
df20 = pd.read_csv("metrics20.csv")
df50 = pd.read_csv("metrics50.csv")
df100 = pd.read_csv("metrics100.csv")

# Convert columns to numeric (handle missing values)
cols = ["epoch", "step", "train/loss", "train/reward", "val/reward"]
for c in cols:
    df20[c] = pd.to_numeric(df20[c], errors="coerce")
    df50[c] = pd.to_numeric(df50[c], errors="coerce")
    df100[c] = pd.to_numeric(df100[c], errors="coerce")

# ======================
# Plot : Training Loss vs Step (optional)
# ======================
loss_df20 = df20.dropna(subset=["train/loss"])
loss_df50 = df50.dropna(subset=["train/loss"])
loss_df100 = df100.dropna(subset=["train/loss"])

plt.figure(figsize=(6, 3))
ax = plt.gca()
ax.xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
plt.ylim(-65, 40)
plt.plot(loss_df20["step"], loss_df20["train/loss"], linewidth=1.2, color="brown", label="CVRP20")
plt.plot(loss_df50["step"], loss_df50["train/loss"], linewidth=1.2, color="orange", label="CVRP50")
plt.plot(loss_df100["step"], loss_df100["train/loss"], linewidth=1.2, color="blue", label="CVRP100")
plt.xlabel("Steps", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True)

plt.subplots_adjust(
    top=0.97,
    bottom=0.17,
    right=0.99,
)

plt.legend()
#plt.legend(loc="upper left")
plt.show()
