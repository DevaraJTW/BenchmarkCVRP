import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import torch
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from rl4co.envs import CVRPEnv
from rl4co.models import REINFORCE
import rl4co.envs.routing.cvrp.render as cvrp_render

from utilities.get_model_file import getModelFile
from models.UltimateCVRP import UltimateCVRP

# =====================================================
# CONFIGURATION
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

INFERENCE_SIZES = list(range(1, 501))
PROBLEM_SIZES = [20, 50, 100]

# =====================================================
# PATCH RL4CO RENDER TO SHOW DEPOT
# =====================================================

_original_render = cvrp_render.render

def render_with_depot(td, actions=None, ax=None):
    return _original_render(
        td, actions, ax, skip_depot=False
    )

CVRPEnv.render = staticmethod(render_with_depot)

# =====================================================
# GUI
# =====================================================

class CVRPGUI:

    def __init__(self, root):

        self.root = root
        self.root.title("CVRP Routing Comparison")
        self.root.geometry("1300x500")
        self.root.minsize(950, 400)

        self.env = None
        self.td = None
        self.ultimate_cvrp = UltimateCVRP()
        self.ultimate_cvrp.convert_Performance()

        # Cache loaded models
        self.models = {}

        self.build_gui()

    # -------------------------------------------------
    # GUI LAYOUT
    # -------------------------------------------------

    def build_gui(self):

        # Controls
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")

        ttk.Label(
            controls, text="Instance Size:"
        ).pack(side="left", padx=5)

        self.instance_size = tk.StringVar(value="50")

        ttk.Combobox(
            controls,
            textvariable=self.instance_size,
            values=INFERENCE_SIZES,
            state="readonly",
            width=8
        ).pack(side="left", padx=5)

        ttk.Label(
            controls, text="Train Size:"
        ).pack(side="left", padx=5)

        self.train_size = tk.StringVar(value="20")

        ttk.Combobox(
            controls,
            textvariable=self.train_size,
            values=PROBLEM_SIZES,
            state="readonly",
            width=8
        ).pack(side="left", padx=5)

        ttk.Button(
            controls,
            text="Generate Instance",
            command=self.generate_instance
        ).pack(side="left", padx=10)

        self.solve_button = ttk.Button(
            controls,
            text="Solve Instance",
            command=self.solve_instance,
            state="disabled"
        )
        self.solve_button.pack(side="left", padx=5)

        # ---------------------------------------------
        # THREE HORIZONTAL PLOTS
        # ---------------------------------------------

        self.figure = Figure(
            figsize=(13, 4),
            dpi=100
        )

        (
            self.ax_instance,
            self.ax_baseline,
            self.ax_best
        ) = self.figure.subplots(1, 3)

        self.reset_axes()

        self.figure.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.08,
            top=0.88,
            wspace=0.20
        )

        self.canvas = FigureCanvasTkAgg(
            self.figure,
            master=self.root
        )

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        # Status bar
        self.status = tk.StringVar(value="Ready")

        ttk.Label(
            self.root,
            textvariable=self.status,
            padding=10
        ).pack(fill="x")

        self.canvas.draw_idle()

    # -------------------------------------------------
    # RESET PLOTS
    # -------------------------------------------------

    def reset_axes(self):

        titles = [
            "Unsolved Instance",
            "Baseline",
            "Best Model"
        ]

        axes = [
            self.ax_instance,
            self.ax_baseline,
            self.ax_best
        ]

        for ax, title in zip(axes, titles):

            ax.clear()
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.15)

    # -------------------------------------------------
    # GENERATE INSTANCE
    # -------------------------------------------------

    def generate_instance(self):

        try:
            num_loc = int(self.instance_size.get())
            self.env = CVRPEnv(
                generator_params={
                    "num_loc": num_loc,
                    "capacity": 30,
                    "demand_min": 1,
                    "demand_max": 9,
                    }
            )

            self.td = self.env.reset(
                batch_size=[1]
            )

            depot = self.td["depot"][0].cpu().numpy()
            locs = self.td["locs"][0].cpu().numpy()

            self.reset_axes()

            # Plot customers
            self.ax_instance.scatter(
                locs[:, 0],
                locs[:, 1],
                s=18,
                color="tab:blue",
                label="Customers",
                zorder=2
            )

            # Plot depot
            self.ax_instance.scatter(
                depot[0],
                depot[1],
                s=100,
                marker="s",
                color="tab:orange",
                edgecolor="black",
                label="Depot",
                zorder=3
            )

            self.ax_instance.set_title(
                f"Unsolved CVRP{num_loc}"
            )
            self.ax_instance.legend(
                loc="upper right",
                fontsize=8
            )
            self.canvas.draw_idle()
            self.solve_button.config(state="normal")
            self.status.set(
                f"Generated CVRP{num_loc} instance. "
                "Ready to solve."
            )

        except Exception as error:
            messagebox.showerror(
                "Generation Error", str(error)
            )
            self.status.set("Generation failed")

    # -------------------------------------------------
    # LOAD MODEL
    # -------------------------------------------------

    def load_model(
        self,
        model_type: str,
        train_size: int,
        instance_size: int,
    ):
        if model_type == "Ultimate":
            model_type = self.ultimate_cvrp.select_model(train_size=train_size, test_size=instance_size)

        print(model_type)
        cache_key = (model_type, train_size)
        if cache_key in self.models:
            return self.models[cache_key]
        
        checkpoint_path = getModelFile(train_size, model_type)

        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        model = REINFORCE.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        policy = model.policy.to(DEVICE)
        policy.eval()

        self.models[cache_key] = policy

        return policy

    # -------------------------------------------------
    # RUN INFERENCE
    # -------------------------------------------------

    def run_model(
        self,
        model_type: str,
        train_size: int,
        instance_size: int,
        ax
    ):

        policy = self.load_model(
            model_type,
            train_size,
            instance_size
        )

        # Same instance for both models
        td_input = self.td.clone().to(DEVICE)

        with torch.inference_mode():

            output = policy(
                td_input,
                phase="test",
                decode_type="greedy",
                return_actions=True
            )

        actions = output["actions"][0].detach().cpu()

        cost = -output["reward"][0].item()

        ax.clear()

        self.env.render(
            self.td[0],
            actions,
            ax=ax
        )

        display_name = (
            "Baseline"
            if model_type == "Baseline"
            else "Best Model"
        )

        ax.set_title(
            f"{display_name} (Train CVRP{train_size})\n"
            f"Tour Length: {cost:.4f}"
        )

        ax.set_aspect("equal")

        return cost

    # -------------------------------------------------
    # SOLVE INSTANCE
    # -------------------------------------------------

    def solve_instance(self):

        if self.td is None:
            messagebox.showwarning(
                "Missing Instance",
                "Generate an instance first."
            )
            return

        train_size = int(self.train_size.get())
        try:
            self.solve_button.config(state="disabled")
            self.status.set("Running inference...")
            self.root.update_idletasks()
            instance_size = self.td["locs"].shape[1] -1

            baseline_cost = self.run_model("Baseline",train_size,instance_size,self.ax_baseline)
            best_cost = self.run_model("Ultimate",train_size,instance_size,self.ax_best)

            self.canvas.draw_idle()
            self.status.set(
                f"Train CVRP{train_size} → "
                f"Test CVRP{instance_size} | "
                f"Baseline: {baseline_cost:.4f} | "
                f"Best: {best_cost:.4f}"
            )

        except Exception as error:
            messagebox.showerror(
                "Inference Error",
                str(error)
            )
            self.status.set("Inference failed")
        finally:
            self.solve_button.config(state="normal")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    root = tk.Tk()
    app = CVRPGUI(root)
    root.mainloop()