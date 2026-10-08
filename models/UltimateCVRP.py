import torch
import re
import matplotlib.pyplot as plt
import numpy as np

from models.architectures import architectureList, getPolicyByModelArchitecture

Performances = {
  'LearnableScalingFactorCVRP20': [-6.475301265716553, -13.654390335083008, -25.296560287475586],
  'LearnableScalingFactorCVRP50': [-6.655007839202881, -13.39260482788086, -24.234600067138672],
  'LearnableScalingFactorCVRP100': [-6.9785943031311035, -13.546346664428711, -24.116262435913086],

  'AggregateCVRP20': [-6.643135070800781, -14.435135841369629, -27.722400665283203],
  'AggregateCVRP50': [-7.048409461975098, -14.101837158203125, -25.69315528869629],
  'AggregateCVRP100': [-7.498499393463135, -14.375325202941895, -25.454744338989258],

  'Set2SetCVRP20': [-6.486959934234619, -13.71340274810791, -25.45231819152832],
  'Set2SetCVRP50': [-6.660026550292969, -13.374135971069336, -24.17545509338379],
  'Set2SetCVRP100': [-6.931849002838135, -13.55203628540039, -24.050743103027344],

  'WeightedAverageCVRP20': [-6.466396808624268, -13.678462028503418, -25.427326202392578],
  'WeightedAverageCVRP50': [-6.64100456237793, -13.407833099365234, -24.221925735473633],
  'WeightedAverageCVRP100': [-6.910485744476318, -13.574901580810547, -24.069400787353516],

  'WeightedAverageSimpleCVRP20': [-6.4797282218933105, -13.705097198486328, -25.396793365478516], 
  'WeightedAverageSimpleCVRP50': [-6.6775922775268555, -13.390804290771484, -24.244709014892578],
  'WeightedAverageSimpleCVRP100': [-7.067850589752197, -13.62024974822998, -24.133094787597656],

  'GRUGraphCTXCVRP20': [-6.475761890411377, -13.735452651977539, -25.509490966796875],
  'GRUGraphCTXCVRP50': [-6.719632148742676, -13.405701637268066, -24.206892013549805],
  'GRUGraphCTXCVRP100': [-6.9845452308654785, -13.58089828491211, -24.030147552490234],

  'GraphSAGEEncoderCVRP20': [-6.633621692657471, -13.897101402282715, -25.705413818359375],
  'GraphSAGEEncoderCVRP50': [-6.943785190582275, -13.793320655822754, -24.835737228393555],
  'GraphSAGEEncoderCVRP100': [-7.509103775024414, -14.256101608276367, -25.160324096679688],

  'LearnableDepotModifierCVRP20': [-6.468982219696045, -13.727325439453125, -25.652448654174805],
  'LearnableDepotModifierCVRP50': [-6.738359451293945, -13.399460792541504, -24.211990356445312],
  'LearnableDepotModifierCVRP100': [-7.0453009605407715, -13.630511283874512, -24.125625610351562],
  
  'SetTransformerEncoderCVRP20': [-6.473565578460693, -13.626568794250488, -25.258037567138672],
  'SetTransformerEncoderCVRP50': [-6.6628923416137695, -13.42316722869873, -24.282133102416992],
  'SetTransformerEncoderCVRP100': [-7.046618938446045, -13.56241226196289, -24.114849090576172],
}

class UltimateCVRP():
    def __init__(self, raw_performance_list: dict[str, dict[int, dict[int, float]]] = None):
        self.RawPerfList = raw_performance_list
        self.problem_sizes = [20, 50, 100]
        if self.RawPerfList == None:
            self.convert_Performance()

    def convert_Performance(self):
        data: dict[str, dict[int, dict[int, float]]] = {}

        for key, values in Performances.items():
            match = re.fullmatch(r"(.+)CVRP(\d+)", key)
            if not match:
                raise ValueError(f"Invalid key: {key}")
            
            architecture = match.group(1)
            train_size = int(match.group(2))

            data.setdefault(architecture, {})[train_size] = {
                test_size: float(value)
                for test_size, value in zip(self.problem_sizes, values, strict=True)
            }

        self.RawPerfList = data
        return self.RawPerfList

    def select_model(self, train_size: int, test_size: int):
        if self.RawPerfList is None:
            self.convert_Performance()
    
        if test_size < 1:
            raise ValueError("Test size must be positive")
    
        scores = {}
    
        for architecture, train_data in self.RawPerfList.items():
            performances = train_data[train_size]
    
            x = np.array(sorted(performances.keys()))
            y = np.array([performances[size] for size in x])
    
            # Fit quadratic curve using all 3 known points
            coefficients = np.polyfit(x, y, 2)
    
            # Estimate performance at any test size
            scores[architecture] = np.polyval(
                coefficients, test_size
            )
    
        # Higher reward is better
        return max(scores, key=scores.get)

    def get_policy(self, architecture_name):
        class dummy:
            name = "cvrp"
        dummy_env = dummy()
        policy = getPolicyByModelArchitecture(architecture_name, dummy_env, 128, 3, 8)
        return policy

    def selector(self, train_size, test_size):
        if self.RawPerfList == None:
            self.convert_Performance()

        architecture_name = self.select_model(train_size, test_size)
        policy = self.get_policy(architecture_name)
        return architecture_name, policy

    def plot_performances(self, train_size):
        if self.RawPerfList is None:
            self.convert_Performance()

        x_plot = np.linspace(20, 100, 200)

        plt.figure(figsize=(8, 4))
        plt.rcParams.update({"font.size": 14})

        for architecture, train_data in self.RawPerfList.items():
            performances = train_data[train_size]

            x = np.array(sorted(performances.keys()))
            y = np.array([performances[size] for size in x])

            coefficients = np.polyfit(x, y, 2)
            y_plot = np.polyval(coefficients, x_plot)

            plt.plot(x_plot, y_plot, label=architecture)
            plt.scatter(x, y, s=15)

        plt.xlabel("Instance Size", fontsize=14)
        plt.ylabel("Estimated Reward", fontsize=14)
        #plt.title(f"Performance Curves - Train CVRP{train_size}", fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend(loc="upper right", fontsize=8.8)
        plt.tight_layout()
        plt.show()
