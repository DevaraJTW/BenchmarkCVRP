import subprocess
from models.architectures import architectureList

trainSizeList = [
    35, 50, 75, # 1     ABC
    50, 75, 35, # 2     BCA
    75, 35, 50, # 3     CAB
    75, 50, 35, # 4     CBA
    50, 35, 75, # 5     BAC
    35, 75, 50, # 6     ACB
    50, 75, 35, # 7     BCA
    75, 50, 35, # 8     CBA
    35, 50, 75, # 9     ABC
    75, 35, 50, # 10    CAB
]

trainSizeUpList = [20,50]
trainSizeDownList = [100,50]

trainingList = {
    13: trainSizeUpList,
    14: trainSizeDownList,
    15: trainSizeUpList,
    16: trainSizeDownList,
    17: trainSizeUpList,
    18: trainSizeDownList,
}

log_file = open("experiment_log.txt", "w")

for enumArch, numLocList in trainingList.items():
    epoch = 0
    for numLoc in numLocList:
        epoch += 75
        print(f"Starting {architectureList[enumArch]} on CVRP{numLoc}")
        subprocess.run(
            [
                "python",
                "mixed_training.py",
                str(enumArch),
                str(numLoc),
                str(epoch),
            ],
            check=True,
            #stdout=log_file,
            stderr=log_file,

        )
        print(f"Finished {architectureList[enumArch]} on CVRP{numLoc}")

log_file.close()