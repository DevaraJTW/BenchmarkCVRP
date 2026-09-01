import subprocess
from models.architectures import architectureList

trainingList = {
    0: [20,50,100],
    1: [20,50,100],
    2: [20,50,100],
    3: [20,50,100],
    4: [20,50,100],
    5: [20,50,100],
    6: [20,50,100],
    7: [20,50,100],
}

log_file = open("experiment_log.txt", "w")

for enumArch, numLocList in trainingList.items():
    for numLoc in numLocList:
        print(f"Starting {architectureList[enumArch]} on CVRP{numLoc}")
        subprocess.run(
            [
                "python",
                "training.py",
                str(enumArch),
                str(numLoc),
            ],
            check=True,
            #stdout=log_file,
            stderr=log_file,

        )
        print(f"Finished {architectureList[enumArch]} on CVRP{numLoc}")

log_file.close()