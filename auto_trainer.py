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
}

for enumArch, numLocList in trainingList.items():
    for numLoc in numLocList:
        print(f"\nStarting {architectureList[enumArch]} on CVRP{numLoc}\n")
        subprocess.run(
            [
                "python",
                "training.py",
                str(enumArch),
                str(numLoc),
            ],
            check=True
        )
        print(f"\nFinished {architectureList[enumArch]} on CVRP{numLoc}\n")
        