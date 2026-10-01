
performances = {
   'BaselineCVRP20': [-6.480193614959717, -13.692167282104492, -25.40791893005371], 
   'BaselineCVRP50': [-6.649388313293457, -13.39975357055664, -24.27684783935547], 
   'BaselineCVRP100': [-6.983904838562012, -13.611343383789062, -24.144760131835938],
   
   'LearnableScalingFactorCVRP20': [-6.475301265716553, -13.654390335083008, -25.296560287475586],
   'LearnableScalingFactorCVRP50': [-6.655007839202881, -13.39260482788086, -24.234600067138672],
   'LearnableScalingFactorCVRP100': [-6.9785943031311035, -13.546346664428711, -24.116262435913086],
   
   'LearnableDepotModifierCVRP20': [-6.468982219696045, -13.727325439453125, -25.652448654174805],
   'LearnableDepotModifierCVRP50': [-6.738359451293945, -13.399460792541504, -24.211990356445312],
   'LearnableDepotModifierCVRP100': [-7.0453009605407715, -13.630511283874512, -24.125625610351562],
}

mixPerformances = {
  'BaselineUp': [-6.838509559631348, -13.843806266784668, -25.04463005065918], 
  'BaselineDown': [-6.845996856689453, -13.730825424194336, -24.789091110229492], 

  'LearnableScalingFactorUp': [-6.687300205230713, -13.582318305969238, -24.640798568725586], 
  'LearnableScalingFactorDown': [-6.874485969543457, -13.753826141357422, -24.82077407836914], 

  'LearnableDepotModifierUp': [-6.721968173980713, -13.708908081054688, -24.84619140625], 
  'LearnableDepotModifierDown': [-6.80620002746582, -13.703827857971191, -24.761240005493164]
}

relative_gaps = {
      'Baseline': [3.7269780851404017, 5.644943090670452],
      'LearnableScalingFactor': [2.175030787653809, 6.164727901393338],
      'LearnableDepotModifier': [2.9867237746956703, 5.212841778155639],
   }
relative_gaps = {}

gap_reduction = {
      'LearnableScalingFactor': [1.5519472974865929, -0.5197848107228857],
      'LearnableDepotModifier': [0.7402543104447314, 0.43210131251481343],
   }
gap_reduction = {}

archList = [
   "Baseline",
   "LearnableScalingFactor",
   "LearnableDepotModifier",
]

def calculateRelativeGapsMixed():
  for arch in archList:
      relative_gaps[arch] = [
         ((mixPerformances[arch+"Up"][2] - performances[arch+"CVRP100"][2])/performances[arch+"CVRP100"][2])*100,
         ((mixPerformances[arch+"Down"][0] - performances[arch+"CVRP20"][0])/performances[arch+"CVRP20"][0])*100,
      ]

  print(relative_gaps)
  return

def compareGapsMixed():
  for arch in archList[1:]:
      gap_reduction[arch] = [
         relative_gaps["Baseline"][0] - relative_gaps[arch][0],
         relative_gaps["Baseline"][1] - relative_gaps[arch][1],
      ]

  print(gap_reduction)
  return

calculateRelativeGapsMixed()
compareGapsMixed()





