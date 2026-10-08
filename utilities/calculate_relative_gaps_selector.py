from models.architectures import architectureList

performances = {
   'BaselineCVRP20': [-6.477413654327393, -13.692718505859375, -25.392253875732422], 
   'BaselineCVRP50': [-6.646161079406738, -13.405354499816895, -24.266624450683594], 
   'BaselineCVRP100': [-6.985416412353516, -13.610718727111816, -24.131628036499023],

   'UltimateCVRPCVRP20': [-6.464076042175293, -13.628316879272461, -25.24919319152832], 
   'UltimateCVRPCVRP50': [-6.641445159912109, -13.372490882873535, -24.169282913208008], 
   'UltimateCVRPCVRP100': [-6.909908771514893, -13.55475902557373, -24.02848243713379],
}

relative_gaps = {
   'BaselineCVRP20': [-0.0, 2.143650927294803, 5.223956864106745],
   'BaselineCVRP50': [2.605166723706304, -0.0, 0.5594169360657665],
   'BaselineCVRP100': [7.842678963180613, 1.5319567065363842, -0.0],
   
   'UltimateCVRPCVRP20': [-0.0, 1.913076618557042, 5.080265712111873],
   'UltimateCVRPCVRP50': [2.7439206559384486, -0.0, 0.5859732358986794],
   'UltimateCVRPCVRP100': [6.897083611497365, 1.3630081657683568, -0.0]
}

gap_reduction = {
   'UltimateCVRPCVRP20': [0.0, 0.2305743087377612, 0.14369115199487137],
   'UltimateCVRPCVRP50': [-0.13875393223214472, 0.0, -0.026556299832912966],
   'UltimateCVRPCVRP100': [0.9455953516832478, 0.16894854076802734, 0.0],
}

def calculateRelativeGaps():
  for arch in [architectureList[0], architectureList[19]]:
      n_list = ["20", "50", "100"]

      for n in n_list:
        relative_gaps[arch+"CVRP"+n] = [
           ((performances[arch+"CVRP"+n][0] - performances[arch+"CVRP20"][0])/performances[arch+"CVRP20"][0])*100,
           ((performances[arch+"CVRP"+n][1] - performances[arch+"CVRP50"][1])/performances[arch+"CVRP50"][1])*100,
           ((performances[arch+"CVRP"+n][2] - performances[arch+"CVRP100"][2])/performances[arch+"CVRP100"][2])*100,
        ]

  print(relative_gaps)
  return

def compareGaps():
  for arch in [architectureList[19]]:
      n_list = ["20", "50", "100"]

      for n in n_list:
         gap_reduction[arch+"CVRP"+n] = [
            relative_gaps["BaselineCVRP"+n][0] - relative_gaps[arch+"CVRP"+n][0],
            relative_gaps["BaselineCVRP"+n][1] - relative_gaps[arch+"CVRP"+n][1],
            relative_gaps["BaselineCVRP"+n][2] - relative_gaps[arch+"CVRP"+n][2],
         ]

  print(gap_reduction)
  return

calculateRelativeGaps()
compareGaps()





