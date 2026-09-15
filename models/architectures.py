from models.default_baseline import getBaselinePolicy
from models.complete_context_embedding import getAggregatePolicy
from models.set2set_graph_ctx import getSet2SetPolicy
from models.weighted_average import getWeightedAveragePolicy
from models.weighted_average_simple import getWeightedAverageSimplePolicy
from models.gru_graph_ctx import getGRUGraphCTXPolicy
from models.attention_sage_encoder import getAttentionSAGEEncoderPolicy
from models.learnable_depot_modifier import getLearnableDepotProbsPolicy
from models.learnable_scaling_factor import getLearnableScalingFactorPolicy
from models.baseline_esf import getBaselineESFPolicy

architectureList = [
    "Baseline",                 # 0
    "Aggregate",                # 1
    "Set2Set",                  # 2
    "WeightedAverage",          # 3
    "WeightedAverageSimple",    # 4
    "GRUGraphCTX",              # 5
    "AttentionSAGEEncoder",     # 6
    "LearnableDepotModifier",   # 7
    "LearnableScalingFactor",   # 8
    #"BaselineESF",              # 9
]

def getPolicyByModelArchitecture(
        modelArchitecture,
        env,
        embeddingDimension,
        numEncoderLayers,
        numberOfHeads,
        trainSize: int = None,
        testSize: int = None,
        ):
    
    match modelArchitecture:
        case "Baseline":
            policy = getBaselinePolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
        case "Aggregate":
            policy = getAggregatePolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
        case "Set2Set":
            policy = getSet2SetPolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
        case "WeightedAverage":
            policy = getWeightedAveragePolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
        case "WeightedAverageSimple":
            policy = getWeightedAverageSimplePolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
        case "GRUGraphCTX":
            policy = getGRUGraphCTXPolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
        case "AttentionSAGEEncoder":
            policy = getAttentionSAGEEncoderPolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
        case "LearnableDepotModifier":
            policy = getLearnableDepotProbsPolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
        case "LearnableScalingFactor":
            policy = getLearnableScalingFactorPolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads)
        #case "BaselineESF":
        #    policy = getBaselineESFPolicy(env, embeddingDimension, numEncoderLayers, numberOfHeads, trainSize, testSize)

    return policy