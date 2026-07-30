from models.default_baseline import getBaselinePolicy
from models.complete_context_embedding import getAggregatePolicy
from models.set2set_graph_ctx import getSet2SetPolicy
from models.weighted_average import getWeightedAveragePolicy
from models.weighted_average_simple import getWeightedAverageSimplePolicy
from models.gru_graph_ctx import getGRUGraphCTXPolicy
from models.attention_sage_encoder import getAttentionSAGEEncoderPolicy

architectureList = [
    "Baseline",                 # 0
    "Aggregate",                # 1
    "Set2Set",                  # 2
    "WeightedAverage",          # 3
    "WeightedAverageSimple",    # 4
    "GRUGraphCTX",              # 5
    "AttentionSAGEEncoder"      # 6
]

def getPolicyByModelArchitecture(
        modelArchitecture,
        env,
        embeddingDimension,
        numEncoderLayers,
        numberOfHeads
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

    return policy