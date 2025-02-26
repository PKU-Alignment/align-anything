from .early_fusion_gru_models import EarlyFusionCnnRNN
from .early_fusion_tsfm_models import EarlyFusionCnnTransformer

REGISTERED_MODELS = {
    "EarlyFusionCnnTransformer": EarlyFusionCnnTransformer,
    "EarlyFusionCnnRNN": EarlyFusionCnnRNN,
}
