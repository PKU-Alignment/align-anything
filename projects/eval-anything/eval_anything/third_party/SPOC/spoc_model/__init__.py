# Copyright 2024 Allen Institute for AI
# ==============================================================================

from transformers import AutoConfig, AutoModel

from .early_fusion_tsfm_models import (
    EarlyFusionCnnTransformer,
    EarlyFusionCnnTransformerAgent,
    EarlyFusionCnnTransformerConfig,
)


globals()['EarlyFusionCnnTransformer'] = EarlyFusionCnnTransformer
globals()['EarlyFusionCnnTransformerAgent'] = EarlyFusionCnnTransformerAgent

AutoConfig.register('MM', EarlyFusionCnnTransformerConfig)
# AutoModel.register("MM", EarlyFusionCnnTransformer)
