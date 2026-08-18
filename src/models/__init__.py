"""Neural joint source-channel encoder and decoder."""

from models.jscc_decoder import JSCCDecoder
from models.jscc_encoder import (
    JSCCEncoder,
    deterministic_layer_gates,
    normalize_complex_power,
)
from models.soft_codebook import (
    CodecRepresentationLoss,
    SoftCodebook,
    continuous_latent_loss,
    soft_codebook_projection,
    top_k_token_accuracy,
)
from models.channel_state import CHANNEL_STATE_DIM, build_channel_state
from models.learned_gate import (
    LearnedLayerGate,
    gate_budget_loss,
    gate_smoothness_loss,
    load_learned_gate_checkpoint,
    save_learned_gate_checkpoint,
)
from models.latent_refiner import (
    LatentRefiner,
    load_latent_refiner_checkpoint,
    save_latent_refiner_checkpoint,
)
from models.jammer_estimator import (
    JAMMER_TYPE_CLASSES,
    JammerEstimate,
    JammerEstimator,
    jammer_estimation_loss,
    load_jammer_estimator_checkpoint,
    save_jammer_estimator_checkpoint,
)
from models.adaptive_latent_refiner import (
    MoEAdaptiveLatentRefiner,
    load_adaptive_latent_refiner_checkpoint,
    no_jammer_identity_regularization,
    save_adaptive_latent_refiner_checkpoint,
)
from models.resource_allocator import allocate_resources, deallocate_resources

__all__ = [
    "CHANNEL_STATE_DIM",
    "CodecRepresentationLoss",
    "JSCCDecoder",
    "JSCCEncoder",
    "JAMMER_TYPE_CLASSES",
    "JammerEstimate",
    "JammerEstimator",
    "LearnedLayerGate",
    "LatentRefiner",
    "MoEAdaptiveLatentRefiner",
    "SoftCodebook",
    "build_channel_state",
    "continuous_latent_loss",
    "deterministic_layer_gates",
    "gate_budget_loss",
    "gate_smoothness_loss",
    "jammer_estimation_loss",
    "load_adaptive_latent_refiner_checkpoint",
    "load_jammer_estimator_checkpoint",
    "load_learned_gate_checkpoint",
    "load_latent_refiner_checkpoint",
    "normalize_complex_power",
    "no_jammer_identity_regularization",
    "save_adaptive_latent_refiner_checkpoint",
    "save_jammer_estimator_checkpoint",
    "save_learned_gate_checkpoint",
    "save_latent_refiner_checkpoint",
    "allocate_resources",
    "deallocate_resources",
    "soft_codebook_projection",
    "top_k_token_accuracy",
]
