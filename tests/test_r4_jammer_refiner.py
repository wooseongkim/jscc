import torch
from torch import nn

from models.adaptive_latent_refiner import MoEAdaptiveLatentRefiner
from models.jammer_estimator import JAMMER_TYPE_CLASSES, JammerEstimator
from models.adaptive_latent_refiner import save_adaptive_latent_refiner_checkpoint
from models.jammer_estimator import save_jammer_estimator_checkpoint
from speech_jscc.training.r4_waveform_finetune import R4ForwardCondition, R4WaveformForward


class _Codec(nn.Module):
    representation_shape = (2, 3, 4)
    waveform_samples = 3

    def decode_representation(self, representation):
        return representation.mean(dim=(1, 3))


class _Encoder(nn.Module):
    channel_state_dim = 5

    def forward(self, representation, state):
        flat = representation.flatten(1)
        return torch.complex(flat.mean(dim=1, keepdim=True), flat.std(dim=1, keepdim=True)).repeat(1, 1920)


class _Decoder(nn.Module):
    def forward(self, symbols, state):
        return symbols.real[:, :24].reshape(-1, 2, 3, 4)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()


def _condition():
    return R4ForwardCondition(
        snr_db=5.0,
        tti=0,
        tap_coefficients=torch.tensor([[1 + 0j, .2 + .1j, .1 - .1j, .05 + 0j, .02j, .01 + 0j]]),
        noise_seed=23,
    )


def test_no_refiner_preserves_raw_decoder_reconstruction():
    engine = R4WaveformForward(_Codec(), _Model())
    output = engine.forward(torch.randn(1, 2, 3, 4), channel_condition=_condition())

    assert output.refiner_mode == "no_refiner"
    assert torch.equal(output.reconstruction, output.raw_reconstruction)
    assert output.jammer_posterior is None
    assert output.jammer_mask_prob is None


def test_oracle_and_learned_refiner_modes_run_after_physical_forward():
    codec, model = _Codec(), _Model()
    estimator = JammerEstimator(hidden_dim=8)
    refiner = MoEAdaptiveLatentRefiner(
        representation_shape=codec.representation_shape,
        # R4 forwards the observable receiver state, whose schema has 8 values;
        # this is distinct from the transmitter encoder state dimension.
        channel_state_dim=8,
        num_experts=len(JAMMER_TYPE_CLASSES),
        hidden_dim=8,
    )
    representation = torch.randn(1, 2, 3, 4)
    common = dict(
        jammer_estimator=estimator,
        adaptive_refiner=refiner,
    )
    oracle = R4WaveformForward(codec, model, refiner_mode="oracle_mask_refiner", **common).forward(
        representation, channel_condition=_condition(), jammer_type="broadband_awgn", jammer_jsr_db=0.0,
    )
    learned = R4WaveformForward(codec, model, refiner_mode="learned_mask_refiner", **common).forward(
        representation, channel_condition=_condition(), jammer_type="broadband_awgn", jammer_jsr_db=0.0,
    )

    for output in (oracle, learned):
        assert output.reconstruction.shape == representation.shape
        assert output.jammer_mask_prob is not None
        assert output.jammer_posterior is not None
        assert torch.isfinite(output.reconstruction).all()
    assert torch.equal(oracle.jammer_mask_prob, oracle.jammer_mask)


def test_learned_mode_requires_estimator_and_refiner():
    with __import__("pytest").raises(ValueError, match="jammer_estimator"):
        R4WaveformForward(_Codec(), _Model(), refiner_mode="learned_mask_refiner")


def test_learned_mode_loads_explicit_refinement_checkpoints(tmp_path):
    codec, model = _Codec(), _Model()
    estimator_path = tmp_path / "estimator.pt"
    refiner_path = tmp_path / "refiner.pt"
    torch.save(save_jammer_estimator_checkpoint(JammerEstimator(hidden_dim=8)), estimator_path)
    torch.save(
        save_adaptive_latent_refiner_checkpoint(
            MoEAdaptiveLatentRefiner(
                representation_shape=codec.representation_shape,
                channel_state_dim=8,
                num_experts=len(JAMMER_TYPE_CLASSES),
                hidden_dim=8,
            )
        ),
        refiner_path,
    )

    engine = R4WaveformForward(
        codec, model,
        refiner_mode="learned_posterior_moe_refiner",
        jammer_estimator_checkpoint=estimator_path,
        adaptive_refiner_checkpoint=refiner_path,
    )

    assert engine.jammer_estimator is not None
    assert engine.adaptive_refiner is not None
