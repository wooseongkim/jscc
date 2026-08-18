from __future__ import annotations

import inspect

import pytest
import torch

from speech_jscc.evaluation.digital_crc_erasure import (
    CRC16_BITS,
    DigitalCRCTransport,
    GlobalBitInterleaver,
    crc_block_layout,
    erase_failed_crc_blocks,
    per_re_power_from_r_and_p,
    qpsk_hard_llrs,
)


R = [3, 4, 3, 1, 5, 1, 4, 3]
P = [
    0.13763791878150522, 0.2077466505307008, 0.21556578332765963,
    0.10236416924966271, 0.10394956790345766, 0.11150587368078434,
    0.06598904366003426, 0.05524099286619532,
]


def test_crc_layout_keeps_independent_layer_time_packets() -> None:
    layout = crc_block_layout(layers=8, frames=50, index_bit_width=10, block_frames=10)
    assert layout.packet_count == 40
    assert layout.payload_bits_per_packet == 100
    assert layout.packet_bits_per_packet == 100 + CRC16_BITS
    assert layout.total_information_bits == 8 * 5 * (100 + CRC16_BITS)


def test_erasure_zeros_exact_failed_layer_time_block_only() -> None:
    representation = torch.ones(1, 8, 50, 4)
    layout = crc_block_layout(layers=8, frames=50, index_bit_width=10, block_frames=10)
    erased = erase_failed_crc_blocks(representation, layout, failed_packet_ids={17})
    # packet 17 == layer 3, block [20, 30)
    assert erased[:, 3, 20:30].eq(0).all()
    assert erased[:, 3, :20].eq(1).all()
    assert erased[:, 3, 30:].eq(1).all()
    assert erased[:, 2].eq(1).all()
    assert erased.shape == representation.shape


def test_global_interleaver_uses_full_fixed_r4_qpsk_capacity() -> None:
    interleaver = GlobalBitInterleaver.from_r4_profile(R, P, paired_seed=11)
    assert interleaver.data_re_count == 5760
    assert interleaver.qpsk_bit_count == 11520
    assert interleaver.layer_re_counts == [240 * value for value in R]
    assert interleaver.total_transmit_energy == pytest.approx(5760.0)
    assert torch.equal(interleaver.permutation, GlobalBitInterleaver.from_r4_profile(R, P, paired_seed=11).permutation)


def test_interleaver_uses_cpu_seed_stream_then_moves_to_requested_device() -> None:
    # ``meta`` exercises a non-CPU target without requiring CUDA in CI. The
    # paired random stream remains CPU-defined, exactly as in CUDA runs.
    interleaver = GlobalBitInterleaver.from_r4_profile(R, P, paired_seed=17, device="meta")
    assert interleaver.permutation.device.type == "meta"


def test_r4_power_contract_is_loaded_not_reoptimized() -> None:
    power = per_re_power_from_r_and_p(R, P)
    assert sum(240 * repetition * value for repetition, value in zip(R, power, strict=True)) == pytest.approx(5760.0)
    assert all(value > 0 for value in power)


def test_noiseless_sionna_ldpc_qpsk_round_trip_is_bit_perfect() -> None:
    transport = DigitalCRCTransport(index_bit_width=10, crc_block_frames=10, ldpc_iterations=12)
    indices = torch.randint(0, 1024, (1, 8, 50), generator=torch.Generator().manual_seed(7))
    result = transport.noiseless_round_trip(indices, R, P, paired_seed=99)
    assert torch.equal(result.recovered_indices, indices)
    assert result.crc_failure_count == 0
    assert result.erasure_ratio == 0.0
    assert result.ldpc_metadata["implementation"] == "sionna.phy.fec.ldpc.LDPC5GEncoder"


def test_forced_crc_failure_becomes_packet_erasure_without_shape_change() -> None:
    transport = DigitalCRCTransport(index_bit_width=10, crc_block_frames=10, ldpc_iterations=12)
    indices = torch.randint(0, 1024, (1, 8, 50), generator=torch.Generator().manual_seed(11))
    encoded = transport.noiseless_round_trip(indices, R, P, paired_seed=12)
    # A fully sign-inverted LLR stream forces decoder/CRC failures; no index
    # is accepted merely because its decoded integer happens to be in range.
    failed = transport.transmit(indices, R, P, paired_seed=12, llrs=-qpsk_hard_llrs(encoded.qpsk_symbols))
    assert failed.packet_failures.any()
    assert failed.recovered_indices.shape == indices.shape


def test_transport_has_no_jammer_or_oracle_inputs() -> None:
    signature = inspect.signature(DigitalCRCTransport.transmit)
    forbidden = {"jammer_mask", "jammer_type", "jsr_db", "true_channel", "oracle_csi", "interference"}
    assert not (set(signature.parameters) & forbidden)
