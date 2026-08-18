"""Digital CRC-erasure source/channel transport for the fixed R4 budget.

This module intentionally owns only the digital source/channel-code boundary.
The R4 OFDM, pilot, fading, LS-CSI, and equalization path remains the shared
physical path in :mod:`speech_jscc.training.r4_waveform_finetune`.

CRC source packets are independent for every RVQ layer and time block.  Their
rate-matched LDPC code bits are then interleaved over the *global* fixed R4
data-RE pool.  This is necessary because the selected UEP profile has layers
with only one nominal repetition and cannot carry a complete RVQ+CRC packet
inside a layer-local QPSK silo.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import torch
from torch import Tensor


CRC16_BITS = 16


@dataclass(frozen=True)
class CRCBlock:
    packet_id: int
    layer: int
    frame_start: int
    frame_stop: int


@dataclass(frozen=True)
class CRCBlockLayout:
    layers: int
    frames: int
    index_bit_width: int
    block_frames: int
    blocks: tuple[CRCBlock, ...]

    @property
    def packet_count(self) -> int:
        return len(self.blocks)

    @property
    def payload_bits_per_packet(self) -> int:
        return self.block_frames * self.index_bit_width

    @property
    def packet_bits_per_packet(self) -> int:
        return self.payload_bits_per_packet + CRC16_BITS

    @property
    def total_information_bits(self) -> int:
        return self.packet_count * self.packet_bits_per_packet


@dataclass(frozen=True)
class GlobalBitInterleaver:
    """Deterministic full-budget interleaver for R4 QPSK bit positions."""

    repetition: tuple[int, ...]
    power_shares: tuple[float, ...]
    per_re_power: tuple[float, ...]
    layer_re_counts: list[int]
    data_re_count: int
    qpsk_bit_count: int
    total_transmit_energy: float
    permutation: Tensor

    @classmethod
    def from_r4_profile(
        cls,
        repetition: Iterable[int],
        power_shares: Iterable[float],
        *,
        paired_seed: int,
        device: torch.device | str = "cpu",
    ) -> "GlobalBitInterleaver":
        rep = tuple(int(value) for value in repetition)
        shares = tuple(float(value) for value in power_shares)
        if len(rep) != 8 or len(shares) != 8:
            raise ValueError("R4 profile requires exactly eight layers")
        if any(value < 1 for value in rep) or sum(rep) != 24:
            raise ValueError("R4 repetition must be integer, >=1, and sum to 24")
        if any(not math.isfinite(value) or value <= 0 for value in shares):
            raise ValueError("power shares must be finite and positive")
        if not math.isclose(sum(shares), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("power shares must sum to one")
        power = tuple(per_re_power_from_r_and_p(rep, shares))
        layer_re_counts = [240 * value for value in rep]
        data_re_count = sum(layer_re_counts)
        if data_re_count != 5760:
            raise AssertionError("R4 profile must use exactly 5760 data RE")
        total_energy = sum(count * value for count, value in zip(layer_re_counts, power, strict=True))
        if not math.isclose(total_energy, 5760.0, rel_tol=0.0, abs_tol=1e-5):
            raise AssertionError("R4 power profile must have total energy 5760")
        # A separate generator ensures paired conditions, not global torch RNG,
        # determine the interleaver.
        generator = torch.Generator(device="cpu").manual_seed(int(paired_seed))
        # Generate on CPU because the paired seed contract deliberately uses a
        # CPU generator. Moving the resulting integer permutation makes the
        # exact same mapping valid for CPU and CUDA evaluations.
        permutation = torch.randperm(data_re_count * 2, generator=generator, device="cpu").to(device)
        return cls(
            repetition=rep, power_shares=shares, per_re_power=power,
            layer_re_counts=layer_re_counts, data_re_count=data_re_count,
            qpsk_bit_count=data_re_count * 2, total_transmit_energy=float(total_energy),
            permutation=permutation,
        )

    def map_code_bits(self, code_bits: Tensor) -> Tensor:
        if code_bits.ndim != 2 or code_bits.shape[-1] != self.qpsk_bit_count:
            raise ValueError(f"code_bits must have shape [B,{self.qpsk_bit_count}]")
        return code_bits[:, self.permutation.to(code_bits.device)]

    def unmap_llrs(self, llrs: Tensor) -> Tensor:
        if llrs.ndim != 2 or llrs.shape[-1] != self.qpsk_bit_count:
            raise ValueError(f"llrs must have shape [B,{self.qpsk_bit_count}]")
        inverse = torch.empty_like(self.permutation)
        inverse[self.permutation] = torch.arange(self.qpsk_bit_count, device=self.permutation.device)
        return llrs[:, inverse.to(llrs.device)]


def per_re_power_from_r_and_p(repetition: Iterable[int], power_shares: Iterable[float]) -> list[float]:
    """Return ``a_i = 24 p_i / r_i`` for the fixed R4 energy contract."""
    rep = [int(value) for value in repetition]
    shares = [float(value) for value in power_shares]
    if len(rep) != len(shares):
        raise ValueError("repetition and power shares lengths differ")
    return [24.0 * share / count for count, share in zip(rep, shares, strict=True)]


def crc_block_layout(*, layers: int, frames: int, index_bit_width: int, block_frames: int) -> CRCBlockLayout:
    if min(layers, frames, index_bit_width, block_frames) <= 0:
        raise ValueError("CRC layout dimensions must be positive")
    if frames % block_frames:
        raise ValueError("frames must be divisible by crc_block_frames")
    blocks = tuple(
        CRCBlock(packet_id=(layer * (frames // block_frames)) + block,
                 layer=layer, frame_start=block * block_frames,
                 frame_stop=(block + 1) * block_frames)
        for layer in range(layers)
        for block in range(frames // block_frames)
    )
    return CRCBlockLayout(layers, frames, index_bit_width, block_frames, blocks)


def _crc16_ccitt(bits: Tensor) -> int:
    crc = 0xFFFF
    for bit in bits.detach().to("cpu", dtype=torch.int64).tolist():
        xor = ((crc >> 15) & 1) ^ int(bit)
        crc = (crc << 1) & 0xFFFF
        if xor:
            crc ^= 0x1021
    return crc


def append_crc16(payload: Tensor) -> Tensor:
    if payload.ndim != 1 or not torch.all((payload == 0) | (payload == 1)):
        raise ValueError("payload must be a one-dimensional binary tensor")
    crc = _crc16_ccitt(payload)
    checksum = torch.tensor([(crc >> shift) & 1 for shift in range(15, -1, -1)], dtype=payload.dtype, device=payload.device)
    return torch.cat((payload, checksum))


def check_crc16(packet: Tensor) -> bool:
    if packet.ndim != 1 or packet.numel() < CRC16_BITS:
        raise ValueError("packet must contain payload and CRC")
    payload, supplied = packet[:-CRC16_BITS], packet[-CRC16_BITS:]
    expected = append_crc16(payload)[-CRC16_BITS:]
    return bool(torch.equal(supplied.to(expected.device), expected))


def indices_to_bits(indices: Tensor, bit_width: int) -> Tensor:
    if indices.ndim != 3 or bit_width <= 0:
        raise ValueError("indices must have shape [B,L,T] and bit_width must be positive")
    shifts = torch.arange(bit_width - 1, -1, -1, device=indices.device)
    return ((indices.to(torch.long).unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def bits_to_indices(bits: Tensor) -> Tensor:
    if bits.ndim < 1:
        raise ValueError("bits must have at least one dimension")
    width = bits.shape[-1]
    shifts = torch.arange(width - 1, -1, -1, device=bits.device)
    return ((bits.to(torch.long) << shifts).sum(dim=-1)).to(torch.long)


def erase_failed_crc_blocks(representation: Tensor, layout: CRCBlockLayout, failed_packet_ids: set[int]) -> Tensor:
    if representation.ndim != 4 or tuple(representation.shape[1:3]) != (layout.layers, layout.frames):
        raise ValueError("representation shape does not match CRC layout")
    output = representation.clone()
    for block in layout.blocks:
        if block.packet_id in failed_packet_ids:
            output[:, block.layer, block.frame_start:block.frame_stop].zero_()
    return output


def qpsk_modulate(bits: Tensor) -> Tensor:
    if bits.ndim != 2 or bits.shape[-1] % 2:
        raise ValueError("QPSK input must be [B, even_bits]")
    signs = 2.0 * bits.to(torch.float32) - 1.0
    return torch.complex(signs[:, 0::2], signs[:, 1::2]) / math.sqrt(2.0)


def qpsk_hard_llrs(symbols: Tensor, magnitude: float = 20.0) -> Tensor:
    if symbols.ndim != 2:
        raise ValueError("symbols must have shape [B,N]")
    # Sionna's LDPC decoder convention is positive LLR -> bit one.
    return torch.stack((symbols.real, symbols.imag), dim=-1).reshape(symbols.shape[0], -1).sign() * float(magnitude)


@dataclass
class DigitalTransportResult:
    recovered_indices: Tensor
    qpsk_symbols: Tensor
    code_bits: Tensor
    packet_failures: Tensor
    layout: CRCBlockLayout
    interleaver: GlobalBitInterleaver
    ldpc_metadata: dict[str, object]

    @property
    def crc_failure_count(self) -> int:
        return int(self.packet_failures.sum().item())

    @property
    def erasure_ratio(self) -> float:
        return float(self.packet_failures.float().mean().item())


class DigitalCRCTransport:
    """Sionna 3GPP 38.212 LDPC packet transport over the fixed global R4 budget."""

    def __init__(self, *, index_bit_width: int, crc_block_frames: int = 10, ldpc_iterations: int = 20):
        if index_bit_width <= 0 or crc_block_frames <= 0 or ldpc_iterations <= 0:
            raise ValueError("digital transport dimensions must be positive")
        self.index_bit_width = int(index_bit_width)
        self.crc_block_frames = int(crc_block_frames)
        self.ldpc_iterations = int(ldpc_iterations)

    def _codec(self, k: int, n: int, device: torch.device) -> tuple[object, object, dict[str, object]]:
        try:
            from sionna.phy.fec.ldpc.encoding import LDPC5GEncoder
            from sionna.phy.fec.ldpc.decoding import LDPC5GDecoder
        except ImportError as error:  # pragma: no cover - dependency gate
            raise RuntimeError("Sionna 5G LDPC is required; install `sionna`") from error
        # Sionna accepts its own string device identifiers (``"cpu"``),
        # whereas the paired R4 path carries a ``torch.device``.
        sionna_device = str(device)
        if sionna_device.startswith("cuda"):
            sionna_device = "cuda" if sionna_device == "cuda" else sionna_device
        encoder = LDPC5GEncoder(k=k, n=n, device=sionna_device)
        decoder = LDPC5GDecoder(encoder, num_iter=self.ldpc_iterations, device=sionna_device)
        metadata = {
            "implementation": "sionna.phy.fec.ldpc.LDPC5GEncoder",
            "decoder": "sionna.phy.fec.ldpc.LDPC5GDecoder",
            "standard": "3GPP TS 38.212",
            "base_graph": getattr(encoder, "_bg", None),
            "information_bits_per_packet": k,
            "rate_matched_bits_per_packet": n,
            "effective_rate": float(k / n),
            "decoder_iterations": self.ldpc_iterations,
        }
        return encoder, decoder, metadata

    def _packet_bits(self, indices: Tensor, layout: CRCBlockLayout) -> Tensor:
        index_bits = indices_to_bits(indices, layout.index_bit_width)
        packets: list[Tensor] = []
        for block in layout.blocks:
            payload = index_bits[:, block.layer, block.frame_start:block.frame_stop].reshape(indices.shape[0], -1)
            packets.append(torch.stack([append_crc16(row) for row in payload], dim=0))
        return torch.stack(packets, dim=1)

    def transmit(
        self,
        indices: Tensor,
        repetition: Iterable[int],
        power_shares: Iterable[float],
        *,
        paired_seed: int,
        llrs: Tensor | None = None,
    ) -> DigitalTransportResult:
        if indices.ndim != 3:
            raise ValueError("indices must have shape [B,L,T]")
        max_index = 1 << self.index_bit_width
        if int(indices.min()) < 0 or int(indices.max()) >= max_index:
            raise ValueError("RVQ index is outside configured fixed-width range")
        layout = crc_block_layout(layers=indices.shape[1], frames=indices.shape[2], index_bit_width=self.index_bit_width, block_frames=self.crc_block_frames)
        interleaver = GlobalBitInterleaver.from_r4_profile(repetition, power_shares, paired_seed=paired_seed, device=indices.device)
        if interleaver.qpsk_bit_count % layout.packet_count:
            raise AssertionError("global R4 bit capacity must split evenly across CRC packets")
        n = interleaver.qpsk_bit_count // layout.packet_count
        k = layout.packet_bits_per_packet
        if n < k:
            raise ValueError(f"fixed R4 budget cannot carry CRC payload: n={n}, k={k}")
        encoder, decoder, metadata = self._codec(k, n, indices.device)
        packets = self._packet_bits(indices, layout)
        encoded = encoder(packets.reshape(-1, k)).reshape(indices.shape[0], layout.packet_count, n)
        flat = encoded.reshape(indices.shape[0], -1)
        mapped_bits = interleaver.map_code_bits(flat)
        qpsk = qpsk_modulate(mapped_bits)
        received_llrs = qpsk_hard_llrs(qpsk) if llrs is None else llrs
        unmapped = interleaver.unmap_llrs(received_llrs).reshape(indices.shape[0], layout.packet_count, n)
        decoded = decoder(unmapped.reshape(-1, n)).reshape(indices.shape[0], layout.packet_count, k).round().to(torch.long)
        failures = torch.zeros((indices.shape[0], layout.packet_count), dtype=torch.bool, device=indices.device)
        recovered = torch.zeros_like(indices)
        for block in layout.blocks:
            packet = decoded[:, block.packet_id]
            valid = torch.tensor([check_crc16(row) for row in packet], device=indices.device, dtype=torch.bool)
            failures[:, block.packet_id] = ~valid
            payload = packet[:, :layout.payload_bits_per_packet].reshape(indices.shape[0], layout.block_frames, self.index_bit_width)
            values = bits_to_indices(payload)
            recovered[:, block.layer, block.frame_start:block.frame_stop] = values
        return DigitalTransportResult(recovered, qpsk, flat, failures, layout, interleaver, metadata)

    def noiseless_round_trip(self, indices: Tensor, repetition: Iterable[int], power_shares: Iterable[float], *, paired_seed: int) -> DigitalTransportResult:
        return self.transmit(indices, repetition, power_shares, paired_seed=paired_seed)


def candidate_hash(*, repetition: Iterable[int], power_shares: Iterable[float]) -> str:
    payload = repr((tuple(repetition), tuple(float(value) for value in power_shares))).encode()
    return hashlib.sha256(payload).hexdigest()
