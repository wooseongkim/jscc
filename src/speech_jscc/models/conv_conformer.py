from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.jscc_encoder import deterministic_layer_gates, normalize_complex_power


def balanced_ragged_valid_mask(*, frames: int, max_symbols: int,
                               valid_symbols: int) -> Tensor:
    """Return a deterministic, temporally balanced fixed-width validity mask."""
    total_slots = int(frames) * int(max_symbols)
    missing = total_slots - int(valid_symbols)
    if frames <= 0 or max_symbols <= 0 or missing < 0 or missing > frames:
        raise ValueError("balanced ragged layout requires at most one missing symbol per frame")
    mask = torch.ones(frames, max_symbols, dtype=torch.bool)
    if missing:
        short_frames = torch.floor(
            (torch.arange(missing, dtype=torch.float64) + 0.5) * frames / missing
        ).to(torch.long)
        if torch.unique(short_frames).numel() != missing:
            raise ValueError("cannot distribute short frames uniquely")
        mask[short_frames, -1] = False
    return mask


def pack_valid_symbols(value: Tensor, valid_mask: Tensor) -> Tensor:
    if value.shape[-2:] != valid_mask.shape:
        raise ValueError("fixed-width symbols and valid mask shapes do not match")
    return value[..., valid_mask]


def unpack_valid_symbols(value: Tensor, valid_mask: Tensor) -> Tensor:
    if value.shape[-1] != int(valid_mask.sum()):
        raise ValueError("packed symbol count does not match valid mask")
    output = value.new_zeros(*value.shape[:-1], *valid_mask.shape)
    output[..., valid_mask] = value
    return output


def masked_complex_power_normalize(value: Tensor, valid_mask: Tensor,
                                   target_power: float) -> Tensor:
    if not torch.is_complex(value):
        raise ValueError("masked power normalization requires complex symbols")
    valid = pack_valid_symbols(value, valid_mask)
    power = valid.abs().square().mean(dim=-1, keepdim=True).clamp_min(1e-12)
    normalized = valid * (float(target_power) / power).sqrt()
    return unpack_valid_symbols(normalized, valid_mask)


class FeedForwardModule(nn.Module):
    def __init__(self, d_model: int, expansion: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * expansion),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(d_model * expansion, d_model), nn.Dropout(dropout))

    def forward(self, value: Tensor) -> Tensor: return self.network(value)


class ConvolutionModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 != 1: raise ValueError("convolution kernel size must be odd")
        self.norm = nn.LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.channel_norm = nn.GroupNorm(1, d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: Tensor) -> Tensor:
        value = self.norm(value).transpose(1, 2)
        value = F.glu(self.pointwise_in(value), dim=1)
        value = F.silu(self.channel_norm(self.depthwise(value)))
        return self.dropout(self.pointwise_out(value).transpose(1, 2))


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_expansion: int, kernel_size: int, dropout: float):
        super().__init__()
        if d_model % heads: raise ValueError("d_model must be divisible by attention heads")
        self.ff1 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.convolution = ConvolutionModule(d_model, kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ffn_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, value: Tensor) -> Tensor:
        value = value + .5 * self.ff1(value)
        normalized = self.attn_norm(value)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        value = value + self.attn_dropout(attended)
        value = value + self.convolution(value)
        return self.final_norm(value + .5 * self.ff2(value))


class LayerMixer(nn.Module):
    def __init__(self, d_model: int, heads: int, blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(d_model, heads, 2 * d_model,
            dropout, activation="gelu", batch_first=True, norm_first=True) for _ in range(blocks))

    def forward(self, value: Tensor) -> Tensor:
        batch, layers, frames, width = value.shape
        mixed = value.permute(0, 2, 1, 3).reshape(batch * frames, layers, width)
        for block in self.blocks: mixed = block(mixed)
        return mixed.reshape(batch, frames, layers, width).permute(0, 2, 1, 3)


class StateFiLM(nn.Module):
    def __init__(self, state_dim: int, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.network = nn.Sequential(nn.Linear(state_dim, d_model), nn.SiLU(), nn.Linear(d_model, 2 * d_model))

    def forward(self, value: Tensor, state: Tensor) -> Tensor:
        scale, shift = self.network(state).chunk(2, dim=-1)
        return self.norm(value) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]


class ConvConformerEncoder(nn.Module):
    def __init__(self, shape, channel_uses, channel_state_dim, target_power, *, d_model, blocks, heads,
                 expansion, kernel, dropout, mixer_blocks, symbol_frames, complex_channels,
                 temporal_symbol_layout):
        super().__init__(); self.representation_shape=tuple(shape); self.num_layers,self.frames,self.latent_dim=self.representation_shape
        self.total_channel_uses=int(channel_uses); self.channel_shape=(self.total_channel_uses,); self.channel_state_dim=channel_state_dim
        self.target_power=float(target_power); self.symbol_frames=symbol_frames; self.complex_channels_per_symbol_frame=complex_channels
        self.temporal_symbol_layout=str(temporal_symbol_layout)
        if self.total_channel_uses % self.num_layers: raise ValueError("channel uses must divide equally across layers")
        uses=self.total_channel_uses//self.num_layers; self.layer_channel_uses=(uses,)*self.num_layers
        if self.temporal_symbol_layout == "balanced_ragged":
            if symbol_frames != self.frames:
                raise ValueError("balanced ragged layout must preserve source temporal positions")
            valid_mask=balanced_ragged_valid_mask(frames=symbol_frames,max_symbols=complex_channels,valid_symbols=uses)
        elif self.temporal_symbol_layout == "dense_interpolate":
            if symbol_frames * complex_channels != uses: raise ValueError(f"symbol_frames * complex channels must equal {uses}")
            valid_mask=torch.ones(symbol_frames,complex_channels,dtype=torch.bool)
        else:
            raise ValueError("temporal_symbol_layout must be dense_interpolate or balanced_ragged")
        self.register_buffer("symbol_valid_mask",valid_mask,persistent=False)
        self.uses_temporal_interpolation = self.symbol_frames != self.frames
        self.input_norm=nn.LayerNorm(self.latent_dim); self.input_projection=nn.Linear(self.latent_dim,d_model)
        self.layer_embedding=nn.Parameter(torch.zeros(self.num_layers,d_model)); nn.init.normal_(self.layer_embedding,std=.02)
        self.state_conditioner=StateFiLM(channel_state_dim,d_model)
        self.local_conv=nn.Conv1d(d_model,d_model,5,padding=2,groups=d_model)
        self.conformer=nn.ModuleList(ConformerBlock(d_model,heads,expansion,kernel,dropout) for _ in range(blocks))
        self.layer_mixer=LayerMixer(d_model,heads,mixer_blocks,dropout)
        self.resample_conv=nn.Conv1d(d_model,d_model,3,padding=1)
        self.symbol_heads=nn.ModuleList(nn.Linear(d_model,2*complex_channels) for _ in range(self.num_layers))
        self.register_buffer("default_layer_power",torch.ones(self.num_layers))

    def forward(self, representation, channel_state, *, layer_gates=None, layer_power_allocation=None, return_aux=False):
        if tuple(representation.shape[1:]) != self.representation_shape: raise ValueError("representation shape mismatch")
        batch=representation.shape[0]
        if channel_state.shape != (batch,self.channel_state_dim): raise ValueError("channel_state shape mismatch")
        gates=torch.ones(batch,self.num_layers,device=representation.device,dtype=representation.dtype) if layer_gates is None else layer_gates.to(representation)
        value=representation*gates[:,:,None,None]
        value=self.input_projection(self.input_norm(value))+self.layer_embedding[None,:,None,:]
        value=self.state_conditioner(value,channel_state)
        flat=value.reshape(batch*self.num_layers,self.frames,-1)
        flat=flat+self.local_conv(flat.transpose(1,2)).transpose(1,2)
        for block in self.conformer: flat=block(flat)
        value=self.layer_mixer(flat.reshape(batch,self.num_layers,self.frames,-1))
        flat=value.reshape(batch*self.num_layers,self.frames,-1).transpose(1,2)
        if self.uses_temporal_interpolation:
            flat=F.interpolate(flat,size=self.symbol_frames,mode="linear",align_corners=False)
        value=self.resample_conv(flat).transpose(1,2).reshape(batch,self.num_layers,self.symbol_frames,-1)
        partitions=[]; fixed_partitions=[]
        allocation=self.default_layer_power if layer_power_allocation is None else layer_power_allocation
        if allocation.ndim==1: allocation=allocation[None,:].expand(batch,-1)
        weights=allocation.to(representation)*gates.square(); fractions=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
        for layer,head in enumerate(self.symbol_heads):
            ri=head(value[:,layer]).reshape(batch,self.symbol_frames,self.complex_channels_per_symbol_frame,2)
            fixed=torch.view_as_complex(ri.contiguous())
            if self.temporal_symbol_layout == "balanced_ragged":
                fixed=masked_complex_power_normalize(fixed,self.symbol_valid_mask,1.0)
                branch=pack_valid_symbols(fixed,self.symbol_valid_mask)
            else:
                branch=normalize_complex_power(fixed.reshape(batch,-1),1.0)
                fixed=branch.reshape(batch,self.symbol_frames,self.complex_channels_per_symbol_frame)
            branch_power=self.target_power*self.total_channel_uses*fractions[:,layer]/self.layer_channel_uses[layer]
            branch=branch*branch_power[:,None].sqrt()
            partitions.append(branch)
            fixed_partitions.append(
                unpack_valid_symbols(branch,self.symbol_valid_mask)
                if self.temporal_symbol_layout=="balanced_ragged"
                else branch.reshape(batch,self.symbol_frames,self.complex_channels_per_symbol_frame)
            )
        symbols=normalize_complex_power(torch.cat(partitions,1),self.target_power)
        if return_aux:
            return symbols,{"layer_gates":gates,"layer_power_fractions":fractions,"temporal_feature_shape":torch.tensor(value.shape,device=value.device),
                "per_layer_symbol_power":torch.stack([part.abs().square().mean(1) for part in partitions],1),
                "fixed_width_symbols":torch.stack(fixed_partitions,1),"symbol_valid_mask":self.symbol_valid_mask}
        return symbols


class ConvConformerDecoder(nn.Module):
    def __init__(self, shape, channel_uses, channel_state_dim, *, d_model, blocks, heads, expansion, kernel, dropout, mixer_blocks, symbol_frames, complex_channels,
                 temporal_symbol_layout):
        super().__init__(); self.representation_shape=tuple(shape); self.num_layers,self.frames,self.latent_dim=self.representation_shape
        self.total_channel_uses=int(channel_uses); self.channel_shape=(self.total_channel_uses,); self.channel_state_dim=channel_state_dim
        self.layer_channel_uses=(self.total_channel_uses//self.num_layers,)*self.num_layers; self.symbol_frames=symbol_frames; self.complex_channels_per_symbol_frame=complex_channels
        self.temporal_symbol_layout=str(temporal_symbol_layout)
        uses=self.layer_channel_uses[0]
        if self.temporal_symbol_layout=="balanced_ragged":
            if symbol_frames!=self.frames:raise ValueError("balanced ragged layout must preserve source temporal positions")
            valid_mask=balanced_ragged_valid_mask(frames=symbol_frames,max_symbols=complex_channels,valid_symbols=uses)
        elif self.temporal_symbol_layout=="dense_interpolate":
            if symbol_frames*complex_channels!=uses:raise ValueError(f"symbol_frames * complex channels must equal {uses}")
            valid_mask=torch.ones(symbol_frames,complex_channels,dtype=torch.bool)
        else:raise ValueError("temporal_symbol_layout must be dense_interpolate or balanced_ragged")
        self.register_buffer("symbol_valid_mask",valid_mask,persistent=False)
        self.uses_temporal_interpolation=self.symbol_frames!=self.frames
        self.input_projection=nn.Linear(2*complex_channels,d_model); self.layer_embedding=nn.Parameter(torch.zeros(self.num_layers,d_model)); nn.init.normal_(self.layer_embedding,std=.02)
        self.state_conditioner=StateFiLM(channel_state_dim,d_model)
        self.conformer=nn.ModuleList(ConformerBlock(d_model,heads,expansion,kernel,dropout) for _ in range(blocks))
        self.layer_mixer=LayerMixer(d_model,heads,mixer_blocks,dropout); self.resample_conv=nn.Conv1d(d_model,d_model,3,padding=1)
        self.reconstruction_heads=nn.ModuleList(nn.Linear(d_model,self.latent_dim) for _ in range(self.num_layers))

    def forward(self, received, channel_state, *, layer_gates=None):
        if not torch.is_complex(received) or received.shape[1:] != self.channel_shape: raise ValueError("received shape mismatch")
        batch=received.shape[0]; packed=received.reshape(batch,self.num_layers,-1)
        chunks=(unpack_valid_symbols(packed,self.symbol_valid_mask)
                if self.temporal_symbol_layout=="balanced_ragged"
                else packed.reshape(batch,self.num_layers,self.symbol_frames,self.complex_channels_per_symbol_frame))
        value=self.input_projection(torch.view_as_real(chunks).flatten(-2))+self.layer_embedding[None,:,None,:]
        value=self.state_conditioner(value,channel_state); flat=value.reshape(batch*self.num_layers,self.symbol_frames,-1)
        for block in self.conformer: flat=block(flat)
        value=self.layer_mixer(flat.reshape(batch,self.num_layers,self.symbol_frames,-1))
        flat=value.reshape(batch*self.num_layers,self.symbol_frames,-1).transpose(1,2)
        if self.uses_temporal_interpolation:
            flat=F.interpolate(flat,size=self.frames,mode="linear",align_corners=False)
        value=self.resample_conv(flat).transpose(1,2).reshape(batch,self.num_layers,self.frames,-1)
        output=torch.stack([head(value[:,layer]) for layer,head in enumerate(self.reconstruction_heads)],1)
        return output if layer_gates is None else output*layer_gates[:,:,None,None].to(output)


class ConvConformerJSCC(nn.Module):
    architecture="conv_conformer_v1"; architecture_version="conv_conformer_v1"
    def __init__(self, representation_shape=(8,50,1024), channel_uses=1920, channel_state_dim=8, target_power=1.0,
                 *, d_model=256, encoder_conformer_blocks=4, decoder_conformer_blocks=4, num_attention_heads=4,
                 ffn_expansion=4, convolution_kernel_size=15, dropout=.1, layer_mixer_blocks=1,
                 symbol_frames=30, complex_channels_per_symbol_frame=8,
                 temporal_symbol_layout="dense_interpolate"):
        super().__init__(); common=dict(d_model=d_model,heads=num_attention_heads,expansion=ffn_expansion,kernel=convolution_kernel_size,
            dropout=dropout,mixer_blocks=layer_mixer_blocks,symbol_frames=symbol_frames,complex_channels=complex_channels_per_symbol_frame,
            temporal_symbol_layout=temporal_symbol_layout)
        self.encoder=ConvConformerEncoder(representation_shape,channel_uses,channel_state_dim,target_power,blocks=encoder_conformer_blocks,**common)
        self.decoder=ConvConformerDecoder(representation_shape,channel_uses,channel_state_dim,blocks=decoder_conformer_blocks,**common)
        self.model_config={"d_model":d_model,"encoder_conformer_blocks":encoder_conformer_blocks,"decoder_conformer_blocks":decoder_conformer_blocks,
            "num_attention_heads":num_attention_heads,"ffn_expansion":ffn_expansion,"convolution_kernel_size":convolution_kernel_size,
            "dropout":dropout,"layer_mixer_blocks":layer_mixer_blocks,"symbol_frames":symbol_frames,"complex_channels_per_symbol_frame":complex_channels_per_symbol_frame}
        self.model_config["temporal_symbol_layout"]=temporal_symbol_layout

    def encode(self,representation,channel_state): return self.encoder(representation,channel_state)
    def decode(self,received,channel_state): return self.decoder(received,channel_state)
