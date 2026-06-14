"""Spatial patch encoder, causal cross-attention decoder and token loss."""

from __future__ import annotations
from dataclasses import dataclass
import math
from .model import MAX_TOKENS, VOCABULARY, _build_model, _torch


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "patch_transformer"
    d_model: int = 64
    heads: int = 4
    encoder_layers: int = 1
    decoder_layers: int = 2
    feedforward: int = 128
    patch_size: int = 4
    dropout: float = 0.1

    def __post_init__(self):
        if self.architecture not in {"patch_transformer", "cnn_gru"}:
            raise ValueError("architecture must be patch_transformer or cnn_gru")
        integers = (
            self.d_model,
            self.heads,
            self.encoder_layers,
            self.decoder_layers,
            self.feedforward,
            self.patch_size,
        )
        if any(type(value) is not int for value in integers):
            raise ValueError("model dimensions must be integers")
        if not 16 <= self.d_model <= 256 or not 1 <= self.heads <= 8 or self.d_model % self.heads:
            raise ValueError("d_model must be 16..256 and divisible by 1..8 heads")
        if (
            not 1 <= self.encoder_layers <= 4
            or not 1 <= self.decoder_layers <= 4
            or not self.d_model <= self.feedforward <= 1024
        ):
            raise ValueError("attention depth or feed-forward budget exceeded")
        if self.patch_size not in {4, 8, 16}:
            raise ValueError("patch_size must be 4, 8, or 16")
        if type(self.dropout) not in (float, int) or not math.isfinite(self.dropout) or not 0 <= self.dropout < 0.5:
            raise ValueError("dropout must be finite in [0, .5)")


def build_model(config: ModelConfig):
    torch = _torch()
    if config.architecture == "cnn_gru":
        return _build_model(config.d_model)
    nn = torch.nn

    class PatchProgramTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            grid = 32 // config.patch_size
            self.patch_embedding = nn.Conv2d(3, config.d_model, config.patch_size, stride=config.patch_size)
            self.row_embedding = nn.Embedding(grid, config.d_model)
            self.column_embedding = nn.Embedding(grid, config.d_model)
            encoder = nn.TransformerEncoderLayer(
                config.d_model, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True
            )
            self.visual_encoder = nn.TransformerEncoder(
                encoder, config.encoder_layers, norm=nn.LayerNorm(config.d_model), enable_nested_tensor=False
            )
            self.token_embedding = nn.Embedding(len(VOCABULARY), config.d_model, padding_idx=0)
            self.position_embedding = nn.Embedding(MAX_TOKENS + 1, config.d_model)
            decoder = nn.TransformerDecoderLayer(
                config.d_model, config.heads, config.feedforward, config.dropout, batch_first=True, norm_first=True
            )
            self.program_decoder = nn.TransformerDecoder(
                decoder, config.decoder_layers, norm=nn.LayerNorm(config.d_model)
            )
            self.output = nn.Linear(config.d_model, len(VOCABULARY), bias=False)
            self.output.weight = self.token_embedding.weight
            # Encoder/decoder stacks clone layers; initialize each weight
            # independently rather than leaving depth-wise identical matrices.
            for stack in (self.visual_encoder, self.program_decoder):
                for parameter in stack.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)
            nn.init.normal_(self.token_embedding.weight, std=config.d_model**-0.5)
            with torch.no_grad():
                self.token_embedding.weight[0].zero_()
            for embedding in (self.row_embedding, self.column_embedding, self.position_embedding):
                nn.init.normal_(embedding.weight, std=0.02)

        def encode_images(self, images):
            if (
                images.ndim != 4
                or tuple(images.shape[1:]) != (3, 32, 32)
                or not images.is_floating_point()
                or not bool(torch.isfinite(images).all())
            ):
                raise ValueError("images must be finite floating-point Bx3x32x32 tensors")
            if bool((images < 0).any()) or bool((images > 1).any()) or images.shape[0] < 1:
                raise ValueError("image values must be in [0, 1] and batch must be nonempty")
            patches = self.patch_embedding(images)
            grid = patches.shape[-1]
            positions = (
                self.row_embedding(torch.arange(grid, device=images.device))[:, None, :]
                + self.column_embedding(torch.arange(grid, device=images.device))[None, :, :]
            )
            memory = patches.flatten(2).transpose(1, 2) + positions.reshape(1, grid * grid, self.config.d_model)
            return self.visual_encoder(memory)

        def decode_prefix(self, memory, input_ids):
            if (
                input_ids.ndim != 2
                or not 1 <= input_ids.shape[1] <= MAX_TOKENS
                or input_ids.shape[0] != memory.shape[0]
                or input_ids.dtype != torch.long
            ):
                raise ValueError("decoder requires aligned BxT long IDs with T in [1, 32]")
            if bool((input_ids < 0).any()) or bool((input_ids >= len(VOCABULARY)).any()):
                raise ValueError("decoder ID outside vocabulary")
            length = input_ids.shape[1]
            positions = torch.arange(length, device=input_ids.device)
            target = (
                self.token_embedding(input_ids) * math.sqrt(self.config.d_model)
                + self.position_embedding(positions)[None, :, :]
            )
            causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=input_ids.device), diagonal=1)
            decoded = self.program_decoder(target, memory, tgt_mask=causal, tgt_key_padding_mask=input_ids.eq(0))
            return self.output(decoded)

        def forward(self, images, input_ids):
            return self.decode_prefix(self.encode_images(images), input_ids)

    return PatchProgramTransformer()


def token_loss(logits, targets, *, smoothing: float = 0.0):
    torch = _torch()
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or logits.shape[-1] != len(VOCABULARY):
        raise ValueError("loss needs aligned BxTxV logits and BxT targets")
    if type(smoothing) not in (int, float) or not math.isfinite(smoothing) or not 0 <= smoothing <= 0.2:
        raise ValueError("label smoothing must be finite in [0, .2]")
    count = int(targets.ne(0).sum())
    if count == 0:
        raise ValueError("batch has no non-padding targets")
    summed = torch.nn.functional.cross_entropy(
        logits.reshape(-1, len(VOCABULARY)),
        targets.reshape(-1),
        ignore_index=0,
        reduction="sum",
        label_smoothing=smoothing,
    )
    if not bool(torch.isfinite(summed)):
        raise ValueError("nonfinite token loss")
    return summed, count
