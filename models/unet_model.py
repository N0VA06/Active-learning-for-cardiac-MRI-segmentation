"""
unet_model.py
─────────────
Segmentation model with two architecture options:

  "attention_unet" (default)
      MONAI AttentionUnet – adds gating signals at skip connections so the
      decoder focuses on the cardiac region; better for small structures
      (RV, Myocardium) than a plain UNet.

  "unet"
      Standard MONAI residual UNet (kept as fallback).

Both wrapped with MCDropoutUNet for MC-Dropout uncertainty estimation.
"""
from __future__ import annotations
from typing import Sequence

import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet, UNet


class MCDropoutUNet(nn.Module):
    """
    Wrapper that adds MC-Dropout support to any MONAI encoder-decoder.

    enable_mc_dropout()  – forces all Dropout layers to stay active in eval.
    disable_mc_dropout() – restores normal eval behaviour.
    """

    def __init__(
        self,
        in_channels:   int = 1,
        out_channels:  int = 4,
        channels:      Sequence[int] = (32, 64, 128, 256, 320),
        strides:       Sequence[int] = (2, 2, 2, 2),
        num_res_units: int = 3,
        dropout_prob:  float = 0.15,
        architecture:  str = "attention_unet",
    ) -> None:
        super().__init__()
        self._mc_mode = False

        if architecture == "attention_unet":
            # AttentionUnet does not have num_res_units; uses dropout param
            self.net = AttentionUnet(
                spatial_dims = 2,
                in_channels  = in_channels,
                out_channels = out_channels,
                channels     = channels,
                strides      = strides,
                dropout      = dropout_prob,
            )
        else:
            self.net = UNet(
                spatial_dims  = 2,
                in_channels   = in_channels,
                out_channels  = out_channels,
                channels      = channels,
                strides       = strides,
                num_res_units = num_res_units,
                dropout       = dropout_prob,
            )

    # ── MC helpers ────────────────────────────────────────────────────────────

    def enable_mc_dropout(self) -> None:
        self._mc_mode = True
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def disable_mc_dropout(self) -> None:
        self._mc_mode = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._mc_mode:
            for m in self.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    m.train()
        return self.net(x)


def build_model(cfg: dict, device: torch.device) -> MCDropoutUNet:
    """Instantiate MCDropoutUNet from config and move to device."""
    mc = cfg["model"]
    model = MCDropoutUNet(
        in_channels   = mc["in_channels"],
        out_channels  = mc["out_channels"],
        channels      = mc["channels"],
        strides       = mc["strides"],
        num_res_units = mc.get("num_res_units", 3),
        dropout_prob  = mc["dropout_prob"],
        architecture  = mc.get("architecture", "attention_unet"),
    )
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {mc.get('architecture','attention_unet')}  "
          f"| Params: {total/1e6:.2f}M")
    return model.to(device)
