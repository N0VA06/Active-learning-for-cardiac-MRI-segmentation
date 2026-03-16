"""
models/early_exit_unet.py
=========================
Non-invasive Early Exit wrapper for MONAI AttentionUNet.

Design
------
All ``ConvTranspose2d`` modules in the base model are the decoder
upsamplers (MONAI uses strided Conv2d for downsampling, so every
ConvTranspose2d is strictly a decoder layer).  We sort them by
*out_channels* descending so index 0 = deepest (bottleneck-adjacent)
decoder layer regardless of PyTorch DFS ordering.

``exit_indices`` picks which of those layers to attach lightweight
segmentation heads to.  With channels=[32,64,128,256,320] and
exit_indices=(1,2) the hooks land on the 128-ch (~40×40) and
64-ch (~80×80) decoder outputs — mid-resolution features that are
semantically rich but structurally uncertain.

Two operating modes (controlled by ``set_exit_mode``)
-----------------------------------------------------
* False (default) – ``forward`` returns ``final_logits`` only.
  Fully backward-compatible; existing train / eval code unchanged.

* True – ``forward`` returns ``(final_logits, [exit_logits, ...])``
  where exits are ordered shallow→deep.  Used in uncertainty.py
  and training auxiliary losses.

Gradient flow during training
------------------------------
In ``model.train()`` the hooks store un-detached tensors so gradients
propagate: exit-loss → exit-head → captured ConvTranspose2d output →
earlier encoder/decoder layers.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Exit head
# ──────────────────────────────────────────────────────────────────────────────

class EarlyExitHead(nn.Module):
    """
    Lightweight per-pixel classification head for an intermediate
    decoder feature map.

    Architecture
    ------------
    3×3 conv (in_ch → mid_ch)  →  InstanceNorm2d  →  LeakyReLU
    →  1×1 conv (mid_ch → out_ch)

    The spatial size is adjusted with bilinear upsampling before the
    convolutions so the output always matches the full input resolution.

    Parameters
    ----------
    in_channels  : channels of the hooked ConvTranspose2d output
    out_channels : number of segmentation classes
    target_size  : expected output spatial size (default 160 for ACDC)
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        target_size:  int = 160,
    ) -> None:
        super().__init__()
        mid = max(in_channels // 2, 16)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=1),
        )
        self.target_size = target_size

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        if h != self.target_size or w != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(x)


# ──────────────────────────────────────────────────────────────────────────────
# Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class EarlyExitWrapper(nn.Module):
    """
    Wraps any MONAI UNet-family model and injects early-exit heads via
    forward hooks on selected ConvTranspose2d decoder layers.

    Parameters
    ----------
    base_model   : pre-built MONAI AttentionUNet / UNet
    out_channels : number of segmentation classes
    exit_indices : indices into the sorted ConvTranspose2d list
                   (0 = deepest decoder layer).  Default (1, 2) hooks
                   the 2nd and 3rd deepest layers.
    target_size  : full-resolution spatial size for exit head output
    """

    def __init__(
        self,
        base_model:   nn.Module,
        out_channels: int,
        exit_indices: Tuple[int, ...] = (1, 2),
        target_size:  int = 160,
    ) -> None:
        super().__init__()
        self.base          = base_model
        self.out_channels  = out_channels
        self._return_exits = False

        # ── Discover and sort decoder ConvTranspose2d layers ──────────────
        ct_layers: List[Tuple[str, nn.ConvTranspose2d]] = [
            (name, mod)
            for name, mod in base_model.named_modules()
            if isinstance(mod, nn.ConvTranspose2d)
        ]
        # Deepest first (largest out_channels)
        ct_layers.sort(key=lambda x: -x[1].out_channels)

        n_found = len(ct_layers)
        if n_found == 0:
            raise RuntimeError(
                "No ConvTranspose2d layers found in base model. "
                "EarlyExitWrapper requires a decoder that uses transposed convolutions."
            )
        if max(exit_indices) >= n_found:
            raise ValueError(
                f"exit_indices={exit_indices} requires at least "
                f"{max(exit_indices) + 1} ConvTranspose2d layers; "
                f"found only {n_found}."
            )

        self._hook_targets = [ct_layers[i] for i in exit_indices]

        # ── Build exit heads ──────────────────────────────────────────────
        self.exit_heads = nn.ModuleList([
            EarlyExitHead(
                in_channels  = mod.out_channels,
                out_channels = out_channels,
                target_size  = target_size,
            )
            for _, mod in self._hook_targets
        ])

        # ── Feature capture buffers ───────────────────────────────────────
        self._n_exits   = len(exit_indices)
        self._captured: List[Optional[Tensor]] = [None] * self._n_exits

        # ── Register persistent hooks ─────────────────────────────────────
        self._handles: List = []
        for i, (_, mod) in enumerate(self._hook_targets):
            self._handles.append(
                mod.register_forward_hook(self._make_capture_hook(i))
            )

    # ── Hook factory ──────────────────────────────────────────────────────

    def _make_capture_hook(self, idx: int):
        def _hook(_module: nn.Module, _inp, out: Tensor) -> None:
            # Keep grad in training mode; detach in eval to save memory
            self._captured[idx] = out if self.training else out.detach()
        return _hook

    # ── Public helpers ────────────────────────────────────────────────────

    def remove_hooks(self) -> None:
        """Permanently remove all registered hooks (call before pickling)."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def set_exit_mode(self, enabled: bool) -> None:
        """
        Switch return type of forward().
        False (default): returns ``final_logits`` — backward-compatible.
        True           : returns ``(final_logits, [exit1_logits, ...])``
        """
        self._return_exits = enabled

    def enable_mc_dropout(self) -> None:
        """Put all Dropout layers in train() mode for MC-Dropout inference."""
        self.eval()
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train()

    def disable_mc_dropout(self) -> None:
        self.eval()

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: Tensor):
        # Reset capture buffers each pass
        self._captured = [None] * self._n_exits

        final_logits = self.base(x)

        if self._return_exits:
            exit_logits = [
                head(feat)
                for head, feat in zip(self.exit_heads, self._captured)
                if feat is not None
            ]
            return final_logits, exit_logits

        return final_logits


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_early_exit_model(cfg: dict, device: torch.device) -> EarlyExitWrapper:
    """
    Build an ``EarlyExitWrapper`` from the project config.

    Reads from ``cfg["early_exit"]``:
        exit_indices : list[int], default [1, 2]

    Falls back gracefully: if the key is absent, uses defaults.
    """
    from models.unet_model import build_model

    ee_cfg     = cfg.get("early_exit", {})
    ex_idx     = tuple(ee_cfg.get("exit_indices", [1, 2]))
    tgt_size   = cfg["preprocessing"]["spatial_size"][0]

    base_model = build_model(cfg, device)

    wrapper = EarlyExitWrapper(
        base_model   = base_model,
        out_channels = cfg["model"]["out_channels"],
        exit_indices = ex_idx,
        target_size  = tgt_size,
    ).to(device)

    n_base  = sum(p.numel() for p in base_model.parameters())
    n_exits = sum(p.numel() for p in wrapper.exit_heads.parameters())
    hooks   = [(name, mod.out_channels)
               for name, mod in wrapper._hook_targets]

    print(
        f"[EarlyExit] Base params: {n_base:,}  |  "
        f"Exit-head params: {n_exits:,}  "
        f"({100*n_exits/(n_base+n_exits):.1f}% overhead)\n"
        f"[EarlyExit] Hooked layers: "
        + ", ".join(f"'{n}' (out_ch={c})" for n, c in hooks)
    )
    return wrapper