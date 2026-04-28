import dataclasses
from typing import Any, Dict, Optional, Tuple

import numpy as np
import einops

from openpi import transforms
from openpi.models import model as _model


def _to_numpy_image(x: Any) -> np.ndarray:
    """Equivalent to your ImagesToNumpy._to_numpy (plus strict HWC output)."""
    # torch -> numpy
    try:
        import torch
        if torch.is_tensor(x):
            x = x.detach().cpu()
            # TCHW -> THWC
            if x.ndim == 4 and x.shape[1] in (1, 3, 4):
                x = x.permute(0, 2, 3, 1)
            # CHW -> HWC
            if x.ndim == 3 and x.shape[0] in (1, 3, 4):
                x = x.permute(1, 2, 0)
            x = x.numpy()
    except Exception:
        pass

    x = np.asarray(x)

    # dtype handling -> uint8
    if x.dtype.kind == "f":
        mx = float(np.nanmax(x)) if x.size else 0.0
        if mx <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    elif x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)

    # Ensure HWC if single image
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = einops.rearrange(x, "c h w -> h w c")

    return x


def _pad_last_dim_keep_type(x: Any, target_dim: int, pad_value: float = 0.0):
    """Equivalent to your PadLastDim (keeps torch dtype/device if x is torch)."""
    d = int(x.shape[-1])
    if d == target_dim:
        return x
    if d > target_dim:
        return x[..., :target_dim]

    pad = target_dim - d

    # torch path
    try:
        import torch
        if torch.is_tensor(x):
            pad_tensor = torch.full((*x.shape[:-1], pad), pad_value, dtype=x.dtype, device=x.device)
            return torch.cat([x, pad_tensor], dim=-1)
    except Exception:
        pass

    # numpy path
    pad_arr = np.full((*x.shape[:-1], pad), pad_value, dtype=x.dtype)
    return np.concatenate([x, pad_arr], axis=-1)


@dataclasses.dataclass(frozen=True)
class MyLeRobotInputs(transforms.DataTransformFn):
    """
    Strict replacement of:
      - ImagesToNumpy
      - InjectImageMask
      - PadLastDim('state', 32)
      - PadLastDim('action'/'action', 32)
    """
    model_type: _model.ModelType
    target_dim: int = 32
    image_keys: Tuple[str, ...] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    def __call__(self, data: dict) -> dict:
        # state
        if "state" in data:
            state = data["state"]
        elif "observation/state" in data:
            state = data["observation/state"]
        elif "observation.state" in data:
            state = data["observation.state"]
        else:
            raise KeyError("Missing state (expected: state / observation.state / observation/state).")

        state = _pad_last_dim_keep_type(state, self.target_dim, 0.0)

        # images dict (already repacked in your DataConfig)
        img = data.get("image", None)
        if not isinstance(img, dict):
            raise KeyError("Missing image dict at data['image'].")

        # convert images to numpy uint8 + build mask (fixed keys, missing->False)
        new_img: Dict[str, Any] = {}
        mask: Dict[str, bool] = {}

        for k in self.image_keys:
            present = (k in img) and (img[k] is not None)
            mask[k] = bool(present)
            if present:
                new_img[k] = _to_numpy_image(img[k])
            else:
                # keep it None (do NOT pad image here) to match your original InjectImageMask semantics
                new_img[k] = None

        inputs = {
            "state": state,
            "image": new_img,
            "image_mask": mask,
        }

        # action (training): accept action or action
        action = data.get("action", None)
        if action is None:
            action = data.get("action", None)
        if action is not None:
            action = _pad_last_dim_keep_type(action, self.target_dim, 0.0)
            inputs["action"] = action

        # prompt passthrough
        if "prompt" in data and data["prompt"] is not None:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class MyLeRobotOutputs(transforms.DataTransformFn):
    action_dim: int 

    def __call__(self, data: dict) -> dict:
        if "action" not in data:
            raise KeyError("Model output missing 'action'.")
        a = np.asarray(data["action"])
        if a.ndim == 1:
            return {"action": a[: self.action_dim]}
        return {"action": a[..., : self.action_dim]}
