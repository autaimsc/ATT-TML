import torch
import torch.nn.functional as F


def feature_map_to_patch_importance(
    importance_map: torch.Tensor,
    patch_size: int = 16,
    normalize: bool = True,
    eps: float = 1e-8,
):

    if importance_map.dim() == 3:
        importance_map = importance_map.unsqueeze(1)

    B, C, H, W = importance_map.shape

    assert C == 1, "importance map must have one channel."
    assert H % patch_size == 0
    assert W % patch_size == 0

    Hp = H // patch_size
    Wp = W // patch_size

    patches = F.unfold(
        importance_map,
        kernel_size=patch_size,
        stride=patch_size,
    )

    patches = patches.transpose(1, 2)
    patch_importance = patches.mean(dim=-1)
    patch_importance = patch_importance.view(B, Hp, Wp)

    if normalize:
        flat = patch_importance.view(B, -1)
        min_val = flat.min(dim=1, keepdim=True).values
        max_val = flat.max(dim=1, keepdim=True).values

        flat = (flat - min_val) / (max_val - min_val + eps)
        patch_importance = flat.view(B, Hp, Wp)

    return patch_importance