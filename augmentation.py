import torch
import torch.nn.functional as F


def random_neighbor_patch_swap(
    images: torch.Tensor,
    patch_size: int = 16,
    swap_prob: float = 0.2,
):
    B, C, H, W = images.shape

    assert H % patch_size == 0
    assert W % patch_size == 0

    Hp = H // patch_size
    Wp = W // patch_size

    patches = F.unfold(
        images,
        kernel_size=patch_size,
        stride=patch_size,
    )

    patches = patches.transpose(1, 2)
    N = Hp * Wp
    patches = patches.reshape(B, Hp, Wp, -1)

    out = patches.clone()

    neighbors = [
        (-1, -1),
        (-1,  0),
        (-1,  1),
        ( 0, -1),
        ( 0,  1),
        ( 1, -1),
        ( 1,  0),
        ( 1,  1),
    ]

    for b in range(B):

        mask = torch.rand(Hp, Wp, device=images.device) < swap_prob[b]
        directions = torch.randint(
            0,
            8,
            (Hp, Wp),
            device=images.device,
        )
        for i in range(Hp):
            for j in range(Wp):

                if not mask[i, j]:
                    continue
                di, dj = neighbors[directions[i, j]]
                ni = i + di
                nj = j + dj
                if ni < 0 or ni >= Hp:
                    continue

                if nj < 0 or nj >= Wp:
                    continue

                tmp = out[b, i, j].clone()
                out[b, i, j] = out[b, ni, nj]
                out[b, ni, nj] = tmp

    out = out.reshape(B, N, -1)
    out = out.transpose(1, 2)
    out = F.fold(
        out,
        output_size=(H, W),
        kernel_size=patch_size,
        stride=patch_size,
    )

    return out