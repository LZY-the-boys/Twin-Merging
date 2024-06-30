import torch


def magnitude(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor
    if len(tensor.shape) == 1:
        # rank=1
        return tensor
    
    k = int(density * tensor.view(-1).shape[0])

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.topk(w, k=k, largest=True)
    mask.view(-1)[topk.indices] = 1

    return tensor * mask


def bernoulli(
    tensor: torch.Tensor, 
    density: float, # 1 - mask_rate (probability of drawing "1")
    rescale: bool = True
) -> torch.Tensor:
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    if len(tensor.shape) == 1:
        # rank=1
        return tensor

    # mask = 1 - torch.bernoulli(
    #     torch.full_like(input=tensor, fill_value=1 - density)
    # )
    mask = torch.bernoulli(
        torch.full_like(input=tensor, fill_value=density).float()
    )

    res = tensor * mask
    if rescale:
        res *= 1 / density
    return res

def svd(
    tensor: torch.Tensor, 
    density: float,
    **kwargs,
):
    if density >= 1:
        return tensor
    if density <= 0:
        return torch.zeros_like(tensor)
    if kwargs.get('new_rank', None) == 0:
        return torch.zeros_like(tensor)
    if len(tensor.shape) == 1:
        # rank=1
        return tensor
    
    # U, S, V = torch.svd(tensor)
    # S = (S >= S[int(len(S) * density)]) * S
    # res = U @ torch.diag(S) @ V.T

    # `torch.linalg.svd()`: good for dense matrix
    # `torch.svd()`: deprecated
    # `torch.svd_lowrank()`: good for huge sparse matrix
    driver = None
    if tensor.is_cuda:
        driver = 'gesvda'
    
    U, S, Vh = torch.linalg.svd(tensor, full_matrices=True, driver=driver)
    if 'new_rank' not in kwargs:
        new_rank = int(density * len(S))
    else:
        new_rank = kwargs['new_rank']
    U, S, Vh = U[:, :new_rank], S[:new_rank], Vh[:new_rank, :]
    res = U @ torch.diag(S) @ Vh
    return res