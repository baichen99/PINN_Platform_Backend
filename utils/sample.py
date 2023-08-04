import torch

def uniform_sample(lower_bounds: list, upper_bounds: list, sample_num: int) -> torch.Tensor:
    assert len(lower_bounds) == len(upper_bounds)
    dim = len(lower_bounds)
    Xs = torch.zeros(sample_num, dim)
    for i in range(dim):
        Xs[:, i] = torch.rand(sample_num) * (upper_bounds[i] - lower_bounds[i]) + lower_bounds[i]
    return Xs