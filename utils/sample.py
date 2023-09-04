import torch

def uniform_sample(lower_bounds: list, upper_bounds: list, sample_num: int) -> torch.Tensor:
    assert len(lower_bounds) == len(upper_bounds)
    dim = len(lower_bounds)
    Xs = []
    for i in range(dim):
        X = torch.rand(sample_num) * (upper_bounds[i] - lower_bounds[i]) + lower_bounds[i]
        Xs.append(X.reshape(-1, 1))
    return Xs