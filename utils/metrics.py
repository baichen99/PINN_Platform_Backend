import torch

def cal_l2_relative_err(pred: torch.tensor, true: torch.tensor) -> list:
    assert pred.size() == true.size()
    numerator = torch.sqrt(torch.sum((pred - true)**2, dim=0, keepdim=True))
    dominator = torch.sqrt(torch.sum(true**2, dim=0, keepdim=True))
    return (numerator / dominator).cpu().flatten().tolist()