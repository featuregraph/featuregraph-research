import torch

def start(mask):
    mask = mask.bool()
    return torch.cat([
        torch.zeros(1, dtype=torch.bool, device=mask.device),
        (~mask[:-1]) & mask[1:]
    ])

def end(mask):
    mask = mask.bool()
    return torch.cat([
        torch.zeros(1, dtype=torch.bool, device=mask.device),
        mask[:-1] & (~mask[1:])
    ])

def between(start, end):
    return (start.cumsum(dim=0) - end.int().cumsum(dim=0)) > 0


operators = {
    'start': start,
    'end': end
}