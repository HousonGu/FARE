"""Exact cross-series top-K over the frozen fit bank, bounded device memory."""
import numpy as np
import torch
from torch.nn import functional as F

def normalize_windows(x, y):
    mean = x.mean(1, keepdims=True)
    std = np.sqrt(x.var(1, keepdims=True) + 1e-5)
    return (x-mean)/std, (y-mean)/std

@torch.no_grad()
def search(x, keys, values, k, block, device, tick=lambda: None):
    if len(keys) < k:
        raise ValueError('Frozen bank smaller than K; do not silently pad')
    q = F.normalize(torch.as_tensor(x, device=device), dim=1)
    best = torch.empty((len(x), 0), device=device)
    ids = torch.empty((len(x), 0), dtype=torch.long, device=device)
    for start in range(0, len(keys), block):
        tick()
        key = F.normalize(torch.tensor(np.asarray(keys[start:start+block]), device=device), dim=1)
        sims = q @ key.T
        # Stable tie rule: lower bank row first. Old best is already sorted;
        # new block rows are strictly later than previous block rows.
        candidate = torch.cat((best, sims), dim=1)
        candidate_ids = torch.cat((ids, torch.arange(start, start+len(key), device=device).expand(len(x), -1)), dim=1)
        order = torch.argsort(candidate, dim=1, descending=True, stable=True)[:, :k]
        best = candidate.gather(1, order)
        ids = candidate_ids.gather(1, order)
    val = torch.tensor(np.asarray(values[ids.cpu().numpy()]), device=device)
    result = (torch.softmax(best, dim=1)[..., None] * val).sum(1)
    return result.cpu().numpy(), best.cpu().numpy(), ids.cpu().numpy()
