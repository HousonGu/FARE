"""Execute unchanged author classes; inject frozen forecasts and exact retrieval.

AST extraction avoids importing unused backbone/LSH libraries. Model.forward and
the quality estimator, Transformer and residual equations remain author code.
"""
import ast
import time
from math import sqrt
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

AUTHOR = Path(__file__).resolve().parents[2] / 'third_party' / 'PIR'

def classes(relative, names, scope):
    path = AUTHOR / relative
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in tree.body} != set(names):
        raise ValueError('Author classes missing')
    exec(compile(tree, str(path), 'exec'), scope)

scope = dict(torch=torch, nn=nn, F=F, np=np, sqrt=sqrt, time=time)
classes('layers/SelfAttention_Family.py', ['FullAttention', 'AttentionLayer'], scope)
classes('layers/Transformer_EncDec.py', ['Encoder', 'EncoderLayer'], scope)
classes('models/PIR.py', ['QualityEstimator', 'Model'], scope)

class FrozenForecast(nn.Module):
    def forward(self, *args):
        return self.prediction

class RetailPIR(scope['Model']):
    def __init__(self, c):
        cfg = SimpleNamespace(seq_len=56, pred_len=7, enc_in=1,
            including_time_features=True, retrieval_stride=7, use_norm=True,
            refine_d_model=c['hidden'], refine_d_ff=c['hidden'], refine_layers=1,
            factor=5, dropout=c['dropout'], output_attention=False,
            n_heads=c['heads'], activation='gelu', retrieval_num=c['retrieval_k'])
        super().__init__(cfg)

    def _build_model(self):
        self.model = FrozenForecast()

    def retrieval(self, x, index):
        return self.cached_global, self.cached_sims, 0

    def revise(self, x, backbone, global_normalized, similarities, marks):
        # All tensors in caller's historical-mean scale. No target argument.
        means = x.mean(1, keepdim=True)
        std = torch.sqrt(x.var(1, unbiased=False, keepdim=True) + 1e-5)
        self.model.prediction = ((backbone - means) / std).detach()
        self.cached_global = global_normalized.detach()
        self.cached_sims = similarities.detach()
        return super().forward(x, None, torch.zeros_like(backbone), marks, mode='refine')

def objective(output, target, weight=1.0):
    prediction, (backbone, estimated) = output
    error = (backbone - target).square().detach().mean(1).unsqueeze(-1)
    return F.mse_loss(prediction, target) + weight * F.l1_loss(estimated, error)
