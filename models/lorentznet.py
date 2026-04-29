import torch
from torch import nn


def unsorted_segment_sum(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def normsq4(p):
    psq = torch.pow(p, 2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p, q):
    psq = p * q
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def psi(p):
    return torch.sign(p) * torch.log(torch.abs(p) + 1)


class LGEB(nn.Module):
    """Lorentz Group Equivariant Block from the LorentzNet release."""

    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_node_attr=0,
        dropout=0.0,
        c_weight=1.0,
        last_layer=False,
    ):
        super().__init__()
        self.c_weight = c_weight
        n_edge_attr = 2

        self.phi_e = nn.Sequential(
            nn.Linear(n_input * 2 + n_edge_attr, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )

        layer = nn.Linear(n_hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), layer)
        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        self.last_layer = last_layer
        self.dropout = nn.Dropout(dropout)
        if last_layer:
            del self.phi_x

    def m_model(self, hi, hj, norms, dots):
        out = torch.cat([hi, hj, norms, dots], dim=1)
        out = self.phi_e(out)
        return self.dropout(out * self.phi_m(out))

    def h_model(self, h, edges, m, node_attr):
        i, _ = edges
        agg = unsorted_segment_sum(m, i, num_segments=h.size(0))
        agg = torch.cat([h, agg, node_attr], dim=1)
        return h + self.phi_h(agg)

    def x_model(self, x, edges, x_diff, m):
        i, _ = edges
        trans = x_diff * self.phi_x(m)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, i, num_segments=x.size(0))
        return x + agg * self.c_weight

    def minkowski_feats(self, edges, x):
        i, j = edges
        x_diff = x[i] - x[j]
        norms = psi(normsq4(x_diff).unsqueeze(1))
        dots = psi(dotsq4(x[i], x[j]).unsqueeze(1))
        return norms, dots, x_diff

    def forward(self, h, x, edges, node_attr):
        i, j = edges
        norms, dots, x_diff = self.minkowski_feats(edges, x)
        m = self.m_model(h[i], h[j], norms, dots)
        if not self.last_layer:
            x = self.x_model(x, edges, x_diff, m)
        h = self.h_model(h, edges, m, node_attr)
        return h, x, m


class LorentzNet(nn.Module):
    """LorentzNet graph classifier.

    Adapted from https://github.com/sdogsq/LorentzNet-release.
    Inputs use four-vector convention ``(E, px, py, pz)``.
    """

    def __init__(
        self,
        n_scalar,
        n_hidden,
        n_class=2,
        n_layers=6,
        c_weight=1e-3,
        dropout=0.0,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding = nn.Linear(n_scalar, n_hidden)
        self.blocks = nn.ModuleList(
            [
                LGEB(
                    n_hidden,
                    n_hidden,
                    n_hidden,
                    n_node_attr=n_scalar,
                    dropout=dropout,
                    c_weight=c_weight,
                    last_layer=(i == n_layers - 1),
                )
                for i in range(n_layers)
            ]
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_class),
        )

    def forward(self, scalars, x, edges, node_mask, n_nodes):
        h = self.embedding(scalars)
        for block in self.blocks:
            h, x, _ = block(h, x, edges, node_attr=scalars)

        h = h * node_mask
        h = h.view(-1, n_nodes, self.n_hidden)
        denom = node_mask.view(-1, n_nodes, 1).sum(dim=1).clamp(min=1.0)
        h = h.sum(dim=1) / denom
        return self.graph_dec(h)
