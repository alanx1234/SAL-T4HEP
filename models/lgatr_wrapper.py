import torch
from torch import nn


def _patch_lgatr_cached_einsum():
    """Use torch.einsum instead of lgatr's opt_einsum path cache.

    Some cluster environments fail while opt_einsum builds a cached contraction
    path for L-GATr's normalization einsums. The plain torch implementation is
    slower but avoids that external path-planning failure.
    """

    def safe_cached_einsum(equation, *operands):
        return torch.einsum(equation, *operands)

    try:
        import lgatr.primitives.bilinear as bilinear
        import lgatr.primitives.invariants as invariants
        import lgatr.primitives.linear as linear
        import lgatr.utils.einsum as einsum_utils
    except ImportError:
        return

    einsum_utils.cached_einsum = safe_cached_einsum
    bilinear.cached_einsum = safe_cached_einsum
    invariants.cached_einsum = safe_cached_einsum
    linear.cached_einsum = safe_cached_einsum


class LGATrJetClassifier(nn.Module):
    """Jet classifier wrapper around the maintained ``lgatr`` package.

    The package is intentionally kept as an external dependency instead of
    vendoring the whole L-GATr library into this repository. Install with:

        python -m pip install lgatr
    """

    def __init__(
        self,
        num_classes,
        hidden_mv_channels=16,
        hidden_s_channels=32,
        num_blocks=12,
        num_heads=8,
        dropout=0.0,
        beam_spurion="xyplane",
        add_time_spurion=True,
        checkpoint_blocks=False,
    ):
        super().__init__()
        try:
            from lgatr import LGATr, extract_scalar, get_num_spurions, get_spurions
            from lgatr.interface import embed_vector
        except ImportError as exc:
            raise ImportError(
                "LGATrJetClassifier requires the external 'lgatr' package. "
                "Install it with: python -m pip install lgatr"
            ) from exc
        _patch_lgatr_cached_einsum()

        self.embed_vector = embed_vector
        self.extract_scalar = extract_scalar
        self.get_spurions = get_spurions
        self.beam_spurion = beam_spurion
        self.add_time_spurion = add_time_spurion
        num_spurions = get_num_spurions(
            beam_spurion=beam_spurion,
            add_time_spurion=add_time_spurion,
        )
        self.gatr = LGATr(
            in_mv_channels=1 + num_spurions,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=None,
            out_s_channels=None,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention={"num_heads": num_heads},
            mlp={},
            dropout_prob=dropout,
            checkpoint_blocks=checkpoint_blocks,
        )
        self.head = nn.Linear(1, num_classes)

    def forward(self, p4, mask=None):
        batch_size, num_particles, _ = p4.shape
        multivectors = self.embed_vector(p4).unsqueeze(-2)
        spurions = self.get_spurions(
            beam_spurion=self.beam_spurion,
            add_time_spurion=self.add_time_spurion,
            device=p4.device,
            dtype=p4.dtype,
        )
        if spurions.numel() > 0:
            spurions = spurions[None, None].expand(batch_size, num_particles, -1, -1)
            multivectors = torch.cat([multivectors, spurions], dim=-2)

        output_mv, _ = self.gatr(multivectors=multivectors, scalars=None)
        scalars = self.extract_scalar(output_mv).squeeze(-1).squeeze(-1)
        if mask is None:
            pooled = scalars.mean(dim=1, keepdim=True)
        else:
            weights = mask.to(scalars.dtype)
            pooled = (scalars * weights).sum(dim=1, keepdim=True)
            pooled = pooled / weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.head(pooled)
