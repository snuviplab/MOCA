import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from colt5_attention.attend import Attend
from colt5_attention.transformer_block import CoordinateDescentRouter, Attention


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class FeatureEmbLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normed = F.normalize(x, dim=-1)
        return normed * self.scale * self.gamma


class SelfAttention(nn.Module):
    def __init__(self, dim, dim_head=32, num_heads=4, dropout=0.0, use_flash=False):
        super().__init__()
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=num_heads,
            use_flash=use_flash,
        )
        self.attn.attend = Attend(dropout=dropout)

    def forward(self, x, mask=None, rotary_emb=None):
        rotary_embs = (rotary_emb, rotary_emb) if exists(rotary_emb) else None
        return self.attn(x, mask=mask, rotary_emb=rotary_embs)


class ConditionalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_routed_queries,
        num_routed_key_values,
        *,
        dim_head=32,
        num_heads=4,
        dropout=0.0,
        use_flash=True,
        null_token_to_unrouted=True,
    ):
        super().__init__()
        self.num_routed_queries = num_routed_queries
        self.num_routed_key_values = num_routed_key_values
        self.null_token_to_unrouted = null_token_to_unrouted
        self.attn = Attention(
            dim=dim,
            dim_head=dim_head,
            heads=num_heads,
            use_flash=use_flash,
        )
        self.attn.attend = Attend(dropout=dropout)

        self.null_tokens = nn.Parameter(torch.randn(1, 1, dim)) if null_token_to_unrouted else None
        self.query_router = CoordinateDescentRouter(dim)
        self.key_value_router = CoordinateDescentRouter(dim)

    def forward(self, x, mask=None, rotary_emb=None):
        # Routing
        query_indices, query_scores, queries, query_mask = self.query_router(
            x, mask=mask, num_tokens=self.num_routed_queries
        )
        kv_indices, key_value_scores, key_values, key_value_mask = self.key_value_router(
            x, mask=mask, num_tokens=self.num_routed_key_values
        )

        # Rotary emb
        q_rotary_emb = rearrange(rotary_emb[query_indices], "b n d -> b 1 n d") if exists(query_indices) else rotary_emb
        k_rotary_emb = rearrange(rotary_emb[kv_indices], "... n d -> ... 1 n d") if exists(kv_indices) else rotary_emb
        rotary_embs = (q_rotary_emb, k_rotary_emb)

        # Attention
        attn_out = self.attn(
            queries,
            context=key_values,
            mask=key_value_mask,
            normalized_scores_kv=key_value_scores,
            rotary_emb=rotary_embs,
        )
        attn_out = attn_out * rearrange(query_scores, "... -> ... 1")

        # scatter back the output
        if exists(self.null_tokens):
            base_out = self.null_tokens.expand_as(x).clone()
        else:
            base_out = torch.zeros_like(x)
        return self.query_router.route_back(base_out, attn_out, query_indices)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerWithSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_layers=2,
        dim_head=64,
        num_heads=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_flash=False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            attn = SelfAttention(dim, dim_head=dim_head, num_heads=num_heads, dropout=attn_dropout, use_flash=use_flash)
            self.layers.append(
                nn.ModuleList(
                    [
                        attn,
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )
        self.out_ln = nn.LayerNorm(dim)

    def forward(self, x, mask=None, rotary_emb=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, rotary_emb=rotary_emb) + x
            x = ff(x) + x

        return self.out_ln(x)


class TransformerWithConditionalAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_layers=2,
        num_routed_queries=8,
        num_routed_key_values=8,
        num_experts=4,
        dim_head=64,
        num_heads=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        use_flash=True,
        null_token_to_unrouted=True,
    ):
        super().__init__()
        num_routed_queries = (
            num_routed_queries if type(num_routed_queries) == list else [num_routed_queries] * num_experts
        )
        num_routed_key_values = (
            num_routed_key_values if type(num_routed_key_values) == list else [num_routed_key_values] * num_experts
        )

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.ModuleList(
                            [
                                ConditionalAttention(
                                    dim,
                                    num_routed_queries=n_r_q,
                                    num_routed_key_values=n_r_kv,
                                    dim_head=dim_head,
                                    num_heads=num_heads,
                                    dropout=attn_dropout,
                                    use_flash=use_flash,
                                    null_token_to_unrouted=null_token_to_unrouted,
                                )
                                for n_r_q, n_r_kv in zip(num_routed_queries, num_routed_key_values)
                            ]
                        ),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )
        self.out_ln = nn.LayerNorm(dim)

    def forward(self, x, mask=None, rotary_emb=None):
        for c_attns, ff in self.layers:
            attn_out = torch.stack([c_attn(x, mask=mask, rotary_emb=rotary_emb) for c_attn in c_attns], axis=0)
            x = attn_out.mean(axis=0) + x
            x = ff(x) + x

        return self.out_ln(x)
