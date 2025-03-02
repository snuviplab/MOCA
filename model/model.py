import torch
from torch import nn
from einops import repeat
from model.module import FeatureEmbLayer, TransformerWithConditionalAttention, TransformerWithSelfAttention
from colt5_attention.transformer_block import RotaryEmbedding


class MOCA(nn.Module):
    def __init__(
        self,
        dim,
        num_feats,
        feature_dims,
        num_items,
        maxlen,
        num_layers,
        dim_head,
        num_heads,
        feature_transformer,
        feature_num_routed_queries,
        feature_num_routed_key_values,
        feature_num_experts,
        item_transformer,
        item_num_routed_queries,
        item_num_routed_key_values,
        item_num_experts,
        feature_dropout,
        attn_dropout,
        ff_dropout,
        use_flash,
        null_token_to_unrouted,
    ):
        super().__init__()

        self.num_feats = num_feats
        self.id_emb_layer = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.rotary_emb = RotaryEmbedding(dim_head)(maxlen)
        self.f_emb_layers = nn.ModuleList([FeatureEmbLayer(f_dim, dim, feature_dropout) for f_dim in feature_dims])

        self.feature_transformer = feature_transformer
        if feature_transformer == "full":
            self.f_trm = TransformerWithSelfAttention(
                dim=dim,
                num_layers=num_layers,
                dim_head=dim_head,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                use_flash=use_flash,
            )
        elif feature_transformer == "conditional":
            self.f_trm = TransformerWithConditionalAttention(
                dim=dim,
                num_layers=num_layers,
                num_routed_queries=feature_num_routed_queries,
                num_routed_key_values=feature_num_routed_key_values,
                num_experts=feature_num_experts,
                dim_head=dim_head,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                use_flash=use_flash,
                null_token_to_unrouted=null_token_to_unrouted,
            )
        elif feature_transformer == "none":
            self.f_trm = nn.Identity()

        self.item_transformer = item_transformer
        if item_transformer == "full":
            self.i_trm = TransformerWithSelfAttention(
                dim=dim,
                num_layers=num_layers,
                dim_head=dim_head,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                use_flash=use_flash,
            )
        elif item_transformer == "conditional":
            self.i_trm = TransformerWithConditionalAttention(
                dim=dim,
                num_layers=num_layers,
                num_routed_queries=item_num_routed_queries,
                num_routed_key_values=item_num_routed_key_values,
                num_experts=item_num_experts,
                dim_head=dim_head,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                use_flash=use_flash,
                null_token_to_unrouted=null_token_to_unrouted,
            )
        elif item_transformer == "none":
            self.i_trm = nn.Identity()

        self.f_concat = nn.Sequential(nn.Linear(num_feats * dim, dim), nn.LayerNorm(dim))

    def forward(self, seq_info):
        seq, *feats = seq_info
        self.rotary_emb = self.rotary_emb.to(seq.device)
        mask = seq > 0
        num_feats = len(feats) + 1

        # Embedding
        id_emb = self.id_emb_layer(seq)
        f_embs_list = [f_emb_layer(f) for f_emb_layer, f in zip(self.f_emb_layers, feats)]
        f_embs_list.append(id_emb)
        f_embs = torch.cat(f_embs_list, axis=1)  # b (n f) d

        # Feature Level
        f_mask = repeat(mask, "b n -> b (f n)", f=self.num_feats)
        f_rotary_emb = repeat(self.rotary_emb, "n d -> (f n) d", f=self.num_feats)
        if self.feature_transformer == "none":
            f_out = self.f_trm(f_embs)
        else:
            f_out = self.f_trm(f_embs, mask=f_mask, rotary_emb=f_rotary_emb)

        # Pooling (Concat and Linear Transformation)
        f_out = torch.cat(f_out.chunk(num_feats, dim=1), dim=-1)  # b n (f d)
        i_embs = self.f_concat(f_out)  # b n d

        # Item Level
        if self.item_transformer == "none":
            out = self.i_trm(i_embs)
        else:
            out = self.i_trm(i_embs, mask=mask, rotary_emb=self.rotary_emb)

        # Logits
        out = out[:, -1, :]
        logits = out @ self.id_emb_layer.weight.T

        return logits
