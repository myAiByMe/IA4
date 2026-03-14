# transformer_block.py - v9 — Sequence Packing support
import torch
import torch.nn as nn
from typing import Optional, Tuple

from attention import MultiHeadAttention, RMSNorm, KVCache
from feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Transformer Block v9.
    Ajoute le support Sequence Packing : cu_seqlens_q/k passés à MultiHeadAttention.
    Tout le reste identique v8.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 use_rope=True, max_seq_len=2048,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024,
                 use_swiglu=True, n_kv_heads=None, use_qk_norm=False,
                 use_flash_attn=True, soft_cap=None):
        super().__init__()

        self.ln1 = RMSNorm(embed_dim)
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout,
            use_rope              = use_rope,
            max_seq_len           = max_seq_len,
            use_yarn              = use_yarn,
            yarn_scale            = yarn_scale,
            yarn_original_max_len = yarn_original_max_len,
            n_kv_heads            = n_kv_heads,
            use_qk_norm           = use_qk_norm,
            use_flash_attn        = use_flash_attn,
            soft_cap              = soft_cap,
        )
        self.ln2 = RMSNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout, use_swiglu=use_swiglu)

    def forward(
        self,
        x            : torch.Tensor,
        mask         : Optional[torch.Tensor] = None,
        past_kv      : Optional[KVCache]      = None,
        use_kv_cache : bool                   = False,
        # ── Sequence Packing ────────────────────────────────────
        cu_seqlens_q : Optional[torch.Tensor] = None,
        cu_seqlens_k : Optional[torch.Tensor] = None,
        max_seqlen_q : Optional[int]          = None,
        max_seqlen_k : Optional[int]          = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        residual = x
        x, new_kv = self.attention(
            self.ln1(x),
            mask         = mask,
            past_kv      = past_kv,
            use_kv_cache = use_kv_cache,
            cu_seqlens_q = cu_seqlens_q,
            cu_seqlens_k = cu_seqlens_k,
            max_seqlen_q = max_seqlen_q,
            max_seqlen_k = max_seqlen_k,
        )
        x = residual + x

        residual = x
        x        = self.ffn(self.ln2(x))
        x        = residual + x

        return x, new_kv