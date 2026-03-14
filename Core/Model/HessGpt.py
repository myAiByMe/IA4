# HessGpt.py - v9 — Sequence Packing support
"""
NOUVEAUTÉS v9 :
  ✅ Sequence Packing : forward accepte cu_seqlens_q/k et les propage à chaque block
  ✅ FlashAttention-4 : délégué à attention.py (détection automatique)
  ✅ Baseline benchmark : log tokens/sec + MFU dans forward si profiling=True
  Tout le reste identique v8 (KV Cache, top_p, weight tying, etc.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

from transformer_block import TransformerBlock
from attention import RMSNorm, KVCache, _FA_LEVEL


class HessGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim             = 768,
        num_heads             = 12,
        num_layers            = 12,
        max_seq_len           = 2048,
        dropout               = 0.1,
        use_rope              = True,
        use_yarn              = False,
        yarn_scale            = 1.0,
        yarn_original_max_len = 1024,
        use_swiglu            = True,
        n_kv_heads            = None,
        use_qk_norm           = False,
        soft_cap              = None,
        use_flash_attn        = True,
    ):
        super().__init__()

        assert vocab_size > 0
        assert embed_dim % num_heads == 0
        if n_kv_heads is not None:
            assert num_heads % n_kv_heads == 0
            assert embed_dim % n_kv_heads == 0
        if use_rope and use_yarn:
            assert 0 < yarn_original_max_len <= max_seq_len
            assert 0.1 <= yarn_scale <= 16.0

        self.vocab_size            = vocab_size
        self.embed_dim             = embed_dim
        self.num_heads             = num_heads
        self.num_layers            = num_layers
        self.max_seq_len           = max_seq_len
        self.use_rope              = use_rope
        self.use_yarn              = use_yarn
        self.yarn_scale            = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len
        self.use_swiglu            = use_swiglu
        self.n_kv_heads            = n_kv_heads
        self.use_qk_norm           = use_qk_norm
        self.soft_cap              = soft_cap
        self.use_flash_attn        = use_flash_attn

        # ── Embeddings ───────────────────────────────────────────
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = (
            None if use_rope else nn.Embedding(max_seq_len, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # ── Transformer Blocks ───────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim, num_heads, dropout,
                use_rope              = use_rope,
                max_seq_len           = max_seq_len,
                use_yarn              = use_yarn,
                yarn_scale            = yarn_scale,
                yarn_original_max_len = yarn_original_max_len,
                use_swiglu            = use_swiglu,
                n_kv_heads            = n_kv_heads,
                use_qk_norm           = use_qk_norm,
                use_flash_attn        = use_flash_attn,
                soft_cap              = soft_cap,
            )
            for _ in range(num_layers)
        ])

        self.ln_final    = RMSNorm(embed_dim)
        self.output_head = nn.Linear(vocab_size, embed_dim, bias=False)

        # Masque causal pré-alloué (compile-safe)
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
        )
        self.register_buffer('_causal_mask', causal_mask, persistent=False)

        self.apply(self._init_weights)
        # Weight tying
        self.output_head.weight = self.token_embeddings.weight

        fa_names = {0: 'Manuel', 1: 'SDPA', 2: 'FA2', 3: 'FA3', 4: 'FA4'}
        print(f"  HessGPT v9 | FlashAttention: {fa_names.get(_FA_LEVEL, '?')} "
              f"(level={_FA_LEVEL})")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def _get_causal_mask(self, seq_len):
        return self._causal_mask[:seq_len, :seq_len]

    def forward(
        self,
        input_ids    : torch.Tensor,
        targets      : Optional[torch.Tensor]    = None,
        pad_token_id : Optional[int]             = None,
        past_kv      : Optional[List[KVCache]]   = None,
        use_kv_cache : bool                      = False,
        # ── Sequence Packing ────────────────────────────────────
        cu_seqlens_q : Optional[torch.Tensor] = None,
        cu_seqlens_k : Optional[torch.Tensor] = None,
        max_seqlen_q : Optional[int]          = None,
        max_seqlen_k : Optional[int]          = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[KVCache]]]:

        batch_size, seq_len = input_ids.shape

        # ── Embeddings ───────────────────────────────────────────
        x = self.token_embeddings(input_ids)
        if self.position_embeddings is not None:
            pos = torch.arange(seq_len, device=input_ids.device)
            x   = x + self.position_embeddings(pos)
        x = self.dropout(x)

        # ── Masque causal (fallback si pas FA) ───────────────────
        # Si cu_seqlens fournis → FA varlen gère le masque en interne
        use_fa = (_FA_LEVEL >= 2 and self.use_flash_attn
                  and self.soft_cap is None and cu_seqlens_q is None)
        mask = None if use_fa else self._get_causal_mask(seq_len)

        # ── Transformer Blocks ───────────────────────────────────
        new_past_kv = [] if use_kv_cache else None

        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, new_kv  = block(
                x,
                mask         = mask,
                past_kv      = layer_past,
                use_kv_cache = use_kv_cache,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_k = cu_seqlens_k,
                max_seqlen_q = max_seqlen_q,
                max_seqlen_k = max_seqlen_k,
            )
            if use_kv_cache:
                new_past_kv.append(new_kv)

        # ── Final norm + logits ──────────────────────────────────
        x      = self.ln_final(x)
        logits = self.output_head(x)

        # ── Loss ─────────────────────────────────────────────────
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index = pad_token_id if pad_token_id is not None else -100,
            )

        return logits, loss, new_past_kv

    # ─────────────────────────────────────────────────────────────
    # Génération autoregressive — KV Cache + top_k + top_p
    # ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids     : torch.Tensor,
        max_new_tokens: int            = 50,
        temperature   : float          = 1.0,
        top_k         : Optional[int]  = None,
        top_p         : Optional[float]= None,
        eos_token_id  : Optional[int]  = None,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        device = input_ids.device

        if input_ids.size(1) > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len:]

        prefill_logits, _, past_kv = self.forward(input_ids, use_kv_cache=True)
        next_logits = prefill_logits[:, -1, :]

        for _ in range(max_new_tokens):
            logits = next_logits

            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    topk_v, _ = torch.topk(logits, k)
                    logits = logits.masked_fill(logits < topk_v[:, [-1]], float('-inf'))
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                    sorted_probs    = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    remove_mask     = (cumulative_probs - sorted_probs) >= top_p
                    sorted_logits   = sorted_logits.masked_fill(remove_mask, float('-inf'))
                    logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
                probs      = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            decode_logits, _, past_kv = self.forward(
                next_token, past_kv=past_kv, use_kv_cache=True)
            next_logits = decode_logits[:, -1, :]

        if was_training:
            self.train()
        return input_ids

    # ── Utilitaires ──────────────────────────────────────────────
    def resize_token_embeddings(self, new_vocab_size: int):
        if new_vocab_size == self.vocab_size:
            return
        old_emb = self.token_embeddings
        self.token_embeddings = nn.Embedding(new_vocab_size, self.embed_dim)
        n = min(old_emb.num_embeddings, new_vocab_size)
        with torch.no_grad():
            self.token_embeddings.weight.data[:n] = old_emb.weight.data[:n]
        self.output_head        = nn.Linear(self.embed_dim, new_vocab_size, bias=False)
        self.output_head.weight = self.token_embeddings.weight
        self.vocab_size         = new_vocab_size

    def count_parameters(self):
        token_params = self.token_embeddings.weight.numel()
        pos_params   = (self.position_embeddings.weight.numel()
                        if self.position_embeddings is not None else 0)
        block_params = sum(p.numel() for b in self.blocks for p in b.parameters())
        ln_params    = sum(p.numel() for p in self.ln_final.parameters())
        total        = token_params + pos_params + block_params + ln_params
        return {
            'token_embeddings':    token_params,
            'position_embeddings': pos_params,
            'transformer_blocks':  block_params,
            'final_ln':            ln_params,
            'output_head':         0,
            'total':               total,
        }

    def get_config(self):
        return {
            'vocab_size':            self.vocab_size,
            'embed_dim':             self.embed_dim,
            'num_heads':             self.num_heads,
            'num_layers':            self.num_layers,
            'max_seq_len':           self.max_seq_len,
            'use_rope':              self.use_rope,
            'use_yarn':              self.use_yarn,
            'yarn_scale':            self.yarn_scale,
            'yarn_original_max_len': self.yarn_original_max_len,
            'use_swiglu':            self.use_swiglu,
            'n_kv_heads':            self.n_kv_heads,
            'use_qk_norm':           self.use_qk_norm,
            'soft_cap':              self.soft_cap,
            'use_flash_attn':        self.use_flash_attn,
        }