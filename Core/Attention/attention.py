# attention.py - v10 — SDPA Prioritaire + Activation Checkpointing ready
"""
NOUVEAUTÉS v10 :
  ✅ SDPA PyTorch 2.8 PRIORITAIRE sur flash_attn
      - Sur B200 SM100 : SDPA = FA3 natif Blackwell (plus rapide que flash_attn 2.x)
      - Sur H100 SM90  : SDPA = FA3 natif Hopper
      - Sur A100 SM80  : SDPA = FA2 optimisé
      - Aucune dépendance externe, toujours disponible avec PyTorch >= 2.0
      - Compatible torch.compile + activation checkpointing

  ✅ flash_attn conservé UNIQUEMENT pour varlen (sequence packing)
      - SDPA ne supporte pas cu_seqlens → flash_attn_varlen_func requis
      - Si flash_attn absent : packing désactivé automatiquement, SDPA utilisé

  ✅ Soft Cap, KV Cache, GQA, QK-Norm, RoPE/YaRN : inchangés

HIÉRARCHIE ATTENTION :
  1. varlen  (flash_attn >= 2.0, cu_seqlens fournis) → sequence packing
  2. SDPA    (PyTorch >= 2.0, prioritaire standard)   → FA3 natif sur B200/H100
  3. Manuel  (soft_cap ou mask custom)               → dernier recours
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ============================================================
# FLASH ATTENTION — Détection hiérarchique
# ============================================================

_FA_LEVEL = 0          # 0=aucun, 2=FA2, 3=FA3, 4=FA4
_FA_VARLEN_FUNC = None # flash_attn_varlen_func si dispo
_FA_FUNC        = None # flash_attn_func si dispo

def _detect_flash_attn():
    """
    Détection de la meilleure backend d'attention disponible.

    Stratégie v10 :
      - SDPA PyTorch 2.8 est PRIORITAIRE car c'est FA3 natif sur B200/H100,
        plus rapide que flash_attn 2.x, sans dépendance, et compatible compile.
      - flash_attn est chargé UNIQUEMENT pour flash_attn_varlen_func
        (sequence packing avec cu_seqlens) — SDPA ne supporte pas varlen.
    """
    global _FA_LEVEL, _FA_VARLEN_FUNC, _FA_FUNC

    # ── Priorité 1 : SDPA natif PyTorch ─────────────────────────
    # Sur B200 SM100 : PyTorch 2.8 dispatche vers FA3 Blackwell nativement.
    # C'est le chemin le plus rapide, le plus stable, et compatible compile.
    if hasattr(F, 'scaled_dot_product_attention'):
        _FA_LEVEL = 1
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 10:
                print("  ⚡ SDPA PyTorch 2.8 — FA3 natif Blackwell SM100 (prioritaire)")
            elif cap[0] >= 9:
                print("  ⚡ SDPA PyTorch — FA3 natif Hopper SM90 (prioritaire)")
            elif cap[0] >= 8:
                print("  ⚡ SDPA PyTorch — FA2 natif Ampere SM80 (prioritaire)")
            else:
                print("  ⚡ SDPA PyTorch natif (prioritaire)")
        else:
            print("  ⚡ SDPA PyTorch natif (CPU)")
    else:
        print("  ⚠️  SDPA non disponible (PyTorch < 2.0) — performances dégradées")

    # ── Priorité 2 : flash_attn pour varlen uniquement ───────────
    # SDPA ne supporte pas cu_seqlens (sequence packing).
    # On charge flash_attn SEULEMENT pour flash_attn_varlen_func.
    # _FA_FUNC reste None → le chemin FA2/FA3/FA4 standard n'est pas utilisé,
    # SDPA le remplace avantageusement.
    try:
        import flash_attn
        version = tuple(int(x) for x in flash_attn.__version__.split(".")[:2])
        if version >= (2, 0):
            from flash_attn.flash_attn_interface import flash_attn_varlen_func
            _FA_VARLEN_FUNC = flash_attn_varlen_func
            # _FA_LEVEL passe à 2 pour signaler que varlen est dispo
            _FA_LEVEL = 2
            print(f"  ⚡ flash_attn {flash_attn.__version__} — varlen uniquement (sequence packing)")
        # _FA_FUNC reste intentionnellement None :
        # le chemin standard utilise SDPA, pas flash_attn_func
    except (ImportError, Exception):
        _FA_VARLEN_FUNC = None
        # Pas de flash_attn = pas de sequence packing, SDPA utilisé pour tout
        if _FA_LEVEL == 1:
            print("  ℹ️  flash_attn absent — sequence packing désactivé, SDPA pour tout")

_detect_flash_attn()


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ============================================================
# RoPE + YaRN
# ============================================================

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000, device=None,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024):
        super().__init__()
        self.dim                   = dim
        self.max_seq_len           = max_seq_len
        self.base                  = base
        self.use_yarn              = use_yarn
        self.yarn_scale            = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len

        if use_yarn:
            assert 0.1 <= yarn_scale <= 16.0
            inv_freq = self._compute_yarn_frequencies()
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.register_buffer('inv_freq', inv_freq)
        self._seq_len_cached = None
        self._cos_cached     = None
        self._sin_cached     = None

    def _compute_yarn_frequencies(self):
        freqs        = torch.arange(0, self.dim, 2).float() / self.dim
        inv_freq_base = 1.0 / (self.base ** freqs)
        if self.yarn_scale == 1.0:
            return inv_freq_base
        alpha = self.yarn_scale
        beta  = max(self.dim // 2, int(self.dim * 0.25))
        dims  = torch.arange(0, self.dim, 2).float()
        scale = torch.where(
            dims < beta,
            torch.ones_like(dims),
            1 + (alpha - 1) * (dims - beta) / (self.dim - beta)
        )
        return inv_freq_base / scale

    def _update_cos_sin_cache(self, seq_len, device, dtype):
        if (seq_len != self._seq_len_cached or
                self._cos_cached is None or
                self._cos_cached.device != device or
                self._cos_cached.dtype != dtype):
            self._seq_len_cached = seq_len
            t     = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb   = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached

    def rotate_half(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, position_offset: int = 0):
        seq_len   = q.shape[2]
        total_len = seq_len + position_offset
        cos, sin  = self._update_cos_sin_cache(total_len, q.device, q.dtype)
        cos = cos[position_offset : position_offset + seq_len][None, None, :, :]
        sin = sin[position_offset : position_offset + seq_len][None, None, :, :]
        return (q * cos) + (self.rotate_half(q) * sin), \
               (k * cos) + (self.rotate_half(k) * sin)

    def forward(self, q, k, position_offset: int = 0):
        return self.apply_rotary_pos_emb(q, k, position_offset)


# ============================================================
# KV Cache type alias
# ============================================================

KVCache = Tuple[torch.Tensor, torch.Tensor]


# ============================================================
# Multi-Head Attention v9
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    MHA v9 — FA4/FA3/FA2/SDPA + Sequence Packing (varlen).

    Nouveaux args forward :
      cu_seqlens_q : [batch+1] int32 — offsets séquences dans le batch packé (optionnel)
      cu_seqlens_k : [batch+1] int32 — idem pour K (= cu_seqlens_q si pas de cache)
      max_seqlen_q : int — longueur max d'une séquence dans le batch packé
      max_seqlen_k : int — idem pour K

    Si cu_seqlens_q est None → comportement identique v8 (padding classique).
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1,
                 use_rope=True, max_seq_len=2048,
                 use_yarn=False, yarn_scale=1.0, yarn_original_max_len=1024,
                 n_kv_heads=None, use_qk_norm=False, use_flash_attn=True,
                 soft_cap=None):
        super().__init__()

        assert embed_dim % num_heads == 0
        if soft_cap is not None:
            assert soft_cap > 0

        self.embed_dim      = embed_dim
        self.num_heads      = num_heads
        self.head_dim       = embed_dim // num_heads
        self.use_rope       = use_rope
        self.use_qk_norm    = use_qk_norm
        self.use_flash_attn = use_flash_attn
        self.soft_cap       = soft_cap

        self.n_kv_heads         = n_kv_heads if n_kv_heads is not None else num_heads
        assert num_heads % self.n_kv_heads == 0
        self.num_queries_per_kv = num_heads // self.n_kv_heads
        self.kv_dim             = self.n_kv_heads * self.head_dim

        self.q_proj   = nn.Linear(embed_dim, embed_dim,    bias=False)
        self.k_proj   = nn.Linear(embed_dim, self.kv_dim,  bias=False)
        self.v_proj   = nn.Linear(embed_dim, self.kv_dim,  bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim,    bias=False)
        self.dropout  = nn.Dropout(dropout)

        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, max_seq_len,
                use_yarn              = use_yarn,
                yarn_scale            = yarn_scale,
                yarn_original_max_len = yarn_original_max_len,
            )
        else:
            self.rope = None

        # ── Capacités FA disponibles ─────────────────────────────
        self._fa_level    = _FA_LEVEL if use_flash_attn else 0
        self._fa_varlen   = _FA_VARLEN_FUNC
        self._fa_func     = _FA_FUNC
        # SDPA disponible (PyTorch >= 2.0)
        self._sdpa_ok     = hasattr(F, 'scaled_dot_product_attention')

        if use_flash_attn and _FA_LEVEL == 0 and not self._sdpa_ok:
            print("⚠️  Flash Attention non disponible (PyTorch < 2.0)")

    # ── helpers ─────────────────────────────────────────────────

    def _attn_scale(self):
        if (self.use_rope and self.rope is not None
                and self.rope.use_yarn and self.rope.yarn_scale > 1.0):
            return math.sqrt(self.rope.yarn_scale) / math.sqrt(self.head_dim)
        return 1.0 / math.sqrt(self.head_dim)

    # ── forward ─────────────────────────────────────────────────

    def forward(
        self,
        x            : torch.Tensor,
        mask         : Optional[torch.Tensor] = None,
        past_kv      : Optional[KVCache]      = None,
        use_kv_cache : bool                   = False,
        # ── Sequence Packing (varlen) ────────────────────────────
        cu_seqlens_q : Optional[torch.Tensor] = None,
        cu_seqlens_k : Optional[torch.Tensor] = None,
        max_seqlen_q : Optional[int]          = None,
        max_seqlen_k : Optional[int]          = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        batch_size, seq_len, _ = x.shape
        scale = self._attn_scale()

        # ── Projections ──────────────────────────────────────────
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.num_heads,   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads,  self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads,  self.head_dim).transpose(1, 2)

        # ── QK-Norm ──────────────────────────────────────────────
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ── RoPE ─────────────────────────────────────────────────
        position_offset = past_kv[0].shape[2] if past_kv is not None else 0
        if self.use_rope:
            q, k = self.rope(q, k, position_offset=position_offset)

        # ── KV Cache ─────────────────────────────────────────────
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv_cache: Optional[KVCache] = (k, v) if use_kv_cache else None

        # ── GQA repeat ───────────────────────────────────────────
        if self.n_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # ── Attention — hiérarchie : varlen > SDPA > Manuel ────────
        #
        #  1. varlen  : sequence packing (cu_seqlens) via flash_attn_varlen_func
        #               SDPA ne supporte pas cu_seqlens → flash_attn requis ici
        #  2. SDPA    : chemin principal — FA3 natif sur B200/H100 via PyTorch 2.8
        #               plus rapide que flash_attn 2.x, compatible compile
        #  3. Manuel  : uniquement si soft_cap ou mask custom

        use_varlen = (cu_seqlens_q is not None
                      and self._fa_varlen is not None
                      and self.soft_cap is None
                      and past_kv is None)

        if use_varlen:
            # ── Chemin varlen — sequence packing ─────────────────
            # flash_attn_varlen_func refuse float32 → cast bf16
            if q.dtype == torch.float32:
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)
            q_var = q.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)
            k_var = k.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)
            v_var = v.permute(0, 2, 1, 3).reshape(-1, self.num_heads, self.head_dim)
            _msl_q = max_seqlen_q if max_seqlen_q is not None else seq_len
            _msl_k = max_seqlen_k if max_seqlen_k is not None else seq_len
            output = self._fa_varlen(
                q_var, k_var, v_var,
                cu_seqlens_q, cu_seqlens_k,
                _msl_q, _msl_k,
                dropout_p     = self.dropout.p if self.training else 0.0,
                softmax_scale = scale,
                causal        = True,
            )
            output = output.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            output = output.transpose(1, 2)

        elif self._sdpa_ok and self.soft_cap is None and mask is None:
            # ── Chemin SDPA — prioritaire pour l'attention standard ──
            # PyTorch 2.8 dispatche automatiquement vers le meilleur kernel :
            #   SM100 (B200)  → FA3 Blackwell natif
            #   SM90  (H100)  → FA3 Hopper natif
            #   SM80  (A100)  → FA2 Ampere natif
            # Cast bf16 si le modèle est en float32 (inférence sans .to(bf16))
            if q.dtype == torch.float32:
                q = q.to(torch.bfloat16)
                k = k.to(torch.bfloat16)
                v = v.to(torch.bfloat16)
            is_causal = (seq_len > 1 and past_kv is None)
            orig_dtype = q.dtype
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = None,
                is_causal = is_causal,
                dropout_p = self.dropout.p if self.training else 0.0,
                scale     = scale,
            )
            if output.dtype != orig_dtype:
                output = output.to(orig_dtype)

        else:
            # ── Chemin Manuel — soft_cap ou mask custom ───────────
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.soft_cap is not None:
                scores = self.soft_cap * torch.tanh(scores / self.soft_cap)
            if seq_len > 1 and past_kv is None:
                if mask is not None:
                    scores = scores.masked_fill(
                        mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                else:
                    total_len   = k.shape[2]
                    causal_bool = torch.triu(
                        torch.ones(seq_len, total_len,
                                   device=q.device, dtype=torch.bool), diagonal=1)
                    scores = scores.masked_fill(
                        causal_bool.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            if self.training and self.dropout.p > 0:
                attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, v)

        # ── Reshape + projection ─────────────────────────────────
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.embed_dim)
        # Aligne dtype sur out_proj.weight (évite mismatch bf16/float32 en inférence)
        if output.dtype != self.out_proj.weight.dtype:
            output = output.to(self.out_proj.weight.dtype)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, new_kv_cache
