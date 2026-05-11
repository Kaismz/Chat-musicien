"""
Modèle GPT (transformer décodeur) pour la génération musicale.

Architecture identique à GPT-2 (LLM.pdf du cours) :
    Token Emb + Pos Emb -> Dropout
    -> [Transformer Block × N]
    -> LayerNorm finale
    -> Linear -> logits

Chaque Transformer Block :
    LayerNorm -> Multi-Head Attention masquée -> Skip connection
    LayerNorm -> Feed-Forward (4× expansion) -> Skip connection
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration par défaut (modèle "Chat Musicien")
# ---------------------------------------------------------------------------
GPT_CONFIG_MUSIC = {
    "vocab_size": 157,        # vocab REMI de notre tokenizer
    "context_length": 256,    # tokens vus en même temps
    "emb_dim": 256,           # dimension d'embedding
    "n_heads": 8,             # nb de têtes d'attention
    "n_layers": 6,            # nb de blocs transformer
    "drop_rate": 0.1,         # dropout
    "qkv_bias": False,        # pas de biais sur Q/K/V (style moderne)
}


# ---------------------------------------------------------------------------
# Multi-Head Attention causale
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out doit être divisible par num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # Projections Q, K, V (un seul Linear par usage, plus rapide)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Projection de sortie (combine les têtes)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Masque causal (triangle supérieur = positions futures bloquées)
        # On le stocke comme buffer (suit le device automatiquement)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 1. Projections
        queries = self.W_query(x)   # (b, T, d_out)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        # 2. Reshape pour multi-head : (b, T, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # 3. Transpose : (b, num_heads, T, head_dim)
        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)

        # 4. Scores d'attention : (b, num_heads, T, T)
        attn_scores = queries @ keys.transpose(2, 3)

        # 5. Masque causal
        mask_bool = self.mask[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 6. Softmax + scaling
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. Pondération des values + reshape : (b, T, d_out)
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # 8. Projection finale
        return self.out_proj(context_vec)


# ---------------------------------------------------------------------------
# Sous-couches : LayerNorm, GELU, FeedForward
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """LayerNorm maison (équivalent à nn.LayerNorm, pour respecter le cours)."""
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm + self.shift


class GELU(nn.Module):
    """Approximation de GELU (utilisée par GPT-2)."""
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """MLP à 2 couches avec expansion 4×."""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Pre-LN + résidu (style GPT-2/3)
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


# ---------------------------------------------------------------------------
# Modèle GPT complet
# ---------------------------------------------------------------------------
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def num_params(self):
        return sum(p.numel() for p in self.parameters())