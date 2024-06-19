import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.register_buffer('pos_enc', self.positional_encoding(max_seq_len, embed_dim, 10000))
    
    @staticmethod
    def positional_encoding(max_seq_len, embed_dim, n):
        position = torch.arange(max_seq_len).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(n) / embed_dim))
        pos_enc = torch.zeros(max_seq_len, embed_dim)
        
        pos_enc[:, 0::2] = torch.sin(position * division_term)
        pos_enc[:, 1::2] = torch.cos(position * division_term)
        return pos_enc  
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        token_embed = self.embedding(x)
        pos_encodings = self.pos_enc[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1).to(x.device)
        return token_embed + pos_encodings
    

class PerceiverAttention(nn.Module):
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout=0.0):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(embed_dim)
        self.lnormq = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)
        self.lnorm2 = nn.LayerNorm(embed_dim)
        self.linear1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, q):
        out = self.lnorm1(x)
        print(f'Query shape:{q.shape}, Key shape: {x.shape}, Value shape:{x.shape}')
       

        out, _ = self.attn(query=q.permute(1, 0, 2), key=x.permute(1, 0, 2), value=x.permute(1, 0, 2))
        resid = out + q
        out = self.lnorm2(resid)
        out = self.linear1(out)
        out = self.act(out)
        out = self.linear2(out)
        out = self.drop(out)
        return out + resid

class PerceiverBlock(nn.Module):
    def __init__(self, embed_dim, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers):
        super().__init__()
        self.cross_attention = PerceiverAttention(embed_dim, attn_mlp_dim, n_heads=1, dropout=dropout)
        self.latent_transformer = LatentTransformer(embed_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers)

    def forward(self, x, l):
        l = self.cross_attention(x, l)
        l = self.latent_transformer(l)
        return l

class LatentTransformer(nn.Module):
    def __init__(self, embed_dim, mlp_dim, n_heads, dropout, n_layers):
        super().__init__()
        self.transformer = nn.ModuleList(
            [PerceiverAttention(embed_dim=embed_dim, mlp_dim=mlp_dim, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, l):
        for trnfr in self.transformer:
            l = trnfr(l, l)
        return l

class Perceiver(nn.Module):
    def __init__(self, vocab_size, max_seq_len, latent_dim, embed_dim, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers, n_blocks,
        num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent = nn.Parameter(
            torch.nn.init.trunc_normal_(
                torch.zeros((latent_dim, 1, embed_dim)), mean=0, std=0.02, a=-2, b=2
            )
        )
        self.embedding_layer = InputEmbedding(vocab_size, embed_dim, max_seq_len)
        self.perceiver_blocks = nn.ModuleList(
            [PerceiverBlock(embed_dim, attn_mlp_dim, trnfr_mlp_dim, trnfr_heads, dropout, trnfr_layers) for _ in range(n_blocks)]
        )
        self.to_probs = nn.Linear(embed_dim, num_tokens)

    def forward(self, x):
        batch_size = x.shape[0]
        latent = self.latent.expand(-1, batch_size, -1)
        x = self.embedding_layer(x)
        for pb in self.perceiver_blocks:
            latent = pb(x, latent)

        logits = self.to_probs(latent)
        return F.log_softmax(logits, dim=-1)
