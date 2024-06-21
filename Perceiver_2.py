import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)

    @staticmethod      
    def positional_encoding(max_seq_len, embed_dim, n):
        # Generate an empty matrix
        position = torch.arange(max_seq_len, device=device).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * (-np.log(n) / embed_dim))
        pos_enc = torch.zeros(max_seq_len, embed_dim, device=device)
        pos_enc[:, 0::2] = torch.sin(position * division_term)
        pos_enc[:, 1::2] = torch.cos(position * division_term)
        return pos_enc  
    
    def forward(self, x):
        # (batch, seq_length) ---> (batch, seq_length, embed_size) 
        _, seq_len = x.size()
        token_embed = self.embedding(x)
        pos_encodings = InputEmbedding.positional_encoding(self.max_seq_len, self.embed_dim, 10000).to(device)
        self.register_buffer('pos_enc', pos_encodings)
        return token_embed + pos_encodings[:seq_len]
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads = 4):
        super().__init__()
        assert embed_dim % heads == 0
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads # Dimensions of vector seen by each head
        self.w_q = nn.Linear(embed_dim, embed_dim, bias = False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias = False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias = False)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias = False)
        self.dropuout = nn.Dropout(0.1)
        self.attention_scores = None
          
    def attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]
        #(batch, head, seq_len, d_k) ---> (batch, head, seq_length, seq_length)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        # apply padding mask
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        self.attention_scores = attention_scores
        #(batch, head, seq_length, seq_length) --> (batch, head, seq_len, d_k)
        return torch.matmul(attention_scores, value)
        
    def forward(self, x, mask=None):
        batch_size, sequence_length, _ = x.size()
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        # Split each tensor into heads, where each head has size d_k
        Q = Q.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        attention_output = self.attention(Q, K, V, mask=mask)
        attention_output = self.dropuout(attention_output)
        # (batch, head, seq_len, d_k) --> (batch, seq_len, head, d_k) --> (batch, seq_len, head * d_k)
        combined_heads = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.heads * self.d_k)
        return self.w_o(combined_heads)
    
class CrossAttention(nn.Module):
    def __init__(self, latent_dim, input_dim, heads=4):
        super().__init__()
        assert latent_dim % heads == 0
        self.heads = heads
        self.d_k = latent_dim // heads
        self.w_q = nn.Linear(latent_dim, latent_dim, bias=False)
        self.w_k = nn.Linear(input_dim, latent_dim, bias=False)
        self.w_v = nn.Linear(input_dim, latent_dim, bias=False)
        self.w_o = nn.Linear(latent_dim, latent_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

    def attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_scores, value)

    def forward(self, latent, input, mask=None):
        batch_size, seq_length, _ = input.size()
        _, latent_length, _ = latent.size()
        Q = self.w_q(latent)
        K = self.w_k(input)
        V = self.w_v(input)
        Q = Q.view(batch_size, latent_length, self.heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.heads, self.d_k).transpose(1, 2)
        attention_output = self.attention(Q, K, V, mask=mask)
        attention_output = self.dropout(attention_output)
        combined_heads = attention_output.transpose(1, 2).contiguous().view(batch_size, latent_length, self.heads * self.d_k)
        return self.w_o(combined_heads)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, d_ff):          
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, heads).to(device)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, embed_dim)
        )
        
    def forward(self, x, mask=None):
        attention = self.attention(x, mask=mask)
        x = self.norm1(attention + x)
        fforward = self.feed_forward(x)
        return self.norm2(fforward + x)

class TransformerModelWithCrossAttention(nn.Module):
    def __init__(self, embed_dim, latent_dim, heads, d_ff, seq_len, latent_len, N, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_layer = InputEmbedding(num_tokens, embed_dim, seq_len)
        self.latent = nn.Parameter(torch.randn(latent_len, latent_dim))
        self.cross_attention = CrossAttention(latent_dim, embed_dim, heads)
        self.transformer_layers = nn.ModuleList([TransformerBlock(latent_dim, heads, d_ff) for _ in range(N)])
        self.dropout = nn.Dropout(0.1)
        self.to_probs = nn.Linear(latent_dim, num_tokens)  # Map to the number of tokens

    def forward(self, x):
        x = self.embedding_layer(x).to(device)
        batch_size, seq_length, embed_dim = x.size()
        latent = self.latent.unsqueeze(0).expand(batch_size, -1, -1).to(device)
        latent = self.cross_attention(latent, x)
        for layer in self.transformer_layers:
            latent = layer(latent)
        latent = latent.mean(dim=1)
        x = self.to_probs(latent)  # Shape: [batch_size, latent_len, num_tokens]
        return F.log_softmax(x, dim=-1)  # Apply log_softmax for the NLL loss

