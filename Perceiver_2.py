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
        position = torch.arange(max_seq_len, device=device).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * (-np.log(n) / embed_dim))
        pos_enc = torch.zeros(max_seq_len, embed_dim, device=device)
        pos_enc[:, 0::2] = torch.sin(position * division_term)
        pos_enc[:, 1::2] = torch.cos(position * division_term)
        return pos_enc  
    
    def forward(self, x):
        _, seq_len = x.size()
        token_embed = self.embedding(x)
        pos_encodings = InputEmbedding.positional_encoding(self.max_seq_len, self.embed_dim, 10000).to(device)
        self.register_buffer('pos_enc', pos_encodings)
        return token_embed + pos_encodings[:seq_len]
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, heads=4):
        super().__init__()
        assert embed_dim % heads == 0
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads
        
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        self.attention_scores = None
          
    def attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        self.attention_scores = attention_scores
        return torch.matmul(attention_scores, value)
        
    def forward(self, x, mask=None):
        batch_size, sequence_length, _ = x.size()
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        Q = Q.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, sequence_length, self.heads, self.d_k).transpose(1, 2)
        
        attention_output = self.attention(Q, K, V, mask=mask)
        attention_output = self.dropout(attention_output)
        
        combined_heads = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.heads * self.d_k)
        return self.w_o(combined_heads)

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, heads=4):
        super().__init__()
        assert embed_dim % heads == 0
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_k = embed_dim // heads
        
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        self.attention_scores = None
          
    def attention(self, query, key, value, mask=None):
        d_k = query.shape[-1]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        self.attention_scores = attention_scores
        return torch.matmul(attention_scores, value)
        
    def forward(self, query, context, mask=None):
        batch_size, query_length, _ = query.size()
        _, context_length, _ = context.size()
        
        Q = self.w_q(query)
        K = self.w_k(context)
        V = self.w_v(context)
        
        Q = Q.view(batch_size, query_length, self.heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, context_length, self.heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, context_length, self.heads, self.d_k).transpose(1, 2)
        
        attention_output = self.attention(Q, K, V, mask=mask)
        attention_output = self.dropout(attention_output)
        
        combined_heads = attention_output.transpose(1, 2).contiguous().view(batch_size, query_length, self.heads * self.d_k)
        return self.w_o(combined_heads)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, heads).to(device)
        self.cross_attention = MultiHeadCrossAttention(embed_dim, heads).to(device)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, d_ff),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, embed_dim)
        )
        
    def forward(self, x, context=None, mask=None):
        if context is not None:
            cross_attention = self.cross_attention(x, context, mask=mask)
            x = self.norm1(cross_attention + x)
        else:
            self_attention = self.self_attention(x, mask=mask)
            x = self.norm1(self_attention + x)
        
        fforward = self.feed_forward(x)
        return self.norm2(fforward + x)
        
class TransformerModel_2(nn.Module):
    def __init__(self, embed_dim, heads, d_ff, seq_len, N, num_tokens):
        super().__init__()
        
        self.num_tokens = num_tokens
        
        self.embedding_layer = InputEmbedding(num_tokens, embed_dim, seq_len)
        self.transformer_layers = nn.ModuleList(
            [TransformerBlock(embed_dim, heads, d_ff) for _ in range(N)]
        )
        
        self.dropout = nn.Dropout(0.1)
        
        self.to_probs = nn.Linear(embed_dim, num_tokens)
    
    def forward(self, x, context=None):
        x = self.embedding_layer(x)
        batch_size, seq_length, embed_dim = x.size()
        
        x = self.dropout(x)
        
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        for i, layer in enumerate(self.transformer_layers):
            if i == 0 and context is not None:
                x = layer(x, context=context, mask=mask)
            else:
                x = layer(x, mask=mask)
        
        x = self.to_probs(x.view(batch_size * seq_length, embed_dim)).view(batch_size, seq_length, self.num_tokens)
        x = F.log_softmax(x, dim=2)
        
        return x
