import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()

    # check if the d_model is divisible by num_heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

    #initialize dimension
    self.d_model = d_model # Model's Dimension
    self.num_heads = num_heads # Number of attention heads
    self.d_k = d_model // num_heads # Dimension of each head's key, query and value

    self.W_q = nn.Linear(d_model, d_model) # query transformation
    self.W_k = nn.Linear(d_model, d_model) # key transformation
    self.W_v = nn.Linear(d_model, d_model) # value transformation
    self.W_o = nn.Linear(d_model, d_model) # output transformation

  def scaled_dot_product_attention(self, Q, K ,V, mask=None):
    # calculate attention scores
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

    # Apply mask if provided (useful for preventing attention to certain parts like padding)
    if mask is not None:
      attn_scores = attn_scores.masked_fill(mask==0, -1e9)
    
    # softmax is applied to obtain attention probabilities
    attn_scores = torch.softmax(attn_scores, dim=-1)

    # multiply by values to obtain final output
    output = torch.matmul(attn_scores, V)
    return output
  
  def split_heads(self, x):
    # Reshape the input to have num_heads for multi-head attention
    batch_size, seq_length, d_model = x.size()
    return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
    # Combine the multiple heads back to original shape
    batch_size, _, seq_length, d_k = x.size()
    return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

  def forward(self, Q, K, V, mask=None):
    # Apply linear transformations and split heads
    Q = self.split_heads(self.W_q(Q))
    K = self.split_heads(self.W_k(K))
    V = self.split_heads(self.W_v(V))
    
    # Perform scaled dot-product attention
    attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
    
    # Combine heads and apply output transformation
    output = self.W_o(self.combine_heads(attn_output))
    return output

class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionWiseFeedForward, self).__init__()
    self.dropout_layer = nn.Dropout(p=dropout)
    self.fc_1 = nn.Linear(d_model, d_ff)
    self.fc_2 = nn.Linear(d_ff, d_model)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.fc_1(x)
    out = self.dropout_layer(self.relu(out))
    return self.fc_2(out)
    

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length=3000, dropout=0.1):
    super(PositionalEncoding, self).__init__()
    self.dropout_layer = nn.Dropout(p=dropout)

    pe = torch.zeros(max_seq_length, d_model)
    position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    self.register_buffer('pe', pe.unsqueeze(0))
  
  def forward(self, x):
    return self.dropout_layer(x + self.pe[:, :x.size(1)])

class EncoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super(EncoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    attn_output = self.self_attn(x, x, x, mask)
    x = self.norm1(x + self.dropout(attn_output))
    ff_output = self.feed_forward(x)
    x = self.norm2(x + self.dropout(ff_output))
    return x

class DecoderLayer(nn.Module):
  def __init__(self, d_model, num_heads, d_ff, dropout):
    super(DecoderLayer, self).__init__()
    self.self_attn = MultiHeadAttention(d_model, num_heads)
    self.cross_attn = MultiHeadAttention(d_model, num_heads)
    self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, enc_output, src_mask, tgt_mask):
    attn_output = self.self_attn(x, x, x, tgt_mask)
    x = self.norm1(x + self.dropout(attn_output))
    attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
    x = self.norm2(x + self.dropout(attn_output))
    ff_output = self.feed_forward(x)
    x = self.norm3(x + self.dropout(ff_output))
    return x

class AnomalyGPT(nn.Module):
  def __init__(self, mfcc_dim, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, device):
    super(AnomalyGPT, self).__init__()
    self.device = device
    self.src_linear_input = nn.Linear(mfcc_dim, d_model).to(device)
    self.tgt_linear_input = nn.Linear(mfcc_dim, d_model).to(device)
    self.model_output = nn.Linear(d_model, mfcc_dim).to(device)

    self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout).to(device)

    self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]).to(device)
    self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]).to(device)

    self.dropout = nn.Dropout(dropout).to(device)

  def generate_mask(self, tgt, device):
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device)
    tgt_mask = nopeak_mask
    return tgt_mask

  def forward(self, src, tgt):
    src = self.src_linear_input(src)
    tgt = self.tgt_linear_input(tgt)

    src = self.positional_encoding(src)
    tgt = self.positional_encoding(tgt)

    src = self.dropout(src)
    tgt = self.dropout(tgt)
    tgt_m = self.generate_mask(tgt, self.device)
    enc_output = src
    for enc_layer in self.encoder_layers:
      enc_output = enc_layer(enc_output, mask=None)

    dec_output = tgt
    for dec_layer in self.decoder_layers:
      dec_output = dec_layer(dec_output, enc_output, src_mask=None, tgt_mask=tgt_m)

    output = self.model_output(dec_output)
    return output