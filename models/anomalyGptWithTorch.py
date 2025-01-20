import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, dropout, max_len):
    super().__init__()
    # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # max_len determines how far the position can have an effect on a token (window)
    
    # Info
    self.dropout = nn.Dropout(dropout)
    
    # Encoding - From formula
    pos_encoding = torch.zeros(max_len, d_model)
    positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
    division_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model) # 1000^(2i/d_model)
    
    # PE(pos, 2i) = sin(pos/1000^(2i/d_model))
    pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
    
    # PE(pos, 2i + 1) = cos(pos/1000^(2i/d_model))
    pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
    
    # Saving buffer (same as parameter without gradients needed)
    pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
    self.register_buffer("pos_encoding", pos_encoding)
      
  def forward(self, token_embedding: torch.tensor) -> torch.tensor:
    # Residual connection + pos encoding
    return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    def __init__(
          self, 
          mfcc_dim=20, 
          d_model=1024, 
          nhead=16, 
          num_encoder_layers=6, 
          num_decoder_layers=6, 
          dim_feedforward=2028, 
          dropout=0.1, 
          device="cuda",
          batch_first=True):
        super(Transformer, self).__init__()
        self.src_linear_input = nn.Linear(mfcc_dim, d_model).to(device)
        self.tgt_linear_input = nn.Linear(mfcc_dim, d_model).to(device)
        self.out = nn.Linear(d_model, 20)
        self.dropout = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.model = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
            device=device,
            batch_first=batch_first
        ).to(device)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=600)
        self.positional_encoding_2 = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=600)

    def forward(self, src, tgt):
        src = self.src_linear_input(src)
        tgt = self.tgt_linear_input(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding_2(tgt)

        src = self.dropout(src)
        tgt = self.dropout_2(tgt)

        x = self.model(src, tgt)  
        x = self.out(x)
        return x
  

def build_transformer(
        mfcc_dim=20,
        d_model=512, 
        nhead=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=True,
        device="cuda"
        ):
    transformer = Transformer(
        mfcc_dim=mfcc_dim,
        d_model=d_model, 
        nhead=nhead, 
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        batch_first=batch_first,
        device=device,
        )
    return transformer
