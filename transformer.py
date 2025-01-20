import torch
import numpy as np
import torch.nn as nn
import time
import math
from torch.utils.data import DataLoader
from dataset import AcousticDatasetId
from sklearn.metrics import roc_auc_score

device = torch.device("cuda")

class PositionalEncoding(nn.Module):
  def __init__(self, dim_model, dropout_p, max_len):
    super().__init__()
    # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # max_len determines how far the position can have an effect on a token (window)
    
    # Info
    self.dropout = nn.Dropout(dropout_p)
    
    # Encoding - From formula
    pos_encoding = torch.zeros(max_len, dim_model)
    positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
    division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
    
    # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
    pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
    
    # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
    pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
    
    # Saving buffer (same as parameter without gradients needed)
    pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
    self.register_buffer("pos_encoding", pos_encoding)
      
  def forward(self, token_embedding: torch.tensor) -> torch.tensor:
    # Residual connection + pos encoding
    return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class Transformer(nn.Module):
  def __init__(self, d_model=1024, nhead=16, num_layers=24, dropout_p=0.1, device="cuda"):
    super(Transformer, self).__init__()
    self.linear_layer = nn.Linear(20, d_model)
    self.linear_layer_2 = nn.Linear(20, d_model)
    self.linear_layer_3 = nn.Linear(d_model, 20)
    self.dropout = nn.Dropout(dropout_p)
    self.dropout_2 = nn.Dropout(dropout_p)
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True).to(device)
    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True).to(device)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers).to(device)
    self.model = nn.Transformer(
      d_model=d_model, 
      nhead=nhead, 
      num_encoder_layers=num_layers, 
      num_decoder_layers=num_layers,
      custom_encoder=transformer_encoder,
      custom_decoder=transformer_decoder
    ).to(device)
    self.positional_encoding = PositionalEncoding(dim_model=d_model, dropout_p=dropout_p, max_len=600)
    self.positional_encoding_2 = PositionalEncoding(dim_model=d_model, dropout_p=dropout_p, max_len=600)

  def forward(self, src, tgt):
    src = self.linear_layer(src)
    tgt = self.linear_layer_2(tgt)

    src = self.positional_encoding(src)
    tgt = self.positional_encoding_2(tgt)

    src = self.dropout(src)
    tgt = self.dropout_2(tgt)

    x = self.model(src, tgt)  
    x = self.linear_layer_3(x)
    #(batch_size, seq_len, 1)
    return x

def run_experiment():
  torch.cuda.empty_cache()
  # device = torch.device("cuda")
  model = Transformer(d_model=512, nhead=8, num_layers=6, dropout_p=0.1, device=device).to(device)

  path = "/data/MIMII/fan/id_00"

  train_dataset = AcousticDatasetId(path, include_abnormal=False, pattern=".wav", sampling_rate=16000, mono=True)
  val_dataset = AcousticDatasetId(path, include_abnormal=True, pattern=".wav", sampling_rate=16000, mono=True)
  learning_rate = 0.001
  epochs=20
  batch_size = 32

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  torch.cuda.empty_cache()
  train(train_loader, val_loader, learning_rate, model, epochs, device)

  


def train(train_loader, val_loader, learning_rate, model, epochs, device):
  criterion_reconstruction = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


  for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (inputs, labels) in enumerate(train_loader):
      inputs, labels = inputs.to(device), labels.to(device)
      # (batch_size, d_model, seq_len)
      inputs = inputs.permute(0, 2, 1)
      # (batch_size, seq_len, d_model)
      outputs = model(inputs, inputs)

      loss = criterion_reconstruction(inputs, outputs)

      optimizer.zero_grad()
      loss.backward()

      optimizer.step()
      total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")
    validate(model, val_loader, device)
    


def validate(model, val_loader, device):
  model.eval()
  all_labels = []
  all_anomaly_scores = []

  # Critério para calcular o erro por elemento
  criterion = nn.MSELoss(reduction='none')

  with torch.no_grad():
    for inputs, labels in val_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      inputs = inputs.permute(0, 2, 1)  # (batch_size, seq_len, d_model)

      # Reconstrução do modelo
      outputs = model(inputs, inputs)  # (batch_size, seq_len, d_model)

      # Calcula o erro de reconstrução por elemento
      reconstruction_error = criterion(outputs, inputs)  # (batch_size, seq_len, d_model)

      # Calcula o erro médio por sequência
      anomaly_scores = reconstruction_error.mean(dim=(1, 2))  # (batch_size,)

      # Coleta de pontuações de anomalia e labels reais
      all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  # Calcula o ROC AUC
  roc_auc = roc_auc_score(all_labels, all_anomaly_scores)
  print(f"Validation ROC AUC: {roc_auc:.4f}")






run_experiment()
