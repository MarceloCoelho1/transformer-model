import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from dataset import AcousticDatasetId
from models.anomalyGPT import AnomalyGPT
from models.anomalyGptWithTorch import build_transformer
from mtsa.utils import files_train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda")


def run_experiment(model, lr, epoch, batch_size, path):
  torch.cuda.empty_cache()

  model = build_transformer(mfcc_dim=20, d_model=128, nhead=2, num_encoder_layers=2, num_decoder_layers=2,
                            dim_feedforward=1024, dropout=0.1, batch_first=True, device=device).to(device)

  path = "/data/MIMII/fan/id_04"

  X_train, X_test, y_train, y_test = files_train_test_split(path)
  # normal == 1
  # abnormal == 0
  train_dataset = AcousticDatasetId(X_train, y_train, sampling_rate=16000, mono=True)
  val_dataset = AcousticDatasetId(X_test, y_test, sampling_rate=16000, mono=True)
  learning_rate = 0.001
  epochs = 15
  batch_size = 16

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
  train(train_loader, val_loader, learning_rate, model, epochs, device)


def train(train_loader, val_loader, learning_rate, model, epochs, device):
  criterion_reconstruction = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

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
    print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.4f} | Time: {elapsed_time:.2f}s")
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

      # Calcula o erro médio por sequência (por amostra no batch)
      anomaly_scores = reconstruction_error.mean(dim=(1, 2))  # (batch_size,)

      # Coleta de pontuações de anomalia e labels reais
      all_anomaly_scores.extend(anomaly_scores.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  # Calcula o ROC AUC
  fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_anomaly_scores)
  auc = metrics.auc(fpr, tpr)
  print(f"Validation ROC AUC: {auc}")


if __name__ == "__main__":

  model = build_transformer(mfcc_dim=20, d_model=128, nhead=2, num_encoder_layers=2, num_decoder_layers=2,
                            dim_feedforward=1024, dropout=0.1, batch_first=True, device=device).to(device)

  path = ["/data/MIMII/fan/id_04"]

  learning_rate = [0.001, 0.0001]
  epochs = [15, 20, 25]
  batch_size = [16, 32]

  run_experiment()
