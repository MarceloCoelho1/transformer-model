import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from dataset import AcousticDatasetId
from sklearn.metrics import roc_auc_score
from models.anomalyGPT import AnomalyGPT
from mtsa.utils import files_train_test_split
from sklearn import metrics



device = torch.device("cuda")
def run_experiment():
  torch.cuda.empty_cache()

  model = AnomalyGPT(mfcc_dim=20, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_seq_length=313, dropout=0.1, device=device).to(device)

  path = "/data/MIMII/fan/id_04"

  X_train, X_test, y_train, y_test = files_train_test_split(path)
  # normal == 1
  # abnormal == 0
  train_dataset = AcousticDatasetId(X_train, y_train, sampling_rate=16000, mono=True)
  val_dataset = AcousticDatasetId(X_test, y_test, sampling_rate=16000, mono=True)
  
  learning_rate = 0.001
  epochs=15
  batch_size = 16

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
  train(train_loader, val_loader, learning_rate, model, epochs, device)

  
def train(train_loader, val_loader, learning_rate, model, epochs, device):
  criterion_reconstruction = nn.MSELoss()
  # optimizer = NoamOpt(
  #   model_size=256,  # d_model do seu modelo
  #   factor=2,        # Ajuste conforme necessário (2 é um valor comum)
  #   warmup=4000,     # Número de passos de warmup (experimente valores diferentes)
  #   optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
  # )
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

class NoamOpt:
  "Optim wrapper that implements rate."
  def __init__(self, model_size, factor, warmup, optimizer):
    self.optimizer = optimizer
    self._step = 0
    self.warmup = warmup
    self.factor = factor
    self.model_size = model_size
    self._rate = 0
      
  def step(self):
    "Update parameters and rate"
    self._step += 1
    rate = self.rate()
    for p in self.optimizer.param_groups:
        p['lr'] = rate
    self._rate = rate
    self.optimizer.step()
      
  def rate(self, step = None):
    "Implement `lrate` above"
    if step is None:
      step = self._step
    return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    
def get_std_opt(model):
  return NoamOpt(model.src_embed[0].d_model, 2, 4000,
    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

run_experiment()