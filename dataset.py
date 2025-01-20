import torch
from torch.utils.data import Dataset
from mtsa.utils import Wav2Array
from mtsa.features.mel import Array2Mfcc
import numpy as np


class AcousticDatasetId(Dataset):
  def __init__(self, 
                X,  # Lista de caminhos dos arquivos de áudio
                y,  # Rótulos
                sampling_rate=16000, 
                mono=True,
                device="cuda"):
    """
    Args:
        X (list): Lista com caminhos dos arquivos de áudio.
        y (list): Lista com os rótulos (normal = 0, anômalo = 1).
        sampling_rate (int): Taxa de amostragem para carregar os arquivos de áudio.
        mono (bool): Se os áudios serão convertidos para mono.
    """
    self.labels = [] 
    self.scalar_paths = None
    self.sampling_rate = sampling_rate
    self.mono = mono
    self.X = X  # Caminhos para os arquivos de áudio
    self.y = y  # Rótulos
    self.device = device
    
    # Carregar os arquivos de áudio
    w2a = Wav2Array(sampling_rate=self.sampling_rate, mono=self.mono)
    a2m = Array2Mfcc(sampling_rate=self.sampling_rate)
    
    # Carregar os dados de áudio e transformá-los para MFCC
    audio_data = w2a.transform(self.X)  # Carregar os áudios
    self.scalar_paths = a2m.transform(audio_data)  # Transformar para MFCC
    

    self.scalar_paths = torch.tensor(self.scalar_paths, device=self.device) 
    self.labels = torch.tensor(self.y, device=self.device)  

  def __getitem__(self, idx):
    return self.scalar_paths[idx, :], self.labels[idx]

  def __len__(self):
    return len(self.X)