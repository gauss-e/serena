import torch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
  def __init__(self, text, tokenizer, max_length, stride):
    self.input_ids = []
    self.target_ids = []

    token_ids = tokenizer.encode(text)

    for i in  range(0, len(token_ids), max_length, stride):
      input_chunk = token_ids[i:i + max_length]
      target_chunk = token_ids[i + 1: max_length + 1]
      self.input_ids.append(input_chunk)
      self.target_ids.append(target_chunk)

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return torch.tensor(self.input_ids[idx]), torch.tensor(self.target_ids[idx])