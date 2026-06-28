import torch
from torch.utils.data import Dataset
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

class GPTDataset(Dataset):
  def __init__(self, text, max_length, stride):
    self.input_ids: list[torch.Tensor] = []
    self.target_ids: list[torch.Tensor] = []

    token_ids = tokenizer.encode(text)

    for i in  range(0, len(token_ids) - max_length, stride):
      input_chunk = token_ids[i:i + max_length]
      target_chunk = token_ids[i + 1: i + max_length + 1]
      self.input_ids.append(torch.tensor(input_chunk))
      self.target_ids.append(torch.tensor(target_chunk))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.input_ids[idx], self.target_ids[idx]