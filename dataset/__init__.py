from torch.utils.data import DataLoader
from dataset.inputs_targets_pair import GPTDataset


def create_dataloader(txt, batch_size=4, max_length=256, stride=128,
                      shuffle=True, drop_last=True, num_workers=0):
  dataset = GPTDataset(txt, max_length, stride)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=num_workers)
  return dataloader