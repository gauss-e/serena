import torch
from dataset.inputs_targets_pair import tokenizer

vocab_size = tokenizer.n_vocab
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

pos_embedding_layer = torch.nn.Embedding(4, output_dim)
pos_embedding = pos_embedding_layer(torch.arange(4))

def get_token_embeddings(inputs: torch.Tensor) -> torch.Tensor:
  return token_embedding_layer(inputs)

def get_input_embeddings(token_embeddings) -> torch.Tensor:
  return token_embeddings + pos_embedding