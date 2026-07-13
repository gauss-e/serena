import urllib.request

import tiktoken
import torch

from model.gpt_model import LayerNorm, GPTModel

GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

def main():
    # url = ("https://raw.githubusercontent.com/rasbt/"
    #        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    #        "the-verdict.txt")
    #
    # with urllib.request.urlopen(url) as response:
    #     raw_text = response.read().decode("utf-8")
    #
    # dataloader = dataset.create_dataloader(raw_text, batch_size=8,
    #                                       max_length=4, stride=4, shuffle=False)
    # data_iter = iter(dataloader)
    # inputs, target = next(data_iter)
    # # print(inputs)
    #
    # print(generate_inputs_embeddings(embed_tokens(inputs)))
    #
    # self_attention = SelfAttention(4,2)
    # print(self_attention(inputs))
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    # print(batch)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(logits)
    print(logits.shape)

def generate_text_simple(model, idx, max_new_tokens, context_size):
  for _ in range(max_new_tokens):
    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
      logits = model(idx_cond)

    logits = logits[:, -1, :]
    probes = torch.softmax(logits, dim=-1)
    idx_next = torch.argmax(probes, dim=-1, keepdim=True)
    idx = torch.cat((idx_cond, idx_next), dim=-1)

  return idx


if __name__ == "__main__":
    main()
