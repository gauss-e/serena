import urllib.request

import tiktoken
import torch

import dataset
from model.attention.self_attention import SelfAttention
from model.embeddings import generate_inputs_embeddings, embed_tokens
from model.gpt_model import DummyGPTModel, LayerNorm

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
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(logits)
    print(logits.shape)

    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)

    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
    torch.set_printoptions(sci_mode=False)
    print("Mean:", mean)
    print("Variance:", var)


if __name__ == "__main__":
    main()
