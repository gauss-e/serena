import urllib.request
import dataset
from model.attention.self_attention import SelfAttention
from model.embeddings import generate_inputs_embeddings, embed_tokens


def main():
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")

    with urllib.request.urlopen(url) as response:
        raw_text = response.read().decode("utf-8")

    dataloader = dataset.create_dataloader(raw_text, batch_size=8,
                                          max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, target = next(data_iter)
    # print(inputs)

    print(generate_inputs_embeddings(embed_tokens(inputs)))

    self_attention = SelfAttention(4,2)
    print(self_attention(inputs))


if __name__ == "__main__":
    main()
