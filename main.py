import urllib.request
import dataset

def main():
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")

    with urllib.request.urlopen(url) as response:
        raw_text = response.read().decode("utf-8")

    dataloader = dataset.create_dataloader(raw_text, batch_size=1,
                                          max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)


if __name__ == "__main__":
    main()
