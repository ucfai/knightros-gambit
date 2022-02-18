from urllib.request import urlretrieve
import torch


def get_figshare_dataset(store_path):
    figshare_url = "https://figshare.com/ndownloader/files/34101248"

    print(urlretrieve(figshare_url, store_path))
    return torch.load(store_path)


def main():
    get_figshare_dataset("./test_stockfish_data.pt")


if __name__ == "__main__":
    main()


