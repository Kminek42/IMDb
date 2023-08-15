import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import re

def save_words(wordlist):
    file = open("wordlist.txt", "w")
    for word in wordlist:
        file.write(f"{word}\n")
    
    file.close()

def load_words():
    file = open("wordlist.txt", "r")
    words = file.read().split()[:-1]
    file.close()

    return {word: i + 2 for i, word in enumerate(words)}

def tokenize(text, wordlist, output_len):
    output = [wordlist.get(word, 1) for word in text.split()]
    diff = output_len - len(output)
    return torch.tensor(output[:output_len] + [0] * diff)


class IMDb_Dataset(Dataset):
    def __init__(self, split, review_len):
        super().__init__()

        dataset = load_dataset("imdb", split=split)
        reviews = [" ".join(re.findall(r'\b\w+\b', example["text"].lower())) for example in dataset]
        self.labels = torch.tensor([example["label"] for example in dataset])

        if split == "train":
            words = sorted(list(set(" ".join(reviews).split())))
            save_words(words)

        words = load_words()
        self.words_n = len(words) + 2
        
        self.reviews = [tokenize(text, words, review_len) for text in reviews]
        self.len = len(self.reviews)

    def __getitem__(self, index):
        return self.reviews[index], self.labels[index]

    def __len__(self):
        return self.len
    