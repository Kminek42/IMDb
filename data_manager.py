import torch
from torch.utils.data import Dataset
import os.path
from datasets import load_dataset

def extract_dataset(dataset):
    # this function converts dataset into 2 arrays: reviews and labels

    reviews = []
    labels = []

    for i in range(len(dataset)):
        reviews.append(dataset[i]["text"])
        labels.append(dataset[i]["label"])

    return reviews, labels

def genearte_BoW(*, dataset, bag_size, blocklist):
    # this function generates bag of words from dataset
    # It counts how many times a word occurred in positive and negative reviews 
    # and selects those words whose difference in occurrence in 
    # positive and negative reviews is the greatest 
    # (this words have probably the most impact)

    # reviews: [review_1, review_2, ..., review_n]
    # labels: [label_1, labe_2, ..., label_n]
    # gab_size: how many words are in the BoW
    # blocklist: words that are not couted

    reviews, labels = extract_dataset(dataset)

    word_dict = {}

    for i, review in enumerate(reviews):
        for word in review.split():
            if word in word_dict:
                word_dict[word] += 2 * labels[i] - 1
            
            else:
                word_dict[word] = 2 * labels[i] - 1
    
    for word in word_dict:
        # get abs value of diff
        if word_dict[word] < 0:
            word_dict[word] = -word_dict[word]
        
        # remove useless words
        if word in blocklist:
            word_dict[word] = 0

        if len(word) < 3:
            word_dict[word] = 0
    
    # return only words as a lsit
    word_dict = dict(sorted(word_dict.items(), reverse=True, key=lambda item: item[1]))
    word_dict = list(word_dict.keys())[:bag_size]
    return word_dict

def get_words_density(text, bag):
    output = [0] * len(bag)
    s = 0
    for word in text.split():
        if word in bag:
            output[bag.index(word)] += 1
            s += 1
    
    return torch.tensor(output, dtype=torch.float)

class IMDb_Dataset(Dataset):

    def __init__(self, train, bag_size):
        super().__init__()
        filename = "./bag.csv"

        if os.path.exists(filename):
            print("Bag of words file found.")
            print("Loading bag of words...")
            file = open(file=filename, mode="r")
            BoW = file.read().split(sep="\n")[:-1]
            file.close()
            print(f"Loaded {len(BoW)} words.")

        else:
            print("Bag of words file not found.")
            if not train:
                print("Train model before testing it.")
                exit()
            
            print("Generating bag of words...")
            dataset = load_dataset("imdb", split="train")
            BoW = genearte_BoW(dataset=dataset, bag_size=bag_size, blocklist=["/><br"])
            print("Bag of words generated.")
            file = open(file=filename, mode = "w")
            for word in BoW:
                file.write(f"{word}\n")
            
            file.close()

            print("Bag of words saved.")

        if train:
            dataset = load_dataset("imdb", split="train")
        
        else:
            dataset = load_dataset("imdb", split="test")

        print("Extracting dataset...")
        reviews, labels = extract_dataset(dataset)

        print("Converting reviews to vectors...")
        for i in range(len(reviews)):
            reviews[i] = get_words_density(reviews[i], BoW)

        self.reviews = reviews
        self.labels = labels
        self.len = len(reviews)

    def __getitem__(self, index):
        return self.reviews[index], self.labels[index]

    def __len__(self):
        return self.len