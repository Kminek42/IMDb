from datasets import load_dataset


def get_BoW(*, reviews, labels, bag_size, blocklist):
    # this function generates bag of words from dataset
    # It counts how many times a word occurred in positive and negative reviews 
    # and selects those words whose difference in occurrence in 
    # positive and negative reviews is the greatest 
    # (this words have probably the most impact)

    # reviews: [review_1, review_2, ..., review_n]
    # labels: [label_1, labe_2, ..., label_n]
    # gab_size: how many words are in the BoW
    # blocklist: words that are not couted

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

def extract_dataset(dataset):
    # this function converts dataset into 2 arrays: reviews and labels

    reviews = []
    labels = []

    for i in range(len(dataset)):
        reviews.append(dataset[i]["text"])
        labels.append(dataset[i]["label"])

    return reviews, labels

blocklist = ["/><br"]

train = True

if train:
    dataset = load_dataset("imdb", split="train")
    reviews, labels = extract_dataset(dataset)
    BoW = get_BoW(reviews=reviews, labels=labels, bag_size=64, blocklist=blocklist)
    print(BoW)

