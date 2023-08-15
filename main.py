from datasets import load_dataset
import data_manager as dm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

t0 = time.time()
REVIEW_LEN = 2048
EMBEDDING_DIM = 32

train = True

class ConvBlock(nn.Module):
    def __init__(self, *args, channels, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Conv = nn.Conv1d(channels, channels, 3, padding="same")
        self.Activation = nn.ReLU()
        self.Poll = nn.MaxPool1d(2)

    def forward(self, X):
        X = self.Conv(X)
        X = self.Poll(X)
        X = self.Activation(X)
        return X

class TextConvNet(nn.Module):
    def __init__(self, *args, input_len, embedding_dim, dict_size, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.Embedding = nn.Embedding(dict_size, embedding_dim)

        self.ConvBLock1 = ConvBlock(channels=embedding_dim)
        self.ConvBLock2 = ConvBlock(channels=embedding_dim)
        self.ConvBLock3 = ConvBlock(channels=embedding_dim)
        self.ConvBLock4 = ConvBlock(channels=embedding_dim)

        self.Lin1 = nn.Linear(embedding_dim * input_len // 16, input_len // 8)
        self.Lin2 = nn.Linear(input_len // 8, 2)

        self.Activation = nn.ReLU()

    def forward(self, X):
        X = self.Embedding(X).mT

        X = self.ConvBLock1(X)
        X = self.ConvBLock2(X)
        X = self.ConvBLock3(X)
        X = self.ConvBLock4(X)

        X = nn.Flatten()(X)
        X = self.Lin1(X)
        X = self.Activation(X)
        X = self.Lin2(X)

        return X

dev = torch.device("cpu")
if torch.cuda.is_available():
    dev = torch.device("cuda")

elif torch.backends.mps.is_available():
    dev = torch.device("mps")

print(f"Using {dev} device")

if train:
    dataset = dm.IMDb_Dataset("train", REVIEW_LEN)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True
    )
    
    model = TextConvNet(input_len=REVIEW_LEN, embedding_dim=EMBEDDING_DIM, dict_size=dataset.words_n).to(dev)
    try:
        model = torch.load("model.pth").to(dev)
    except:
        pass

    print(model)
    input()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_n = 4

    print("Start training...")

    t1 = time.time()
    for epoch in range(1, epoch_n + 1):
        loss_sum = 0
        for i, (data_in, target) in enumerate(iter(dataloader)):
            prediction = model.forward(data_in.to(dev))

            optimizer.zero_grad()
            loss = criterion(prediction, target.to(dev))
            loss.backward()
            optimizer.step()

            loss_sum += float(loss)

        print(f"\n{100 * epoch // epoch_n}%")
        print(loss_sum / (dataset.len / dataloader.batch_size))

    t1 = time.time() - t1
    print("Training is completed.")
    torch.save(obj=model.to("cpu"), f="model.pth")
    print("Model saved.")

    t0 = time.time() - t0

    print(f"Total time: {t0}s")
    print(f"Learning time: {t1}s")

else:
    dataset = dm.IMDb_Dataset("test", REVIEW_LEN)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False
    )

    model = torch.load(f="model.pth").to(dev)
    model = model.eval()
    print("Start testing...")

    all = 0
    good = 0

    for data_in, target in iter(dataloader):
        prediction = model.forward(data_in.to(dev))
        target = target.to(dev)
        for i in range(len(prediction)):
            if torch.argmax(prediction[i]) == target[i]:
                good += 1
            
            all += 1


    print("Testing completed.")
    print(f"Accuracy: {100 * good / all}%")
