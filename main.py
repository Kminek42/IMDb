from datasets import load_dataset
import data_manager as dm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

t0 = time.time()
train = True
bag_size = 1024
if train:
    dataset = dm.IMDb_Dataset(train=True, bag_size=bag_size)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True
    )
    
    model = nn.Sequential(
        nn.Dropout(),
        nn.Linear(bag_size, 2 * bag_size + 1),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(2 * bag_size + 1, 2 * bag_size + 1),
        nn.ReLU(),
        nn.Linear(2 * bag_size + 1, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_n = 6

    print("Start training...")

    t1 = time.time()
    for epoch in range(1, epoch_n + 1):
        loss_sum = 0
        for data_in, target in iter(dataloader):
            prediction = model.forward(data_in)

            optimizer.zero_grad()
            loss = criterion(prediction, target)
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
    dataset = dm.IMDb_Dataset(train=False, bag_size=bag_size)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=False
    )

    model = torch.load(f="model.pth")
    model = model.eval()
    print("Start testing...")

    all = 0
    good = 0

    for data_in, target in iter(dataloader):
        prediction = model.forward(data_in)

        for i in range(len(prediction)):
            if torch.argmax(prediction[i]) == target[i]:
                good += 1
            
            all += 1


    print("Testing completed.")
    print(f"Accuracy: {100 * good / all}%")
