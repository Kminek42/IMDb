from datasets import load_dataset
import data_manager as dm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

train = True

bag_size = 256
if train:
    dataset = dm.IMDb_Dataset(train=True, bag_size=bag_size)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True
    )
    
    model = nn.Sequential(
        nn.Linear(bag_size, 2 * bag_size + 1),
        nn.ReLU(),
        nn.Linear(2 * bag_size + 1, 2 * bag_size + 1),
        nn.ReLU(),
        nn.Linear(2 * bag_size + 1, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epoch_n = 4

    print("Start training...")

    for epoch in range(1, epoch_n + 1):
        loss_sum = 0
        for data_in, target in iter(dataloader):
            prediction = model.forward(data_in)

            optimizer.zero_grad()
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss)

        print(loss_sum / (dataset.len / dataloader.batch_size))

        torch.save(obj=model.to("cpu"), f="model.pth")