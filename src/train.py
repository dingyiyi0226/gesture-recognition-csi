import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import models
import utils
from dataset import CsiDataSet


def train():

    EPOCH = 60
    ID = '3'

    dataset = CsiDataSet()

    val_n = int(len(dataset)*0.2)
    test_n = int(len(dataset)*0.2)
    train_n = len(dataset) - val_n - test_n
    print(f'Training set: {train_n}, Validation set: {val_n}, Testing set: {test_n}')
    train_set, val_set, test_set = random_split(dataset, [train_n, val_n, test_n])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.CNN2()
    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    loss_list = []
    acc_list = []

    for epoch in range(EPOCH):
        start_time = time.time()

        train_acc = 0.
        train_loss = 0.
        val_acc = 0.
        val_loss = 0.

        model.train()

        for idx, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.to(device))
            batch_loss = criterion(output, label.to(device))
            batch_loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1).cpu().detach()
            train_acc += pred.eq(label).sum().item()
            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for idx, (data, label) in enumerate(val_loader):
                output = model(data.to(device))
                batch_loss = criterion(output, label.to(device))

                pred = output.argmax(dim=1).cpu().detach()
                val_acc += pred.eq(label).sum().item()
                val_loss += batch_loss.item()

        scheduler.step()

        train_acc = train_acc/len(train_set)
        train_loss = train_loss/len(train_set)
        val_acc = val_acc/len(val_set)
        val_loss = val_loss/len(val_set)

        loss_list.append(val_loss)
        acc_list.append(val_acc)

        end_time = time.time()
        print(f'Epoch {epoch}: {end_time-start_time:2f} (sec)')
        print(f'     Train acc: {train_acc:.3f}, loss: {train_loss:.3f}')
        print(f'     Val   acc: {val_acc:.3f}, loss: {val_loss:.3f}')

    print(f'Best acc: {max(acc_list):.3f}')
    print('Loss:')
    for loss in loss_list:
        print(loss, end=' ')
    print(' ')

    torch.save(model.state_dict(), f'models/model-{ID}.pkl')
    utils.plotlist(loss_list, f'figs/loss-{ID}.png', 'Loss')

    # inference

    pred_all = []
    label_all = []
    test_acc = 0.
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_loader):
            output = model(data.to(device))
            pred = output.argmax(dim=1).cpu().detach()
            test_acc += pred.eq(label).sum().item()

            pred_all.extend(pred.tolist())
            label_all.extend(label.tolist())
    end_time = time.time()
    test_acc = test_acc/len(test_set)

    print(f'Inference time: {end_time-start_time:.2f} (sec)')
    print(f'Inference acc: {test_acc:.3f}')

    utils.confusion(pred_all, label_all, f'figs/conf-{ID}.png', num_classes=7)


if __name__ == '__main__':
    train()
