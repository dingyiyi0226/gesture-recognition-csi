import torch
import torch.nn as nn

class CNN2(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN2, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.2),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.2),
        )

        self.fc = nn.Sequential(
            nn.Linear(128*16*18, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out = self.cnn(x)
        feature = torch.flatten(out, 1)
        out = self.fc(feature)
        return out

class CNN1(nn.Module):
    def __init__(self, num_classes=7):
        super(CNN1, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),
            nn.Dropout(0.2),

            nn.Conv2d(128, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.Conv2d(192, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Dropout(0.2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.Dropout(0.25),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out = self.cnn(x)
        feature = torch.flatten(out, 1)
        out = self.fc(feature)
        return out
