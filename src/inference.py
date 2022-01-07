import time

import torch
from torch.utils.data import DataLoader

from dataset import CsiDataSet
import models


def inference(files):

    model_path = 'models/model-2.pkl'

    start_time = time.time()

    dataset = CsiDataSet(root='data/', files=files)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.CNN2()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)

    model.eval()
    pred_all = []

    with torch.no_grad():
        for idx, (data, label) in enumerate(loader):
            output = model(data.to(device))
            pred = output.argmax(dim=1).cpu().detach()
            pred_all.extend(pred.tolist())

    end_time = time.time()
    print(f'Predict: {pred_all}')
    print(f'Inference time: {end_time-start_time:.2f} (sec)')


if __name__ == '__main__':
    inference('clap/199.pcap')
