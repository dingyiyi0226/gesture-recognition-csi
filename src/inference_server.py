import time

import torch
from torch.utils.data import DataLoader

from dataset import CsiDataSet
import models

class InferenceServer():
    def __init__(self, root):

        self.root = root
        model_path = 'models/model-2.pkl'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.CNN2()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
        self.model.to(self.device)
        self.model.eval()

    def inference(self, files, verbose=True):
        dataset = CsiDataSet(root=self.root, files=files)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        pred_all = []

        start_time = time.time()

        with torch.no_grad():
            for idx, (data, label) in enumerate(loader):
                output = self.model(data.to(self.device))
                pred = output.argmax(dim=1).cpu().detach()
                pred_all.extend(pred.tolist())

        end_time = time.time()
        pred_all = [dataset.get_label(p) for p in pred_all]

        if verbose:
            print(f'Predict: {pred_all}')
            print(f'Inference time: {end_time-start_time:.2f} (sec)')

        return pred_all


if __name__ == '__main__':
    server = InferenceServer('data/')
    server.inference('clap/199.pcap')
