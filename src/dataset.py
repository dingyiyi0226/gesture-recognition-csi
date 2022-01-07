import glob
import os

from CSIKit.reader import NEXBeamformReader
from CSIKit.util import csitools
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CsiDataSet(Dataset):
    def __init__(self, root='data/', files=None):

        self.root = root
        self.file = []
        self.label = []
        self.reader = NEXBeamformReader()
        self.labels = ['circle', 'clap', 'kick', 'push', 'sit', 'slide', 'stand']
        self.transform = None

        if files is None:
            """ Training """

            for idx, label in enumerate(self.labels):
                files = [os.path.relpath(i, root) for i in glob.glob(f'{root}/{label}/*.pcap')]
                files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

                self.file.extend(files)
                self.label.extend([idx for _ in files])
                self.transform = transforms.RandomAffine(0, translate=(0.3, 0))
        else:
            """ Inferencing """
            if type(files) is list:
                self.file.extend(files)
            else:
                self.file.append(files)

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        file = os.path.join(self.root, self.file[index])

        csi_data = self.reader.read_file(file)
        csi_matrix, _, _ = csitools.get_CSI(csi_data, metric='amplitude')
        csi_matrix = torch.from_numpy(csi_matrix).float()
        csi_matrix = csi_matrix[:, 64:192, 0, 0].T  # shape 128*150
        csi_matrix = csi_matrix.clamp(min=-20)  # clamp the small values
        csi_matrix = csi_matrix.reshape(1, 128, 150)

        if self.transform is not None:
            csi_matrix = self.transform(csi_matrix)

        if self.label:
            return csi_matrix, self.label[index]
        else:
            return csi_matrix, -1

    def get_label(self, idx):
        return self.labels[idx]
