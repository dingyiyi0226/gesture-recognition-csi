from CSIKit.reader import NEXBeamformReader
from CSIKit.util import csitools
import numpy as np
import matplotlib.pyplot as plt
import torch

def read_file(file):
    print(file)
    csi_data = NEXBeamformReader().read_file(file)
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric='amplitude')

    print(csi_data.get_metadata())

    csi_matrix = torch.from_numpy(csi_matrix)
    csi_matrix = csi_matrix[:, 64:192, 0, 0].T  # shape 128*150
    csi_matrix = csi_matrix.clamp(min=-20)  # clamp the small values
    # print(csi_matrix.size())

    plot_data(csi_matrix, title=file)

def plot_data(csi_matrix, title=''):
    plt.figure(figsize=(8, 3))
    plt.subplots_adjust(left=0.1, right=1, bottom=0.2)

    plt.imshow(csi_matrix, aspect='auto', cmap='inferno')

    plt.xlabel("Time (s)")
    plt.ylabel("Subcarrier Index")
    plt.title(title)

    cbar = plt.colorbar()
    cbar.set_label('Amplitude (dBm)')

    plt.show()

def plotlist(plotlist, path, title):
    plt.plot(plotlist)
    plt.title(title)
    plt.savefig(path)
    plt.close()

def confusion(pred, label, path, num_classes):
    confusion = np.zeros((num_classes, num_classes))
    for o, t in zip(pred, label):
        confusion[o, t] += 1
    confusion = confusion/(np.sum(confusion, axis=0)+1e-6)

    plt.imshow(confusion)
    plt.colorbar()
    plt.title('Confusion matrix')
    plt.xlabel('Label')
    plt.ylabel('Predict')
    plt.savefig(path)
    plt.close()
