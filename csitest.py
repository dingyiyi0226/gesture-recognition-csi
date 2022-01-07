import numpy as np
import matplotlib.pyplot as plt

from CSIKit.reader import get_reader
from CSIKit.util import csitools
from CSIKit.tools.get_info import display_info

file = "data/circle/5.pcap"
# file = "output_m1.pcap"
# file = "brushteeth_1590158472.dat"


METRIC = "amplitude"
# METRIC = "phase"

my_reader = get_reader(file)
csi_data = my_reader.read_file(file)
csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric=METRIC)
timestamps = csi_data.timestamps

metadata = csi_data.get_metadata()

print(no_frames, no_subcarriers)
print(csi_matrix.shape)


if len(csi_matrix.shape) == 4:
    # if more than 1Tx, 1Rx, use the first Tx, Rx
    csi_matrix = np.transpose(csi_matrix[:, :, 0, 0])
else:
    csi_matrix = np.transpose(csi_matrix)

csi_matrix = np.clip(csi_matrix, -20, csi_matrix.max())  # clip the small values

x = [t-timestamps[0] for t in timestamps]

plt.figure(figsize=(8, 3))
plt.subplots_adjust(left=0.1, right=1, bottom=0.2)

plt.imshow(csi_matrix, extent=[0, max(x), 1, no_subcarriers], aspect='auto', cmap='jet')

plt.xlabel("Time (s)")
plt.ylabel("Subcarrier Index")

plt.title(csi_data.filename)


if METRIC == "amplitude":
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (dBm)')
else:
    cbar = plt.colorbar(ticks=[-np.pi, 0, np.pi])
    cbar.set_label('Phase')
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], labels=[r'$-\pi$', r'$-\frac{\pi}{2}$', 0, r'$\frac{\pi}{2}$', r'$\pi$'])

plt.show()

