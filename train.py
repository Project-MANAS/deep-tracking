import torch

from model import DeepTrackerGRU
from torch.utils.data import DataLoader
from dataloader import DeepTrackDataset

epochs = 5
seq_len = 200
bptt_len = 20

batch_size = 2
img_dim = 256

assert seq_len % bptt_len == 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = DeepTrackerGRU((3, batch_size, 16, img_dim, img_dim), False).to(device)
optimizer = torch.optim.Adam(dt.parameters())

dataset = DeepTrackDataset('./data.t7', seq_len)
data = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=False)

bce_loss = torch.nn.BCELoss()
loss = 0

for i in range(epochs):
    epoch_loss = 0
    loss = 0
    dt.hidden = dt.init_hidden()

    for batch_no, batch in enumerate(data):
        print(batch_no, '-->', batch.size())