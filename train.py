import torch
import os

from model import DeepTrackerGRU
from torch.utils.data import DataLoader
from dataloader import DeepTrackDataset
from utils import SpatialTransformerModule


epochs = 5
seq_len = 100
bptt_len = 20

batch_size = 2
img_dim = 51
print_interval = 10

#assert seq_len % bptt_len == 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = DeepTrackerGRU((3, batch_size, 16, img_dim, img_dim), False).to(device)
optimizer = torch.optim.Adam(dt.parameters())

dataset = DeepTrackDataset('./data.t7', bptt_len)
data = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=False)

bce_loss = torch.nn.BCELoss().to(device)


for i in range(epochs):
    torch.save(dt,'./saved_models/model_' + str(i+1) + '.pt')
    dt.hidden = dt.init_hidden()
    seq_count = 0
    epoch_loss = 0
    for batch_no, batch in enumerate(data):
        seq_count += 1
        dt.zero_grad()
        dt.hidden = dt.hidden.detach()
        loss = 0
        target = batch.transpose(0, 1).to(device)
        for j in range(bptt_len):
            output = dt(torch.zeros(target[j].size()) if j % 10 >= 5 else target[j])
            target_occ = target[j, :, 0] * target[j, :, 1] # ob * vis
            loss += bce_loss(output,target_occ.unsqueeze(1))

        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        if seq_count % seq_len == 0:
            dt.hidden = dt.init_hidden()

        if batch_no % print_interval == 0:
            print("Epoch: %d, Batch no: %d, Batch Loss: %f" % (i, batch_no, loss.data))

    print("Epoch: %d, Epoch Loss: %f" % (i, epoch_loss))

