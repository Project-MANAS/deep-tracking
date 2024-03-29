import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import DeepTrackDataset
from model import DeepTrackerLSTM
from weightedBCECriterion import WeightedBCE

epochs = 25
seq_len = 100

batch_size = 1 # DO NOT CHANGE
img_dim = 51
print_interval = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dt = DeepTrackerLSTM((3, batch_size, 16, img_dim, img_dim), False, False).to(device)
optimizer = torch.optim.Adagrad(dt.parameters(), 0.01)

dataset = DeepTrackDataset('./data.t7', seq_len)
data = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)

bce_loss = WeightedBCE().to(device)

zero_tensor = torch.zeros((batch_size, 2, img_dim, img_dim)).to(device)

def visualize(model, sequence, epoch_no):
    model.init_hidden()
    outputs = []
    with torch.no_grad():
        for inp in sequence:
            output = model(torch.unsqueeze(inp, 0).to(device)).data
            output_np = np.array(output)[0][0] * 255
            outputs.append(output_np.astype('uint8'))
    imageio.mimsave('./saved_gifs/gif_' + str(epoch_no + 1) + '.gif', outputs)

for i in range(epochs):
    epoch_loss = 0
    for batch_no, batch in enumerate(data):
        dt.hidden = dt.init_hidden()
        dt.zero_grad()
        loss = 0
        target = batch.transpose(0, 1).to(device)
        for j in range(seq_len):
            output = dt(zero_tensor if j % 10 >= 5 else target[j])
            loss += bce_loss(target[j], output)

        epoch_loss += loss.data
        loss.backward()
        optimizer.step()

        if batch_no % print_interval == 0:
            print("Epoch: %d, Batch no: %d, Batch Loss: %f" % (i, batch_no, loss.data))

    print("Epoch: %d, Epoch Loss: %f" % (i, epoch_loss / len(data)))
    visualize(dt, dataset[0], i)
    torch.save(dt, './saved_models/model_' + str(i + 1) + '.pt')
