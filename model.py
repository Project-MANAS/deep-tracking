import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, affine_matrix):
        affine_grid = F.affine_grid(affine_matrix, img.size())
        return F.grid_sample(img, affine_grid)


class DeepTracker(nn.Module):
    def __init__(self, hidden_dims: tuple, spatial_transform: bool = False):
        """
        :param hidden_dims: expected to be (batch_size, 3, 16, x_img, y_img)
        :param spatial_transform: (
        """
        # hidden_dims is expected to be (batch, 3, 16, x_img, y_img)
        self.hidden_dims = hidden_dims
        self.hidden = self.init_hidden()
        self.spatial_transform = spatial_transform

        super().__init__()

        self.conv1 = torch.nn.Conv2d(18, 64, 3, 1, 1, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 2, 2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 1, 4, 4)
        self.conv4 = torch.nn.Conv2d(48, 1, 3, 1, 1, 1)

        if self.spatial_transform:
            self.spatial_transformer_module = SpatialTransformerModule()

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def init_hidden(self):
        return (torch.zeros(*self.hidden_dims).cuda(),
                torch.zeros(*self.hidden_dims).cuda())
    
    def _cell(self, inp, h, c, conv):
        net = torch.cat([inp, h], 1)
        net = conv(net)

        gates = torch.stack(torch.split(net, 16, 1))
        gates[0] = self.sigmoid(gates[0])
        gates[1] = self.sigmoid(gates[1])
        gates[2] = self.sigmoid(gates[2])
        gates[3] = self.tanh(gates[3])

        c_forget = torch.mul(c, gates[0])
        c_input = torch.mul(gates[1], gates[3])
        c_new = torch.add(c_input, c_forget)
        c_transform = self.tanh(c_new)
        h_new = torch.mul(gates[2], c_transform)

        return h_new, c_new

    def forward(self, sequence, *args):
        h = self.hidden[0]
        c = self.hidden[1]

        if self.spatial_transform:
            affine_matrix = args[0]

            # Spatial transform both the hidden and cell state
            h = torch.stack([self.spatial_transformer_module(i, affine_matrix) for i in h])
            c = torch.stack([self.spatial_transformer_module(i, affine_matrix) for i in c])
            '''NOTE (squadrick): I know it looks like calling a torch.stack on a list comprehension
               will be slower, but it is not. It's faster than direct index and assign or using a loop. 
               Idek why. '''

            # TODO (squadrick): spatial transform the visibility layer on GPU

        h1, c1 = self._cell(sequence, h[0], c[0], self.conv1)
        h2, c2 = self._cell(h1, h[1], c[1], self.conv2)
        h3, c3 = self._cell(h2, h[2], c[2], self.conv3)

        h_pred = torch.cat([h1, h2, h3], 1)
        h_new = torch.stack([h1, h2, h3], 0)
        c_new = torch.stack([c1, c2, c3], 0)
        
        self.hidden = (h_new, c_new)
        
        return self.sigmoid(self.conv4(h_pred))


