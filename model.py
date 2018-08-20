import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, affine_matrix):
        affine_grid = F.affine_grid(affine_matrix, img.size())
        return F.grid_sample(img, affine_grid)


class DeepTrackerLSTM(nn.Module):
    def __init__(self, hidden_dims: tuple, spatial_transform: bool = False, peephole: bool = False):
        """
        :param hidden_dims: expected to be (3, batch_size, 16, x_img, y_img)
        :param spatial_transform: Bool, set to True to use an STM based on Odom
        :param peephole: Bool, set to True to use a peephole connection in the recurrent connection
        """

        self.hidden_dims = hidden_dims
        self.hidden = self.init_hidden()
        self.spatial_transform = spatial_transform
        self.peephole = peephole

        super().__init__()

        if self.spatial_transform:
            self.spatial_transformer_module = SpatialTransformerModule()

        self.nhl = hidden_dims[2]
        self.conv1 = torch.nn.Conv2d(2 + self.nhl, 4 * self.nhl, 3, 1, 1, 1)
        self.conv2 = torch.nn.Conv2d(2 * self.nhl, 4 * self.nhl, 3, 1, 2, 2)
        self.conv3 = torch.nn.Conv2d(2 * self.nhl, 4 * self.nhl, 3, 1, 4, 4)
        self.conv4 = torch.nn.Conv2d(3 * self.nhl, 1, 3, 1, 1, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        # TODO (squadrick): Allow theis to be a param as well
        assert hidden_dims[0] == 3

    def init_hidden(self):
        return (torch.zeros(*self.hidden_dims).cuda(),
                torch.zeros(*self.hidden_dims).cuda())

    def _cell(self, inp, h, c, conv):
        activations = conv(torch.cat([inp, c if self.peephole else h], 1))

        gates = torch.stack(torch.split(activations, self.nhl, 1))
        forget = self.sigmoid(gates[0])
        input = self.sigmoid(gates[1])
        output = self.sigmoid(gates[2])
        gate = gates[3] if self.peephole else self.tanh(gates[3])

        c_new = torch.mul(forget, c) + torch.mul(input, gate)
        h_new = torch.mul(output, c_new)

        return h_new, c_new

    def forward(self, inp, *args):
        """
        :param inp: Input tensor of size (batch_size, 2, width, height)
                    dim_1 has the observation(0) and the visibility layer(1)
        :param args: affine transformation matrix but only if self.spatial_transform is True
        :return: Output tensor of size (batch_size, 1, width, height)
        """
        h, c = self.hidden

        if self.spatial_transform:
            affine_matrix = args[0]

            h = torch.stack([self.spatial_transformer_module(i, affine_matrix) for i in h])
            c = torch.stack([self.spatial_transformer_module(i, affine_matrix) for i in c])
            '''NOTE (squadrick): I know it looks like calling a torch.stack on a list comprehension
               will be slower, but it is not. It's faster than direct index and assign or using a loop. 
               Idk why. '''

            # TODO (Squadrick): Test whether the shifting of visibility layer is successful
            inp[:, 1] = self.spatial_transformer_module(inp[:, 1].unsqueeze(1), affine_matrix).squeeze(1)

        h1, c1 = self._cell(inp, h[0], c[0], self.conv1)
        h2, c2 = self._cell(h1, h[1], c[1], self.conv2)
        h3, c3 = self._cell(h2, h[2], c[2], self.conv3)

        h_pred = torch.cat([h1, h2, h3], 1)
        h_new = torch.stack([h1, h2, h3], 0)
        c_new = torch.stack([c1, c2, c3], 0)

        self.hidden = (h_new, c_new)

        return self.sigmoid(self.conv4(h_pred))


class DeepTrackerGRU(nn.Module):
    def __init__(self, hidden_dims: tuple, spatial_transform: bool = False):
        """
        :param hidden_dims: expected to be (3, batch_size, 16, x_img, y_img)
        :param spatial_transform: Bool, set to True to use an STM based on Odom
        """
        self.hidden_dims = hidden_dims
        self.hidden = self.init_hidden()
        self.spatial_transform = spatial_transform

        super().__init__()

        if self.spatial_transform:
            self.spatial_transformer_module = SpatialTransformerModule()

        self.nhl = hidden_dims[2]
        self.conv1 = torch.nn.Conv2d(2 + self.nhl, 3 * self.nhl, 3, 1, 1, 1)
        self.conv2 = torch.nn.Conv2d(2 * self.nhl, 3 * self.nhl, 3, 1, 2, 2)
        self.conv3 = torch.nn.Conv2d(2 * self.nhl, 3 * self.nhl, 3, 1, 4, 4)
        self.conv4 = torch.nn.Conv2d(3 * self.nhl, 1, 3, 1, 1, 1)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        # TODO (squadrick): Allow theis to be a param as well
        assert hidden_dims[0] == 3

    def init_hidden(self):
        return torch.zeros(*self.hidden_dims).cuda()

    def _cell(self, inp, h, conv):
        activations = conv(torch.cat([inp, h], 1))

        gates = torch.stack(torch.split(activations, self.nhl, 1))
        update = self.sigmoid(gates[0])
        reset = self.sigmoid(gates[1])
        h_new = torch.mul(update, h) + self.tanh(torch.mul(reset, h))

        return h_new

    def forward(self, inp, *args):
        """
        :param inp: Input tensor of size (batch_size, 2, width, height)
                    dim_1 has the observation(0) and the visibility layer(1)
        :param args: affine transformation matrix but only if self.spatial_transform is True
        :return: Output tensor of size (batch_size, 1, width, height)
        """
        h = self.hidden

        if self.spatial_transform:
            affine_matrix = args[0]

            h = torch.stack([self.spatial_transformer_module(i, affine_matrix) for i in h])
            # TODO (squadrick): Test whether the shifting of visibility layer is successful
            inp[:, 1] = self.spatial_transformer_module(inp[:, 1].unsqueeze(1), affine_matrix).squeeze(1)

        h1 = self._cell(inp, h[0], self.conv1)
        h2 = self._cell(h1, h[1], self.conv2)
        h3 = self._cell(h2, h[2], self.conv3)

        h_pred = torch.cat([h1, h2, h3], 1)
        h_new = torch.stack([h1, h2, h3], 0)

        self.hidden = h_new

        return self.sigmoid(self.conv4(h_pred))
