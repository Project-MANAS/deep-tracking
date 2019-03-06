import torch
import torch.nn as nn

from utils import SpatialTransformerModule, get_convolution_filters


class DeepTracker(nn.Module):
    def __init__(self, hidden_dims: tuple, spatial_transform: bool = False, *args, **kwargs):
        """
        :param hidden_dims: expected to be (layers, batch_size, hidden_channels, x_img, y_img)
        :param spatial_transform: Bool, set to True to use an STM based on Odom
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.hidden_dims = hidden_dims
        self.hidden = self.init_hidden()
        self.spatial_transform = spatial_transform

        super().__init__()

        if self.spatial_transform:
            assert torch.cuda.is_available()
            self.spatial_transformer_module = SpatialTransformerModule()

        self.nhl = hidden_dims[2]
        self.convs = get_convolution_filters(hidden_dims[0], self.nhl, self.num_gates)
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def init_hidden(self):
        raise NotImplementedError

    def _cell(self, *args):
        raise NotImplementedError

    def forward(self, inp, *args):
        raise NotImplementedError

    def detach_hidden_(self):
        raise NotImplementedError


class DeepTrackerLSTM(DeepTracker):
    def __init__(self, hidden_dims: tuple, spatial_transform: bool = False, peephole: bool = False, *args, **kwargs):
        """
        :param peephole: Bool, set to True to use a peephole connection in the recurrent connection
        """
        self.num_gates = 4
        super().__init__(hidden_dims, spatial_transform, args, kwargs)
        self.peephole = peephole

    def init_hidden(self):
        return torch.zeros(*self.hidden_dims).to(self.device), torch.zeros(*self.hidden_dims).to(self.device)

    def _cell(self, inp, h, c, conv):
        activations = conv(torch.cat([inp, c if self.peephole else h], 1))

        gates = torch.stack(torch.split(activations, self.nhl, 1))
        forget = self.sigmoid(gates[0])
        input = self.sigmoid(gates[1])
        output = self.sigmoid(gates[2])
        gate = self.tanh(gates[3])

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

            inp[:, 1] = self.spatial_transformer_module(inp[:, 1].unsqueeze(1), affine_matrix).squeeze(1)

        hs, cs = [], []
        for i in range(self.hidden_dims[0]):
            h_i, c_i = self._cell(inp, h[i], c[i], self.convs[i])
            hs.append(h_i)
            cs.append(c_i)
            inp = h_i

        self.hidden = (torch.stack(hs, 0), torch.stack(cs, 0))

        return self.sigmoid(self.convs[self.hidden_dims[0]](torch.cat(hs, 1)))

    def detach_hidden_(self):
        self.hidden = tuple(map(lambda ten: ten.detach(), self.hidden))


class DeepTrackerGRU(DeepTracker):
    def __init__(self, hidden_dims: tuple, spatial_transform: bool = False, *args, **kwargs):
        self.num_gates = 2
        super().__init__(hidden_dims, spatial_transform, args, kwargs)

    def init_hidden(self):
        return torch.zeros(*self.hidden_dims).to(self.device)

    def _cell(self, inp, h, conv):
        activations = self.sigmoid(conv(torch.cat([inp, h], 1)))

        update, reset = torch.split(activations, self.nhl, 1)
        return torch.mul(update, h) + self.tanh(torch.mul(reset, h))

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
            inp[:, 1] = self.spatial_transformer_module(inp[:, 1].unsqueeze(1), affine_matrix).squeeze(1)

        hs = []
        for i in range(self.hidden_dims[0]):
            h_i = self._cell(inp, h[i], self.convs[i])
            hs.append(h_i)
            inp = h_i

        self.hidden = torch.stack(hs, 0)

        return self.sigmoid(self.convs[self.hidden_dims[0]](torch.cat(hs, 1)))

    def detach_hidden_(self):
        self.hidden = self.hidden.detach()
