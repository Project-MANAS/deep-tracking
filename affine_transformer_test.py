import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision

class SpatialTransformerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, theta):
        # img.shape --> (1, 3, 1920, 1080)
        # theta.shape --> (1, 2, 3)
        grid = F.affine_grid(theta, img.size())
        
        return F.grid_sample(img, grid)

stm = SpatialTransformerModule()

'''
affine transformation matrix 

 cos(theta) sin(theta) x
-sin(theta) cos(theta) y

where theta is closewise angle
x and y are relative translation

'''
aff = np.array([[[0.866, 0.5, 0], [-1, 0.866, 0]]], np.float32)
theta = torch.from_numpy(aff)

img = Image.open("/home/lastjedi/Pictures/view.png")

img = torch.from_numpy(np.array(img, np.float32)).transpose(0, 2).transpose(1, 2)
print(img.size())
output = stm(torch.unsqueeze(img, 0), theta)

img = torchvision.transforms.functional.to_pil_image(output[0])

img.show()

