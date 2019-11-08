import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_conv4d.conv4d import Conv4d as _Conv4d
import pickle
import numpy as np


class LFSR(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.vsn = nn.Sequential(Conv4d(1, 64, (3, 3, 3, 3), (1, 1, 1, 1, )),
                                 nn.LeakyReLU(0.2),
                                 *[SAAC() for _ in range(16)],
                                 Conv4d(64, 60, (3, 3, 2, 2), (0, 0, 1, 1)))
        self.vrn = nn.Sequential(Conv4d_strang(1, 16, (3, 3, 2, 2), (0, 0, 1, 1)),
                                 nn.LeakyReLU(0.2),
                                 Conv4d_strang(16, 64, (3, 3, 2, 2), (0, 0, 1, 1)),
                                 nn.LeakyReLU(0.2),
                                 Conv4d(64, 60, (3, 3, 2, 2), (0, 0, 1, 1)))
        
        
    def forward(self, x):
        x_vsn = self.vsn(x)
        
        x_con = reshape(x, x_vsn)
                
        x_vrn = self.vrn(x_con)
        
        x_con = reshape(x, x_vsn + x_vrn)
        
        return x_con
        
        
    
    def initialize(self):
        with open('params.pkl', 'rb') as file:
            params = pickle.load(file)
            
        weight = params['pre_level1_conv6d_f']
        weight = np.moveaxis(weight, [-2, -1], [1, 0])
        bias = params['pre_level1_conv6d_b'][0]
        self.vsn[0].initialize(torch.tensor(weight), torch.tensor(bias))
        
        for i in range(16):
            weight = params['level' + str(i+1) + '_conv6d_f_mid']
            weight = np.moveaxis(weight, [-2, -1], [1, 0])
            bias = params['level' + str(i+1) + '_conv6d_b_mid'][0]
            self.vsn[i+2].conv_s.initialize(torch.tensor(weight), torch.tensor(bias))
            
            weight = params['level' + str(i+1) + '_conv6d_f_mid2']
            weight = np.moveaxis(weight, [-2, -1], [1, 0])
            bias = params['level' + str(i+1) + '_conv6d_b_mid2'][0]
            self.vsn[i+2].conv_a.initialize(torch.tensor(weight), torch.tensor(bias))
        
        weight = params['level1_conv6d_f']
        weight = np.moveaxis(weight, [-2, -1], [1, 0])
        bias = params['level1_conv6d_b'][0]
        self.vsn[-1].initialize(torch.tensor(weight), torch.tensor(bias))
        
        weight = params['pre_level1_conv6d_f_2']
        weight = np.moveaxis(weight, [-2, -1], [1, 0])
        bias = params['pre_level1_conv6d_b_2'][0]
        self.vrn[0].conv.initialize(torch.tensor(weight), torch.tensor(bias))
        
        weight = params['level1_conv6d_f_post']
        weight = np.moveaxis(weight, [-2, -1], [1, 0])
        bias = params['level1_conv6d_b_post'][0]
        self.vrn[2].conv.initialize(torch.tensor(weight), torch.tensor(bias))
        
        weight = params['level1_conv6d_f_post2']
        weight = np.moveaxis(weight, [-2, -1], [1, 0])
        bias = params['level1_conv6d_b_post2'][0]
        self.vrn[4].initialize(torch.tensor(weight), torch.tensor(bias))
        
            
class SAAC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv_s = Conv4d(64, 64, (3, 3, 1, 1), (0, 0, 1, 1))
        self.conv_a = Conv4d(64, 64, (1, 1, 3, 3), (1, 1))
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv_s(x)
        x = self.relu(x)
        x = self.conv_a(x)
        x = self.relu(x)
        return x
    
    
class Conv4d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super().__init__()
        self.conv = _Conv4d(in_channels, out_channels, kernel_size)
        self.padding = tuple([p for p in padding for _ in (0, 1)])
        
    def forward(self, x):
        if self.padding is not None:
            x = F.pad(x, self.padding)
        x = self.conv(x)
        return x
    
    def initialize(self, weight, bias):
        for i in range(weight.shape[2]):
            self.conv.conv3d_layers[i].weight = torch.nn.Parameter(weight[:, :, i, :, :, :])
        self.conv.bias = torch.nn.Parameter(bias)
    

class Conv4d_strang(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super().__init__()
        self.conv = Conv4d(in_channels, out_channels, kernel_size, padding)
        
       
    def forward(self, x):
        shape = x.shape
        nx = shape[4]//2
        ny = shape[5]//2
        
        x_res = torch.empty(shape[0] * nx * ny, shape[1], shape[2], shape[3], 2, 2)
        for i in range(nx):
            for j in range(ny):
                x_res[(j*nx+i)*shape[0]:(j*nx+i+1)*shape[0], ...] = x[..., i*2:(i+1)*2, j*2:(j+1)*2]
        
        x_res = self.conv(x_res)
        
        x = x_res.reshape(shape[0], nx, ny, x_res.shape[1], x_res.shape[2], x_res.shape[3])
        x = x.permute(0, 3, 4, 5, 1, 2)
        return x
        
        
def reshape(x_small, x_large):
    shape = x_large.shape

    x_res = x_large.permute(0, 2, 3, 1, 4, 5)
    x_res = x_res.reshape(shape[0], 1, shape[2], shape[3], 60)
    
    x_con = torch.empty(shape[0], 1, shape[2], shape[3], 64)
    x_con[..., 0] = x_small[..., 0, 0]
    x_con[..., 7] = x_small[..., 1, 0]
    x_con[..., 56] = x_small[..., 0, 1]
    x_con[..., 63] = x_small[..., 1, 1]
    x_con[..., 1:7] = x_res[..., 0:6]
    x_con[..., 8:56] = x_res[..., 6:54]
    x_con[..., 57:63] = x_res[..., 54:]
    x_con = x_con.reshape(shape[0], 1, shape[2], shape[3], 8, 8)
    return x_con
