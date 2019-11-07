import torch.nn as nn
import torch.nn.functional as F
from pytorch_conv4d.conv4d import Conv4d as Conv4d_

def nix():
    pass

class LFSR(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.vsn = nn.Sequential([Conv4d(1, 64, (3, 3, 3, 3), (1, 1, 1, 1))] + \
                                 [SAAC() for _ in range(16)] + \
                                 [Conv4d(64, 64, (3, 3, 2, 2), (0, 0, 1, 1))])
        self.vrn = nn.Sequential([Conv4d_strang(1, 16, (3, 3, 2, 2), (0, 0, 1, 1)),
                                  Conv4d_strang(16, 64, (3, 3, 2, 2), (0, 0, 1, 1)),
                                  Conv4d(64, 60, (3, 3, 2, 2), (0, 0, 1, 1))])
        
        
    def forward(self, x):
        x_vsn = self.vsn(x)
        
        shape = x_vsn.shape
        
        x_res = x_vsn.permute(0, 2, 3, 1, 4, 5)
        x_res = x_res.reshape(shape[0], 1, shape[2], shape[3], 60)
        
        x_con = torch.empty(shape[0], 1, shape[2], shape[3], 64)
        x_con[..., 0] = x[..., 0, 0]
        x_con[..., 7] = x[..., 1, 0]
        x_con[..., 56] = x[..., 0, 1]
        x_con[..., 63] = x[..., 1, 1]
        x_con[..., 1:7] = x_res[..., 0:6]
        x_con[..., 8:56] = x_res[..., 6:54]
        x_con[..., 57:63] = x_res[..., 54:]
        x_con = x_con.reshape(shape[0], 1, shape[2], shape[3], 8, 8)
        
        x_vrn = self.vrn(x_con)
        
        x_res = (x_vsn + x_vsn).permute(0, 2, 3, 1, 4, 5)
        x_res = x_res.reshape(shape[0], 1, shape[2], shape[3], 60)
        
        x_con = torch.empty(shape[0], 1, shape[2], shape[3], 64)
        x_con[..., 0] = x[..., 0, 0]
        x_con[..., 7] = x[..., 1, 0]
        x_con[..., 56] = x[..., 0, 1]
        x_con[..., 63] = x[..., 1, 1]
        x_con[..., 1:7] = x_res[..., 0:6]
        x_con[..., 8:56] = x_res[..., 6:54]
        x_con[..., 57:63] = x_res[..., 54:]
        x_con = x_con.reshape(shape[0], 1, shape[2], shape[3], 8, 8)
        
        return x_con
        
        
    
    def initialize(self):
        with open('params.pkl', 'rb') as file:
            params = pickle.load(file)
            
            
class SAAC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv_s = Conv4d(64, 64, (3, 3, 1, 1), (0, 0, 1, 1))
        self.conv_a = Conv4d(64, 64, (1, 1, 3, 3), (1, 1))
        
    def forward(self, x):
        x = self.conv_s(x)
        x = self.conv_a(x)
        return x
    
    
class Conv4d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super().__init__()
        self.conv = Conv4d_(in_channels, out_channels, kernel_size)
        
    def forward(self, x):
        if self.padding is not None:
            x = F.pad(x, self.padding)
        x = self.conv(x)
        return x
    

class Conv4d_strang(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding=None):
        super().__init__()
        self.conv = _Conv4d(in_channels, out_channels, kernel_size)
        
       
    def forward(self, x):
        shape = x.shape
        n_xs = shape[4]/2
        n_ys = shape[5]/2
        
        x_res = torch.empty(shape[0] * n_xs * n_ys, shape[1], shape[2], shape[3], 2, 2)
        for i in range(nx):
            for j in range(ny):
                x_res[(j*nx+i)*shape[0], ...] = x[..., i*2:(i+1)*2, j*2:(j+1)*2]
                
        x_res = self.conv(x_res)
        
        x = x_res.reshape(shape[0], nx, ny, shape[1], shape[2], shape[3])
        x = x.permute(0, 3, 4, 5, 1, 2)
        return x
        
        