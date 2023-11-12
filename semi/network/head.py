import torch.nn as nn
import torch


class LinearHead(nn.Module):
    def __init__(self, net, dim_out=1000, linear_eval=True, input_dim=None):
        super().__init__()
        self.net = net

        if input_dim is None:
            input_dim = net.out_dim

        self.fc = nn.Linear(input_dim, dim_out)
        
        if linear_eval:
            for param in self.net.parameters():
                param.requires_grad = False
            self.fc.weight.data.zero_()
            self.fc.bias.data.zero_()

        self.linear_eval = linear_eval

    def forward(self, x, return_feat=False):
        if self.linear_eval:
            with torch.no_grad():
                feat = self.net(x)
        else:
            feat = self.net(x)
        
        if return_feat:
            return self.fc(feat), feat
        return self.fc(feat)





class MLPV2(nn.Module):
    def __init__(self, input_dim=2048, out_dim=128, hidden_dim=None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
        )
        
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, return_feat=False):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.out(x2)
        if return_feat:
            return [x1, x2, x3]
        return x3




class MoCoHead(nn.Module):
    def __init__(self, input_dim=2048, out_dim=128, hidden_dim=2048):
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.out = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.out(x)
        return x