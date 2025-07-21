import torch
import torch.nn as nn

class SimpleTCN(nn.Module):
    def __init__(self, input_dim, num_blocks, num_filters, kernel_size, dropout):
        super(SimpleTCN, self).__init__()
        layers = []
        in_channels = input_dim

        for i in range(num_blocks):
            dilation = 2 ** i
            layers.append(nn.Conv1d(in_channels, num_filters, kernel_size,
                                     padding=(kernel_size - 1) * dilation,
                                     dilation=dilation))
            layers.append(nn.BatchNorm1d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = num_filters

        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(num_filters, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.tcn(x)        # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        out = self.linear(x)   # [B, T, 1]
        return out.squeeze(-1) # [B, T]








'''class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding != 0 else out

class SimpleTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(SimpleTCNBlock, self).__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, T]
        x = self.conv(x)               # [B, C_out, T]
        x = x.transpose(1, 2)          # -> [B, T, C] for LayerNorm
        x = self.norm(x)
        x = x.transpose(1, 2)          # -> [B, C, T]
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SimpleTCN(nn.Module):
    def __init__(self, input_dim, num_blocks=4, num_filters=64, kernel_size=3, dropout=0.2):
        super(SimpleTCN, self).__init__()
        layers = []
        for i in range(num_blocks):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_filters
            layers.append(SimpleTCNBlock(in_ch, num_filters, kernel_size, dilation, dropout))

        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, T, C] -> [B, C, T]
        x = self.tcn(x)  # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        out = self.fc(x)  # [B, T, 1]
        out = out.squeeze(-1)  # [B, T]
        return out  # 整段序列的预测

    '''






'''
class SimpleTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2):
        super(SimpleTCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        #self.norm = nn.LayerNorm(out_channels)  # 使用 LayerNorm 替代 BatchNorm，更贴近 MATLAB
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        #x = x.transpose(1, 2)   # (B, C, T) → (B, T, C)
        x = self.norm(x)
        #x = x.transpose(1, 2)   # (B, T, C) → (B, C, T)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SimpleTCN(nn.Module):
    def __init__(self, input_dim, channels, kernel_size=3, dropout=0.2):
        super(SimpleTCN, self).__init__()
        layers = []
        for i in range(len(channels)):
            in_ch = input_dim if i == 0 else channels[i - 1]
            out_ch = channels[i]
            layers.append(SimpleTCNBlock(in_ch, out_ch, kernel_size, dropout))
        self.tcn = nn.Sequential(*layers)

        self.fc = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, T, C) → (B, C, T)
        x = self.tcn(x)         # (B, C, T)
        x = x[:, :, -1]         # 取最后一个时间步 (B, C)
        out = self.fc(x)        # 输出回归值
        return out.squeeze(1)
'''