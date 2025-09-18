import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class LipNet(nn.Module):
    def __init__(self, opt, vocab_size):
        super().__init__()
        self.opt = opt
        
        self.conv_layer = nn.Sequential(
            
            nn.Conv3d(in_channels= 3, out_channels= 32, kernel_size= (3, 5, 5), stride= (1, 2, 2), padding= (1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size= (1, 2, 2), stride= (1, 2, 2)),
            nn.Dropout3d(opt.dropout),
            
            nn.Conv3d(in_channels= 32, out_channels= 64, kernel_size= (3, 5, 5), stride= (1, 2, 2), padding= (1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size= (1, 2, 2), stride= (1, 2, 2)),
            nn.Dropout3d(opt.dropout),
            
            nn.Conv3d(in_channels= 64, out_channels= 96, kernel_size= (3, 5, 5), stride= (1, 2, 2), padding= (1, 2, 2)),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size= (1, 2, 2), stride= (1, 2, 2)),
            nn.Dropout3d(opt.dropout),
        )
        
        self.gru1 = nn.GRU(input_size= 96 * 3 * 6, opt.rnn_size, bidirectional=True)
        self.drp1 = nn.Dropout(opt.dropout)
        
        self.gru2 = nn.GRU(opt.rnn_size * 2, opt.rnn_size, bidirectional=True)
        self.drp2 = nn.Dropout(opt.dropout)
        
        self.linear = nn.Linear(in_features= opt.rnn_size * 2, out_features= vocab_size + 1)