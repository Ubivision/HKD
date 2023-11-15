from __future__ import print_function

import torch.nn as nn


class HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss



class New_HintLoss(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(New_HintLoss, self).__init__()

    def forward(self, f_s, f_t):
        loss = (f_s - f_t).pow(2).view(f_s.size(0), -1).mean(1)
        return loss
