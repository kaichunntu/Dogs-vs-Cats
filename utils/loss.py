
import numpy as np
import torch
from torch import nn

class Category_Loss(nn.Module):
    def __init__(self, label_weight):
        super(Category_Loss, self).__init__()
        if isinstance(label_weight,list):
            label_weight = torch.tensor(label_weight)
        elif isinstance(label_weight,np.ndarray):
            label_weight = torch.from_numpy(label_weight)
        self.criterion = nn.CrossEntropyLoss(weight=label_weight.to(torch.float32), 
                                             reduction="mean")

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        return loss
