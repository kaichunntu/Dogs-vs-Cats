
from torch import nn

class Category_Loss(nn.Module):
    def __init__(self):
        super(Category_Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        return loss
