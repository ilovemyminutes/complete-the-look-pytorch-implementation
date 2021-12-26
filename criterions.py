import torch
from torch import nn

class HingeLoss(nn.Module):
    """Hinge loss or triplet loss for Complete the Look Framework.
        $L = \sum_{s, p^{+}, p^{-} \in \Tau} max(d_{*}(s, p^{+}) - d_{*}(s, p^{-}) + \alpha, 0)$
    """
    def __init__(self, margin: float = 0.2, reduction: str = 'sum'):
        super(HingeLoss, self).__init__()
        self.m = margin
        self.reduction = reduction
        
    def forward(self, d_pos: torch.Tensor, d_neg: torch.Tensor) -> torch.Tensor:
        loss = torch.clamp(d_pos - d_neg + self.m, min=0.0)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise NotImplementedError

        return loss