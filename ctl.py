from typing import List, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from criterions import HingeLoss


class FeedForwardNetwork(nn.Module):
    """Feed-forward network for SBC

    Reference.
    - Complete the Look: Scene-based Complementary Product Recommendation (2019)
    - Linear -> BN -> ReLU -> Dropout -> Linear -> L2-Norm
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,
        hidden_dim: int = None,
        activation: str = "ReLU",
        dropout: float = 0.5,
    ):
        super(FeedForwardNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.__dict__[activation]()
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_in(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_out(x)
        x = F.normalize(x)
        return x


class CTL(nn.Module):
    """Implementation for 'Complete the Look: Scene-based Complementary Product Recommendation (2019)'
    TODO
      - generalize model backbone intialization
      - weight initialization
    """

    def __init__(
        self,
        categories: List[str],
        backbone: str = "resnet50",
        feature_dim: int = 128,
        pretrained: bool = True,
        freeze: bool = True,
        margin: float = 0.2,
    ):
        super(CTL, self).__init__()
        # image encoder
        self.enc = models.__dict__[backbone](pretrained=pretrained)
        self.enc.fc = nn.Identity()

        if freeze:
            for p in self.enc.parameters():
                p.requires_grad = False

        # essential feed-forward networks
        self.linear_global = FeedForwardNetwork(2048, feature_dim)
        self.linear_region = FeedForwardNetwork(1024, feature_dim)
        self.linear_region_hat = FeedForwardNetwork(1024, feature_dim)

        # look-up table for categories
        self.cats = sorted(categories)
        self.cat2id = {cat: i for i, cat in enumerate(self.cats)}
        self.num_cats = len(self.cats)
        self.cat_emb = nn.Embedding(
            self.num_cats + 1, feature_dim
        )  # num_embeddings = {len(cat) + 1} (to treat UNK cat)
        self.feature_dim = feature_dim

        if margin is not None:
            self.criterion = HingeLoss(margin=margin)

    def forward(
        self,
        s: torch.Tensor, # scene
        p_pos: torch.Tensor, # positive
        p_neg: torch.Tensor, # negative
        c: List[str], # category
        return_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_pos = self._calculate_compatibility(s, p_pos, c)  # scene vs. positive
        d_neg = self._calculate_compatibility(s, p_neg, c)  # scene vs. negative

        if return_loss:
            loss = self.criterion(d_pos, d_neg)
            return d_pos, d_neg, loss
        return d_pos, d_neg

    def _calculate_compatibility(
        self, s: torch.Tensor, p: torch.Tensor, c: List[str]
    ) -> torch.Tensor:
        """
        Args:
            s (torch.Tensor): scene image [B, C, W, H]
            p (torch.Tensor): product image [B, C, W, H]
            c (List[str]): a list of category
        """
        # d_global(s, p): overall scene <-> product
        f_s = self._extract_global(s)  # scene [B, H]
        f_p = self._extract_global(p)  # product [B, H]
        d_global = F.pairwise_distance(f_s, f_p, p=2)

        # d_local(s, p): scene regions <-> product
        f_i = self._extract_region(s, self.linear_region) # [B, R, H]
        f_i_hat = self._extract_region(s, self.linear_region_hat)
        e_c_hat = self._lookup_cat(c)
        a_hat = -torch.cdist(f_i_hat, e_c_hat).squeeze(-1)
        a = F.softmax(a_hat, dim=-1)  # attention weights
        d_local = (a * torch.cdist(f_i, f_p.unsqueeze(1)).squeeze(-1)).sum(dim=1)

        d = (d_global + d_local) / 2
        return d

    def _lookup_cat(self, cats: Union[List[str], List[List[str]]]) -> torch.Tensor:
        if isinstance(cats[0], str): # List[category(str)]
            cats = torch.tensor([self.cat2id.get(c, self.num_cats) for c in cats])
            e_c_hat = self.cat_emb(cats.to(self.device))  # [B, H]
            e_c_hat = F.normalize(e_c_hat)
            e_c_hat = e_c_hat.unsqueeze(1)
            return e_c_hat

        elif isinstance(cats[0], list): # List[List[category(str)]]
            e_c_hats = []
            for per_cats in cats:
                per_cats = torch.tensor(
                    [self.cat2id.get(c, self.num_cats) for c in per_cats]
                )
                e_c_hat = self.cat_emb(per_cats.to(self.device))  # [B, H]
                e_c_hat = F.normalize(e_c_hat)
                e_c_hats.append(e_c_hat)
            e_c_hats = torch.stack(e_c_hats)
            return e_c_hats

        else:
            raise NotImplementedError
        
    def _extract_global(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        x = self.linear_global(x)
        return x

    def _extract_region(
        self, x: torch.Tensor, feedforwardnet: nn.Module
    ) -> torch.Tensor:
        x = self.enc.conv1(x)  # [B, 64, 112, 112]
        x = self.enc.bn1(x)  # [B, 64, 112, 112]
        x = self.enc.relu(x)  # [B, 64, 112, 112]
        x = self.enc.maxpool(x)  # [B, 64, 56, 56]
        x = self.enc.layer1(x)  # [B, 256, 56, 56]
        x = self.enc.layer2(x)  # [B, 512, 28, 28]
        x = self.enc.layer3(x)  # [B, 1024, 14, 14]

        b, c, h, w = x.shape  # R(num of regions) = h x w
        x = x.permute(0, 2, 3, 1)  # [B, h, w, 1024]
        x = x.reshape(b * h * w, c)  # [B*R, 1024]
        x = feedforwardnet(x)  # [B*R, H]

        b_r = x.size(0)  # [B*R, H]
        r = b_r // b  # R
        x = x.reshape(b, r, self.feature_dim)  # [B, R, H]
        return x

    @property
    def device(self):
        d = next(self.parameters()).get_device()
        if d == -1:
            d = torch.device("cpu")
        return d
