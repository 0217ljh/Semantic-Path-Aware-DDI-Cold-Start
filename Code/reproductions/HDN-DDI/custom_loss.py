"""HDN-DDI: byte-exact port of jcsun-00/HDN-DDI/drugbank_test/custom_loss.py

Original source: https://github.com/jcsun-00/HDN-DDI/blob/main/drugbank_test/custom_loss.py
No semantic changes — only re-saved with this docstring header.
"""

from torch import nn
import torch.nn.functional as F


class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature

    def forward(self, p_scores, n_scores):
        if self.adv_temperature:
            weights = F.softmax(self.adv_temperature * n_scores, dim=-1).detach()
            n_scores = weights * n_scores
        p_loss = -F.logsigmoid(p_scores).mean()
        n_loss = -F.logsigmoid(-n_scores).mean()

        return (p_loss + n_loss) / 2, p_loss, n_loss
