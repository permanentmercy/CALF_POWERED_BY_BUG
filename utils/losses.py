# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class Log1pTweedieLoss(nn.Module):
    """
    Tweedie loss for long-tailed positive data, with built-in log1p transform.

    Expects:
        - y_pred: model output, should be in log1p space (i.e., model predicts ln(1 + y_original))
        - y_true: target, also in log1p space (i.e., already transformed by torch.log1p)
    Automatically inverts the log1p transform via expm1 before computing the loss,
    ensuring numerical stability and proper handling of zeros.

    Args:
        p: Tweedie variance power (1 < p < 2). Default 1.5.
        reduction: 'mean', 'sum', or 'none'.
    """
    def __init__(self, p=1.5, reduction='mean'):
        super().__init__()
        if not (1 < p < 2):
            raise ValueError(f"TweedieLoss requires 1 < p < 2, but got p={p}")
        self.p = p
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # 1. Inverse log1p: map back to original positive domain
        y_pred_orig = t.expm1(y_pred)   # expm1(x) = exp(x) - 1, the inverse of log1p
        y_true_orig = t.expm1(y_true)

        # 2. Ensure positivity (avoid numerical issues from tiny negative values)
        y_pred_orig = F.softplus(y_pred_orig)

        # 3. Tweedie deviance
        a = y_true_orig * t.pow(y_pred_orig, 1 - self.p) / (1 - self.p)
        b = t.pow(y_pred_orig, 2 - self.p) / (2 - self.p)
        loss = -a + b

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
