import logging

import torch as t
import torch.nn as nn


@t.no_grad()
def weights_init(m: nn.Module):
    logger = logging.getLogger("Weights initializer")
    classname = m.__class__.__name__
    logger.debug(f"Initializing weights for {classname}")
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(10, 0.05)
        m.bias.data.zero_()
    else:
        logger.debug(f"Did not alter the weights for {classname}")
