"""
Federated CTN-LT Package
联邦学习版本的CTN-LT (Contrastive Transformer Network for Long Tail Classification)
"""

__version__ = '0.1.0'
__author__ = 'Your Name'
__email__ = 'your.email@example.com'

from . import models
from . import federated
from . import data
from . import evaluation

__all__ = ['models', 'federated', 'data', 'evaluation']
