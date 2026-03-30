"""
Federated module for CTN-LT
"""

from .fed_trainer import (
    FederatedClient,
    FederatedServer,
    FederatedTrainer
)

__all__ = [
    'FederatedClient',
    'FederatedServer',
    'FederatedTrainer'
]
