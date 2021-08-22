"""Module for training utils."""
from typing import Dict

from tensorflow.keras import losses as losses_mod
from tensorflow.keras import metrics as metrics_mod
from typeguard import typechecked


@typechecked
def call_loss_object(config: Dict[str, str]):
    """Call a tf.keras loss."""
    if config.get('kwargs'):
        return eval(f'{losses_mod}.' + config['name'])(**config['kwargs'])
    else:
        return eval(f'{losses_mod}.' + config['name'])()


@typechecked
def call_metric_object(config):
    """Call a tf.keras metric."""
    if config.get('kwargs'):
        return eval(f'{metrics_mod}.' + config['name'])(**config['kwargs'])
    else:
        return eval(f'{metrics_mod}.' + config['name'])()
