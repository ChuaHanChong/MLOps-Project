"""Module for training utils."""
from typing import Any
from typing import Dict

import tensorflow as tf
from typeguard import typechecked


@typechecked
def call_loss_object(config: Dict[str, Any]) -> tf.keras.losses:
    """Call a tf.keras loss."""
    if config.get('kwargs'):
        return eval('tf.keras.losses.' + config['name'])(**config['kwargs'])
    else:
        return eval('tf.keras.losses.' + config['name'])()


@typechecked
def call_metric_object(config: Dict[str, Any]) -> tf.keras.losses:
    """Call a tf.keras metric."""
    if config.get('kwargs'):
        return eval('tf.keras.metrics.' + config['name'])(**config['kwargs'])
    else:
        return eval('tf.keras.metrics.' + config['name'])()
