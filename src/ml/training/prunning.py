"""Module for model pruning."""
import tensorflow_model_optimization as tfmot


def apply_pruning_to_layer(
    layer,
    prunable_layer_types=['Conv2D', 'Dense', 'DepthwiseConv2D'],
):
    """Apply pruning to trainable Keras layers.

    Parameters
    ----------
    layer : tf.keras.layers
        TF Keras layers.
    prunable_layer_types : list, optional
        List of prunable Keras layers name,
        by default `['Conv2D', 'Dense', 'DepthwiseConv2D']`.

    Returns
    -------
    tfmot.keras.pruning_wrapper.PruneLowMagnitude
        Wrapped prunable Keras layer.
    """
    if layer.trainable and type(layer).__name__ in prunable_layer_types:
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    else:
        return layer
