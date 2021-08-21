"""Module for image processing."""
from typing import Any
from typing import Dict

import tensorflow as tf
from ml.image import PreprocessImage
from ml.utils import IMAGE_KEY
from ml.utils import LABEL_KEY
from ml.utils import transformed_name


# TFX Transform will call this function.
def preprocessing_fn(
    inputs: Dict[str, Any],
    custom_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    tf.transform's callback function for preprocessing inputs.

    Parameters
    ----------
    inputs : Dict[str, Any]
        Map from feature keys to raw not-yet-transformed features.
    custom_config : Dict[str, Any]
        Custom configurations.

    Returns
    -------
    Dict[str, Any]
        Map from string feature key to transformed feature operations.
    """
    outputs = {}

    # tf.io.decode_png function cannot be applied on a batch of data.
    # We have to use tf.map_fn
    image_features = tf.map_fn(
        lambda x: PreprocessImage(target_size=custom_config['target_size'])(
            tf.io.decode_png(x[0], channels=3),
        ),
        inputs[IMAGE_KEY],
        dtype=tf.float32,
    )
    outputs[transformed_name(IMAGE_KEY)] = image_features
    outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]

    return outputs
