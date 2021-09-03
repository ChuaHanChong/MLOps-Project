"""Module for model building."""
from typing import Optional

import tensorflow as tf
from ml.utils import IMAGE_KEY
from ml.utils import transformed_name
from typeguard import typechecked


keras_applications = {
    'efficientnet-b0': tf.keras.applications.EfficientNetB0,
    'efficientnet-b1': tf.keras.applications.EfficientNetB1,
    'efficientnet-b2': tf.keras.applications.EfficientNetB2,
    'efficientnet-b3': tf.keras.applications.EfficientNetB3,
    'efficientnet-b4': tf.keras.applications.EfficientNetB4,
    'efficientnet-b5': tf.keras.applications.EfficientNetB5,
    'efficientnet-b6': tf.keras.applications.EfficientNetB6,
    'efficientnet-b7': tf.keras.applications.EfficientNetB7,
}

keras_applications_preprocess_input = {
    'efficientnet-b0': tf.keras.applications.efficientnet.preprocess_input,
    'efficientnet-b1': tf.keras.applications.efficientnet.preprocess_input,
    'efficientnet-b2': tf.keras.applications.efficientnet.preprocess_input,
    'efficientnet-b3': tf.keras.applications.efficientnet.preprocess_input,
    'efficientnet-b4': tf.keras.applications.efficientnet.preprocess_input,
    'efficientnet-b5': tf.keras.applications.efficientnet.preprocess_input,
    'efficientnet-b6': tf.keras.applications.efficientnet.preprocess_input,
    'efficientnet-b7': tf.keras.applications.efficientnet.preprocess_input,
}

input_shapes = {
    'efficientnet-b0': 224,
    'efficientnet-b1': 240,
    'efficientnet-b2': 260,
    'efficientnet-b3': 300,
    'efficientnet-b4': 380,
    'efficientnet-b5': 456,
    'efficientnet-b6': 528,
    'efficientnet-b7': 600,
}

default_trainable_expr = {
    'efficientnet-b0': '(stem|block|top|head)',
    'efficientnet-b1': '(stem|block|top|head)',
    'efficientnet-b2': '(stem|block|top|head)',
    'efficientnet-b3': '(stem|block|top|head)',
    'efficientnet-b4': '(stem|block|top|head)',
    'efficientnet-b5': '(stem|block|top|head)',
    'efficientnet-b6': '(stem|block|top|head)',
    'efficientnet-b7': '(stem|block|top|head)',
}


@typechecked
def build_model(
    model_name: str,
    num_classes: int,
    pretrained_weight: Optional[str] = None,
    trainable: bool = False,
    output_activation: Optional[str] = None,
    name: Optional[str] = None,
):
    """
    Build model with TF keras applications.

    Parameters
    ----------
    image_size : int
        Image size.
    model_name : str
        Prebuild TF hub model to be called.
    num_classes : int
        Number of output classes.
    pretrained_weight : str, optional
        Path of pretrained .h5 keras model weight, by default `None`.
    trainable : bool, optional
        A controlling whether the TF hub model is trainable, by default `False`.
    name : str, optional
        Name of the model, by default `None`.

    Returns
    -------
    tf.keras.Model
        A customized keras model with modified head.
    """
    image_size = input_shapes[model_name]
    inputs = tf.keras.layers.Input(
        shape=(image_size, image_size, 3),
        name=transformed_name(IMAGE_KEY),
    )
    inputs = keras_applications_preprocess_input[model_name](inputs)
    base_model = keras_applications[model_name](
        include_top=False,
        input_tensor=inputs,
        weights=pretrained_weight or 'imagenet',
    )
    base_model.trainable = trainable

    x = tf.keras.layers.GlobalAveragePooling2D(name='top_avg_pool')(
        base_model.output,
    )
    x = tf.keras.layers.Dense(
        1000,
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='head_dense',
    )(x)
    x = tf.keras.layers.BatchNormalization(name='head_bn')(x)
    x = tf.keras.layers.ReLU(name='head_relu')(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation=output_activation,
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name='head_output',
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model
