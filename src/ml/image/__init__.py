"""Module for image processing."""
from typing import Optional

import tensorflow as tf
import tensorflow_addons as tfa
import tf_clahe
from ml.image import autoaugment
from typeguard import typechecked


def read_image(path, channels=0):
    """Load image with TF."""
    return tf.io.decode_jpeg(tf.io.read_file(path), channels=channels)


class AugmentImage:
    """Image augmentation."""

    @typechecked
    def __init__(
        self,
        augname: str = 'randaug',
        ra_num_layers: int = 2,
        ra_magnitude: int = 15,
        image_dtype: tf.DType = tf.float32,
    ):
        """Initialize.

        Parameters
        ----------
        augname : str, optional
            Augmentation method with name like `'autoaug'`, `'randaug'`, `'ra_aa'`,
            by default `'randaug'`.
        ra_num_layers : int, optional
            The number of random augmentation transformations to apply
            sequentially to an image, by default `2`.
        ra_magnitude : int, optional
            Shared magnitude across all random augmentation operations,
            by default `15`.
        image_dtype : tf.DType, optional
            Output image data type, by default `tf.float32`.
        """
        self.augname = augname
        self.augparams = dict(
            ra_num_layers=ra_num_layers,
            ra_magnitude=ra_magnitude,
        )
        self.image_dtype = image_dtype

    def __call__(self, images):
        """Augment image.

        Parameters
        ----------
        image : tf.Tensor
            Input images.

        Returns
        -------
        tf.Tensor
            Augmented images.
        """
        images = tf.image.random_flip_left_right(images)
        images = tf.image.random_flip_up_down(images)

        images = tf.clip_by_value(images, 0.0, 255.0)
        images = tf.cast(images, dtype=tf.uint8)
        images = tf.map_fn(
            lambda x: autoaugment.distort_image(
                x,
                self.augname,
                **self.augparams,
            ),
            images,
            dtype=tf.uint8,
        )  # TODO modify to keras preprocessing layer???

        images = tf.cast(images, dtype=self.image_dtype)
        return images


class PreprocessImage:
    """Image preprocessing."""

    @typechecked
    def __init__(
        self,
        target_size: Optional[int] = None,
        sharpness_improvement: bool = True,
        contrast_improvement: bool = True,
        image_padding: bool = True,
        image_dtype: tf.DType = tf.float32,
    ):
        """Initialize.

        Parameters
        ----------
        target_size : int, optional
            Output image size, by default `None`.
        sharpness_improvement : bool, optional
            Improve sharpness of image with unsharp masking,
            by default `True`.
        contrast_improvement : bool, optional
            Improve contrast of image with histogram equalization,
            by default `True`.
        image_padding : bool, optional
            Pad the shorter side of image with zeros,
            by default `True`.
        image_dtype : tf.DType, optional
            Output image data type, by default `tf.float32`.
        """
        self.target_size = target_size
        self.sharpness_improvement = sharpness_improvement
        self.contrast_improvement = contrast_improvement
        self.image_padding = image_padding
        self.image_dtype = image_dtype

    @staticmethod
    @tf.function
    def resize_image(image, target_size):
        """Resize image while maintaining original aspect ratio."""
        shape = tf.shape(image)[:2]
        h = tf.cast(shape[0], tf.float32)
        w = tf.cast(shape[1], tf.float32)

        scale = target_size / tf.math.maximum(h, w)
        size = (tf.math.round(scale * h), tf.math.round(scale * w))
        return tf.image.resize(image, size, method='nearest')

    @staticmethod
    @tf.function
    def improve_sharpness(image):
        """Improve sharpness with unsharp masking."""
        blurred_image = tfa.image.gaussian_filter2d(image, (9, 9), 10)
        return tfa.image.blend(image, blurred_image, factor=-0.5)

    @staticmethod
    @tf.function
    def improve_contrast(image):
        """Improve contrast with (CLAHE).

        CLAHE stands for contrast limited adaptive histogram equalization.
        """
        image = tf.image.rgb_to_hsv(image)
        h, s, v = tf.split(image, 3, axis=-1)
        # TODO call from TF addons once this pull request is completed,
        # https://github.com/tensorflow/addons/pull/2362
        v = tf_clahe.clahe(v)
        return tf.image.hsv_to_rgb(tf.concat([h, s, v], axis=-1))

    @staticmethod
    @tf.function
    def pad_image(image):
        """Pad image with zero pixel values into square size image."""
        shape = tf.shape(image)[:2]
        h = shape[0]
        w = shape[1]

        def _pad(image):
            diff_h = tf.math.maximum(w - h, 0)
            pad_h = tf.cast(tf.math.rint(diff_h / 2), tf.int32)

            diff_w = tf.math.maximum(h - w, 0)
            pad_w = tf.cast(tf.math.rint(diff_w / 2), tf.int32)

            paddings = (
                (pad_h, diff_h - pad_h),
                (pad_w, diff_w - pad_w),
                (0, 0),
            )
            return tf.pad(image, paddings)

        return tf.cond(tf.equal(h, w), lambda: image, lambda: _pad(image))

    def __call__(self, image):
        """Preprocess image.

        Parameters
        ----------
        image : tf.Tensor
            Input image.

        Returns
        -------
        tf.Tensor
            Preprocessed image.
        """
        if self.target_size:
            image = self.resize_image(image, self.target_size)

        if self.sharpness_improvement:
            image = self.improve_sharpness(image)

        if self.contrast_improvement:
            image = self.improve_contrast(image)

        if self.image_padding:
            image = self.pad_image(image)

        image.set_shape([self.target_size, self.target_size, 3])
        image = tf.cast(image, dtype=self.image_dtype)
        return image
