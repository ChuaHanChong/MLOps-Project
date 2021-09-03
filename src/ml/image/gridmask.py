"""
Helper functions for image grid masking.

Reference:
https://arxiv.org/abs/2001.04086.
https://github.com/google/automl/blob/master/efficientdet/aug/gridmask.py
"""
import math

import tensorflow as tf
from tensorflow_addons import image as image_ops
from typeguard import typechecked


class GridMask:
    """GridMask class for grid masking augmentation."""

    @typechecked
    def __init__(
        self,
        prob: float = 0.6,
        ratio: float = 0.6,
        rotate: float = 10.0,
        gridmask_size_ratio: float = 0.5,
        fill: int = 1,
        interpolation: str = 'BILINEAR',
    ):
        """
        Init method.

        Parameters
        ----------
        prob : float, optional
            Probability of occurance, by default `0.6`.
        ratio : float, optional
            Grid mask ratio, by default `0.6`.
            If `0.5`, grid and spacing will be equal.
        rotate : float, optional
            Rotation of grid mesh, by default `10`.
        gridmask_size_ratio : float, optional
            Grid to image size ratio, by default `0.5`.
        fill : int, optional
            Fill value for grids, by default `1`.
        interpolation : str, optional
            Interpolation method for rotation, by default `'BILINEAR'`.
        """
        self.prob = prob
        self.ratio = ratio
        self.rotate = rotate
        self.gridmask_size_ratio = gridmask_size_ratio
        self.fill = fill
        self.interpolation = interpolation

    @tf.function
    def random_rotate(self, mask):
        """Randomly rotates mask on given range."""
        angle = self.rotate * tf.random.normal([], -1, 1)
        angle = math.pi * angle / 180
        return image_ops.rotate(mask, angle, interpolation=self.interpolation)

    @staticmethod
    def crop(mask, h, w):
        """Crop in middle of mask and image corners."""
        ww = hh = tf.shape(mask)[0]
        mask = mask[
            (hh - h) // 2 : (hh - h) // 2 + h,
            (ww - w) // 2 : (ww - w) // 2 + w,
        ]
        return mask

    @tf.function
    def mask(self, h, w):
        """Mask helper function for initializing grid mask of required size."""
        h = tf.cast(h, tf.float32)
        w = tf.cast(w, tf.float32)
        mask_w = mask_h = tf.cast(
            tf.cast((self.gridmask_size_ratio + 1), tf.float32) * tf.math.maximum(h, w),
            tf.int32,
        )
        self.mask_w = mask_w
        mask = tf.zeros(shape=[mask_h, mask_w], dtype=tf.int32)
        gridblock = tf.random.uniform(
            shape=[],
            minval=int(tf.math.minimum(h * 0.5, w * 0.3)),
            maxval=int(tf.math.maximum(h * 0.5, w * 0.3)) + 1,
            dtype=tf.int32,
        )

        if self.ratio == 1:
            length = tf.random.uniform(
                shape=[],
                minval=1,
                maxval=gridblock + 1,
                dtype=tf.int32,
            )
        else:
            length = tf.cast(
                tf.math.minimum(
                    tf.math.maximum(
                        int(tf.cast(gridblock, tf.float32) * self.ratio + 0.5),
                        1,
                    ),
                    gridblock - 1,
                ),
                tf.int32,
            )

        for _ in range(2):
            start_w = tf.random.uniform(
                shape=[],
                minval=0,
                maxval=gridblock + 1,
                dtype=tf.int32,
            )
            for i in range(mask_w // gridblock):
                start = gridblock * i + start_w
                end = tf.math.minimum(start + length, mask_w)
                indices = tf.reshape(tf.range(start, end), [end - start, 1])
                updates = (
                    tf.ones(shape=[end - start, mask_w], dtype=tf.int32) * self.fill
                )
                mask = tf.tensor_scatter_nd_update(mask, indices, updates)
            mask = tf.transpose(mask)

        return mask

    def __call__(self, image):
        """Mask input image tensor with random grid mask."""
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        grid = self.mask(h, w)
        grid = self.random_rotate(grid)
        mask = self.crop(grid, h, w)
        mask = tf.cast(mask, image.dtype)
        mask = tf.reshape(mask, (h, w))
        mask = tf.expand_dims(mask, -1) if image._rank() != mask._rank() else mask
        occur = tf.random.normal([], 0, 1) < self.prob
        image = tf.cond(occur, lambda: image * mask, lambda: image)
        return image


def gridmask(
    image,
    prob=0.5,
    ratio=0.6,
    rotate=10.0,
    gridmask_size_ratio=0.5,
    fill=1,
):
    """Callable instance of GridMask and transforms input image."""
    gridmask_obj = GridMask(
        prob=prob,
        ratio=ratio,
        rotate=rotate,
        gridmask_size_ratio=gridmask_size_ratio,
        fill=fill,
    )
    image = gridmask_obj(image)
    return image
