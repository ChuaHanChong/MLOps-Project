"""Test for data related function."""
from io import BytesIO

import PIL.Image
import tensorflow as tf
from icecream import ic
from ml.image import PreprocessImage
from ml.image import read_image


class TestImagePreprocessing(tf.test.TestCase):
    """Image preprocessing testing."""

    def test_image_resizing(self):
        """Test image resizing with maintaining original aspect ratio."""
        # test equal height and width
        inputs_1 = tf.ones((600, 600, 3), dtype=tf.uint8)
        outputs_1 = PreprocessImage.resize_image(inputs_1, 224)
        ic(outputs_1.shape)
        self.assertEqual(outputs_1.shape, [224, 224, 3])

        # test larger height
        inputs_2 = tf.ones((600, 400, 3), dtype=tf.uint8)
        outputs_2 = PreprocessImage.resize_image(inputs_2, 224)
        ic(outputs_2.shape)
        self.assertEqual(outputs_2.shape, [224, 149, 3])

        # test larger width
        inputs_3 = tf.ones((400, 600, 3), dtype=tf.uint8)
        outputs_3 = PreprocessImage.resize_image(inputs_3, 224)
        ic(outputs_3.shape)
        self.assertEqual(outputs_3.shape, [149, 224, 3])

        # test return same output as input
        inputs_4 = tf.cast(
            tf.random.uniform(
                (600, 400, 3),
                minval=0,
                maxval=256,
                dtype=tf.int64,
            ),
            dtype=tf.uint8,
        )
        outputs_4 = PreprocessImage.resize_image(inputs_4, 600)
        ic(outputs_4.shape)
        self.assertAllEqual(inputs_4, outputs_4)

    def test_image_padding(self):
        """Test image padding on shorter side of image."""
        # test equal height and width
        inputs_1 = tf.ones((600, 600, 3), dtype=tf.uint8)
        outputs_1 = PreprocessImage.pad_image(inputs_1)
        ic(outputs_1.shape)
        self.assertEqual(outputs_1.shape, [600, 600, 3])

        # test larger height
        inputs_2 = tf.ones((600, 400, 3), dtype=tf.uint8)
        outputs_2 = PreprocessImage.pad_image(inputs_2)
        ic(outputs_2.shape)
        self.assertEqual(outputs_2.shape, [600, 600, 3])

        inputs_3 = tf.ones((600, 401, 3), dtype=tf.uint8)
        outputs_3 = PreprocessImage.pad_image(inputs_3)
        ic(outputs_3.shape)
        self.assertEqual(outputs_3.shape, [600, 600, 3])

    def test_image_loading_saving(self):
        """Test image loading and saving."""
        # load raw image
        path = 'tests/images/raw.jpeg'
        image1 = read_image(path)
        self.assertIsInstance(image1, tf.Tensor)
        PIL.Image.fromarray(image1.numpy()).save('tests/images/raw.png')

        # test binary format
        data_file = BytesIO()
        PIL.Image.fromarray(image1.numpy()).save(data_file, format='PNG')
        data_bytes = data_file.getvalue()

        image2 = tf.io.decode_jpeg(data_bytes)
        self.assertIsInstance(image2, tf.Tensor)
        self.assertAllEqual(image1, image2)

        # load saved image
        saved_path = 'tests/images/raw.png'
        saved_image = read_image(saved_path)
        self.assertAllEqual(image1, saved_image)

    def test_image_enhancement(self):
        """Test image sharpness and contrast improvement."""
        path = 'tests/images/raw.jpeg'
        image = read_image(path)

        # test sharpness improvement
        output_1 = PreprocessImage.improve_sharpness(image)
        self.assertIsNotNone(output_1)
        PIL.Image.fromarray(
            tf.cast(
                tf.clip_by_value(output_1, 0.0, 255.0),
                dtype=tf.uint8,
            ).numpy(),
        ).save('tests/images/sharpness-improved.png')

        # test contrast improvement
        output_2 = PreprocessImage.improve_contrast(output_1)
        self.assertIsNotNone(output_2)
        PIL.Image.fromarray(
            tf.cast(
                tf.clip_by_value(output_2, 0.0, 255.0),
                dtype=tf.uint8,
            ).numpy(),
        ).save('tests/images/sharpness-contrast-improved.png')

    def test_image_preprocessing(self):
        """Test full image preprocessing."""
        path = 'tests/images/raw.jpeg'
        image = read_image(path)

        # test full processed image
        output_1 = PreprocessImage(
            target_size=600,
            sharpness_improvement=True,
            contrast_improvement=True,
            image_padding=True,
        )(image)
        self.assertEqual(output_1.shape, [600, 600, 3])
        PIL.Image.fromarray(
            tf.cast(
                tf.clip_by_value(output_1, 0.0, 255.0),
                dtype=tf.uint8,
            ).numpy(),
        ).save('tests/images/full-processed.png')

        # test serialize tensor
        data_bytes = tf.io.serialize_tensor(output_1)
        output_2 = tf.io.parse_tensor(data_bytes, tf.float32)
        self.assertIsInstance(output_2, tf.Tensor)
        self.assertAllEqual(output_1, output_2)

        # test saved image
        saved_path = 'tests/images/full-processed.png'
        saved_output = read_image(saved_path)
        self.assertAllEqual(
            tf.cast(tf.clip_by_value(output_1, 0.0, 255.0), dtype=tf.uint8),
            saved_output,
        )
