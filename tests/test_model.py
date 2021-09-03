"""Test for model related function."""
import re

import tensorflow as tf
from icecream import ic
from ml.model import build_model
from ml.model import default_trainable_expr
from tensorflow.python.keras.utils.layer_utils import count_params


class TestModelWeight(tf.test.TestCase):
    """Model testing."""

    def test_default_trainable_weight(self):
        """Test default trainable weight when model finetuning."""
        num_classes = 1
        models_to_test = ['efficientnet-b0']

        for model_name in models_to_test:
            var_trainable_expr = default_trainable_expr[model_name]
            ic(model_name)

            # with re trainable expression
            model1 = build_model(model_name, num_classes)
            for layer in model1.layers:
                if re.match(var_trainable_expr, layer.name):
                    if not isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = True

            total_params_model1 = count_params(model1.weights)
            trainable_params_model1 = count_params(model1.trainable_weights)
            xtrainable_params_model1 = count_params(model1.non_trainable_weights)
            ic(total_params_model1)
            ic(trainable_params_model1)
            ic(xtrainable_params_model1)

            # without re
            model2 = build_model(model_name, num_classes)
            for layer in model2.layers:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True

            total_params_model2 = count_params(model2.weights)
            trainable_params_model2 = count_params(model2.trainable_weights)
            xtrainable_params_model2 = count_params(model2.non_trainable_weights)

            ic(total_params_model2)
            ic(trainable_params_model2)
            ic(xtrainable_params_model2)

            self.assertEqual(total_params_model1, total_params_model2)
            self.assertEqual(trainable_params_model1, trainable_params_model2)
            self.assertEqual(xtrainable_params_model1, xtrainable_params_model2)

            del model1
            del model2


if __name__ == '__main__':
    tf.test.main()
