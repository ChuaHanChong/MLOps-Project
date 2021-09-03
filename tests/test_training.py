"""Test for training related function."""
import tensorflow as tf
from icecream import ic
from ml.training.lr_schedules import WarmupCosineDecayRestarts


class TestLR(tf.test.TestCase):
    """Learning rate testing."""

    def test_cosine_lr_scheduler_1(self):
        """Test restarts and warmup."""
        steps_per_epoch = 2
        epochs_per_cycle = 5

        num_epochs = 12
        warmup_epochs = 2
        max_lr = 0.1

        learning_rate = WarmupCosineDecayRestarts(
            max_lr,
            first_decay_steps=steps_per_epoch * epochs_per_cycle,
            t_mul=1.0,
            m_mul=1.0,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
        )

        lrs = []
        for i in range(num_epochs * steps_per_epoch):
            lrs.append(learning_rate(i).numpy())
        ic(lrs)

        correct_lrs = [
            0.0,
            0.025,
            0.05,
            0.075,
            0.1,
            0.09755283,
            0.090450846,
            0.07938927,
            0.06545085,
            0.049999997,
            0.034549143,
            0.020610739,
            0.00954915,
            0.002447176,
            0.1,
            0.09755283,
            0.090450846,
            0.07938927,
            0.065450855,
            0.049999997,
            0.034549143,
            0.02061073,
            0.009549156,
            0.002447176,
        ]

        for i in range(len(lrs)):
            self.assertAlmostEqual(lrs[i], correct_lrs[i])

    def test_cosine_lr_scheduler_2(self):
        """Test restarts, warmup and initial lr reducing."""
        steps_per_epoch = 2
        epochs_per_cycle = 5

        num_epochs = 12
        warmup_epochs = 2
        max_lr = 0.1

        learning_rate = WarmupCosineDecayRestarts(
            max_lr,
            first_decay_steps=steps_per_epoch * epochs_per_cycle,
            t_mul=1.0,
            m_mul=1.0,
            steps_per_epoch=steps_per_epoch,
            warmup_epochs=warmup_epochs,
        )

        lrs = []
        for i in range(num_epochs * steps_per_epoch):
            if i != 0 and (i / steps_per_epoch) % 5 == 0:
                # cut initial lr into half after every 5 epochs
                learning_rate.scheduler.initial_learning_rate *= 0.5
            lrs.append(learning_rate(i).numpy())
        ic(lrs)

        correct_lrs = [
            0.0,
            0.025,
            0.05,
            0.075,
            0.1,
            0.09755283,
            0.090450846,
            0.07938927,
            0.06545085,
            0.049999997,
            0.017274572,
            0.010305369,
            0.004774575,
            0.001223588,
            0.05,
            0.048776414,
            0.045225423,
            0.039694633,
            0.032725427,
            0.024999999,
            0.008637286,
            0.0051526823,
            0.002387289,
            0.000611794,
        ]

        for i in range(len(lrs)):
            self.assertAlmostEqual(lrs[i], correct_lrs[i])


if __name__ == '__main__':
    tf.test.main()
