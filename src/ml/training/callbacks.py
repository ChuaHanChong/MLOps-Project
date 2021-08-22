"""Module for training callbacks."""
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import backend as K
from typeguard import typechecked


class LearningRateLogger(tf.keras.callbacks.Callback):
    """A learning rate logger callback."""

    @typechecked
    def __init__(self, steps_per_epoch: Optional[int] = None):
        """Initialize.

        Parameters
        ----------
        steps_per_epoch : int, optional
            Number of steps per epoch, by default None.
        """
        super().__init__()
        if steps_per_epoch is None:
            raise ValueError('Arg `steps_per_epoch` is None.')
        self.steps_per_epoch = steps_per_epoch
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        """Log learning rate at the end of epoch."""
        logs = logs or {}
        if isinstance(
            self.model.optimizer.lr,
            tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            lr = K.get_value(
                self.model.optimizer.lr(epoch * self.steps_per_epoch),
            )
        else:
            lr = K.get_value(self.model.optimizer.lr)
        logs['learning_rate'] = lr


class EarlyStopping(tf.keras.callbacks.EarlyStopping):
    """Customized EarlyStopping.

    It stops training when a monitored metric has stopped improving or
    has reached a target.
    """

    @typechecked
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 0,
        verbose: int = 0,
        mode: str = 'auto',
        baseline: Optional[float] = None,
        target: Optional[float] = None,
        restore_best_weights: bool = False,
    ):
        """Initialize.

        Parameters
        ----------
        monitor : str, optional
            Quantity to be monitored, by default 'val_loss'.
        min_delta : int, optional
            Minimum change in the monitored quantity to qualify
            as an improvement, i.e. an absolute change of less
            than min_delta, will count as no improvement,
            by default 0.
        patience : int, optional
            Number of epochs with no improvement after which
            training will be stopped, by default 0.
        verbose : int, optional
            0: quiet, 1: update messages, by default 0.
        mode : str, optional
            One of {'auto', 'min', 'max'}, by default 'auto'.
            In 'min' mode, the learning rate will be reduced when the
            quantity monitored has stopped decreasing;
            in 'max' mode it will be reduced when the quantity monitored has
            stopped increasing;
            in 'auto' mode, the direction is automatically inferred from the
            name of the monitored quantity.
        baseline : int, optional
            Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline, by default None.
        target : int, optional
            Target value for the monitored quantity.
            Training will stop if the model reach the target value,
            by default None.
        restore_best_weights : bool, optional
            Whether to restore model weights from the epoch with the best
            value of the monitored quantity, by default False.
        """
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            baseline=baseline,
            restore_best_weights=restore_best_weights,
        )
        self.target = target

    def on_train_begin(self, logs=None):
        """Set up monitor metric at begining of training."""
        super().on_train_begin(logs=logs)
        if self.target is None:
            self.target = -np.Inf if self.monitor_op == np.less else np.Inf

    def on_epoch_end(self, epoch, logs=None):
        """Monitor the metric at the end of epoch."""
        current = self.get_monitor_value(logs)
        if current is None:
            return

        super().on_epoch_end(epoch, logs=logs)

        # terminate training when monitored metric has reached the target
        if self.monitor_op(current, self.target):
            self.model.stop_training = True


class ReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    """Customized ReduceLROnPlateau.

    It reduces learning rate when a metric has stopped improving.
    The supported lr schedulers are:
    - WarmupCosineDecayRestarts
    """

    @typechecked
    def __init__(
        self,
        monitor: str = 'val_loss',
        factor: float = 0.1,
        patience: int = 10,
        verbose: int = 0,
        mode: str = 'auto',
        min_delta: float = 0.0001,
        cooldown: int = 0,
        min_lr: float = 0.0,
        steps_per_epoch: Optional[int] = None,
        **kwargs,
    ):
        """Initialize.

        Parameters
        ----------
        monitor : str, optional
            Quantity to be monitored, by default 'val_loss'.
        factor : float, optional
            Factor by which the learning rate will be reduced, by default 0.1.
        patience : int, optional
            number of epochs with no improvement after which learning rate
            will be reduced, by default 10.
        verbose : int, optional
            0: quiet, 1: update messages, by default 0.
        mode : str, optional
            One of {'auto', 'min', 'max'}, by default 'auto'.
            In 'min' mode, the learning rate will be reduced when the
            quantity monitored has stopped decreasing;
            in 'max' mode it will be reduced when the quantity monitored has
            stopped increasing;
            in 'auto' mode, the direction is automatically inferred from the
            name of the monitored quantity.
        min_delta : float, optional
            Threshold for measuring the new optimum, to only focus on
            significant changes, by default 0.0001.
        cooldown : int, optional
            Number of epochs to wait before resuming normal operation after
            lr has been reduced, by default 0.
        min_lr : int, optional
            Lower bound on the learning rate, by default 0.
        steps_per_epoch : int, optional
            Number of steps per epoch, by default None.
        """
        super().__init__(
            monitor=monitor,
            factor=factor,
            patience=patience,
            verbose=verbose,
            mode=mode,
            min_delta=min_delta,
            cooldown=cooldown,
            min_lr=min_lr,
            **kwargs,
        )
        if steps_per_epoch is None:
            raise ValueError('Arg `steps_per_epoch` is None.')
        self.steps_per_epoch = steps_per_epoch

    def on_epoch_end(self, epoch, logs=None):
        """Monitor the metric at the end of epoch."""
        logs = logs or {}
        if isinstance(
            self.model.optimizer.lr,
            tf.keras.optimizers.schedules.LearningRateSchedule,
        ):
            logs['lr'] = K.get_value(
                self.model.optimizer.lr(epoch * self.steps_per_epoch),
            )
        else:
            logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is not None:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    if (
                        self.model.optimizer.lr.__class__.__name__
                        == 'WarmupCosineDecayRestarts'
                    ):
                        old_lr = float(
                            K.get_value(
                                self.model.optimizer.lr.scheduler.initial_learning_rate,
                            ),
                        )
                    else:
                        old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        if (
                            self.model.optimizer.lr.__class__.__name__
                            == 'WarmupCosineDecayRestarts'
                        ):
                            self.model.optimizer.lr.scheduler.initial_learning_rate = (
                                new_lr
                            )
                        else:
                            K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print(
                                '\nEpoch %05d: ReduceLROnPlateau reducing '
                                'learning rate to %s.' % (epoch + 1, new_lr),
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0


class AverageModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """Customized AverageModelCheckpoint.

    It saves average model weights and,
    optionally, assigns the averaged weights.
    """

    @typechecked
    def __init__(
        self,
        update_weights: bool,
        filepath: str,
        monitor: str = 'val_loss',
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = 'auto',
        save_freq: str = 'epoch',
        **kwargs,
    ):
        """Initialize.

        Parameters
        ----------
        update_weights : bool
            If `True`, assign the moving average weights to the model, and save
            them. If False, keep the old non-averaged weights, but the saved
            model uses the average weights.
        filepath : str
            Path to save the model file.
        monitor : str, optional
            The metric name to monitor, by default 'val_loss'.
        verbose : int, optional
            0: quiet, 1: update messages, by default 0.
        save_best_only : bool, optional
             Save when the model is considered the "best" and the latest best
             model according to the quantity monitored will not be overwritten,
             by default False.
        save_weights_only : bool, optional
            If True, then only the model's weights will be saved,
            else the full model is saved, by default False.
        mode : str, optional
            One of {'auto', 'min', 'max'}, by default 'auto'.
        save_freq : str, optional
            Checkpoint saving frequency, by default 'epoch'.
        """
        self.update_weights = update_weights
        super().__init__(
            filepath,
            monitor,
            verbose,
            save_best_only,
            save_weights_only,
            mode,
            save_freq,
            **kwargs,
        )

    def _get_optimizer(self):
        optimizer = self.model.optimizer
        if type(optimizer).__name__ in [
            'LossScaleOptimizer',
            'LossScaleOptimizerV1',
        ]:
            optimizer = optimizer._optimizer
        return optimizer

    def set_model(self, model):
        """Set Keras model and writes graph if specified."""
        super().set_model(model)
        optimizer = self._get_optimizer()
        if not isinstance(
            optimizer,
            tfa.optimizers.average_wrapper.AveragedOptimizerWrapper,
        ):
            raise TypeError(
                'AverageModelCheckpoint is only used when training'
                'with MovingAverage or StochasticAverage',
            )

    def _save_model(self, epoch, logs):
        optimizer = self._get_optimizer()
        assert isinstance(
            optimizer,
            tfa.optimizers.average_wrapper.AveragedOptimizerWrapper,
        )

        if self.update_weights:
            optimizer.assign_average_vars(self.model.trainable_variables)
            return super()._save_model(epoch, logs)
        else:
            non_avg_weights = self.model.get_weights()
            optimizer.assign_average_vars(self.model.trainable_variables)
            result = super()._save_model(epoch, logs)
            self.model.set_weights(non_avg_weights)
            return result
