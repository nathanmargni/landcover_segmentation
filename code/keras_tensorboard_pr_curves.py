from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.pr_curve import metadata
from tensorflow.keras.metrics import Metric
from keras.utils import metrics_utils
from keras import backend
import tensorflow as tf
import numpy as np


class ClassifierCurve(Metric):
    def __init__(
        self,
        num_thresholds=200,
        thresholds=None,
        metric1="precision",
        metric2="recall",
        top_k=None,
        class_id=None,
        name="classifier_curve",
        **kwargs,
    ):
        self.num_thresholds = num_thresholds
        self.thresholds = thresholds
        self.metric1 = metric1
        self.metric2 = metric2
        self.top_k = top_k
        self.class_id = class_id

        super().__init__(name=name, **kwargs)
        shape = [self.num_thresholds]
        self.true_positives = self.add_weight(
            name="tp", shape=shape, initializer="zeros"
        )
        self.true_negatives = self.add_weight(
            name="tn", shape=shape, initializer="zeros"
        )
        self.false_positives = self.add_weight(
            name="fp", shape=shape, initializer="zeros"
        )
        self.false_negatives = self.add_weight(
            name="fn", shape=shape, initializer="zeros"
        )

        self._init_from_thresholds = thresholds is not None
        if thresholds is not None:
            # If specified, use the supplied thresholds.
            self.num_thresholds = len(thresholds) + 2
            thresholds = sorted(thresholds)
            self._thresholds_distributed_evenly = (
                metrics_utils.is_evenly_distributed_thresholds(
                    np.array([0.0] + thresholds + [1.0])
                )
            )
        else:
            if num_thresholds <= 1:
                raise ValueError(
                    "Argument `num_thresholds` must be an integer > 1. "
                    f"Received: num_thresholds={num_thresholds}"
                )

            # Otherwise, linearly interpolate (num_thresholds - 2) thresholds in
            # (0, 1).
            self.num_thresholds = num_thresholds
            thresholds = [
                (i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)
            ]
            self._thresholds_distributed_evenly = True

        # Add an endpoint "threshold" below zero and above one for either
        # threshold method to account for floating point imprecisions.
        self._thresholds = np.array(
            [0.0 - backend.epsilon()] + thresholds + [1.0 + backend.epsilon()]
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        updates = metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,
            },
            y_true,
            y_pred,
            thresholds=self._thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight,
        )

        return updates

    def reset_state(self):
        zeros = tf.zeros_like(self.true_positives)
        self.true_positives.assign(zeros)
        self.true_negatives.assign(zeros)
        self.false_positives.assign(zeros)
        self.false_negatives.assign(zeros)

    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "metric1": self.metric1,
            "metric2": self.metric2,
            "top_k": self.top_k,
            "class_id": self.class_id,
        }
        # optimization to avoid serializing a large number of generated thresholds
        if self._init_from_thresholds:
            # We remove the endpoint thresholds as an inverse of how the thresholds
            # were initialized. This ensures that a metric initialized from this
            # config has the same thresholds.
            config["thresholds"] = self._thresholds[1:-1]

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @staticmethod
    def calc_metric(metric_name, tp, fp, tn, fn):
        # this is not a complete set of metrics, just enough for PR/ROC curves
        metric_name = metric_name.lower()
        if metric_name in ("precision",):
            return tf.math.divide_no_nan(tp, tf.math.add(tp, fp))
        elif metric_name in ("recall", "tpr", "sensitivity"):
            return tf.math.divide_no_nan(tp, tf.math.add(tp, fn))
        elif metric_name in ("fpr",):
            return tf.math.divide_no_nan(fp, tf.math.add(fp, tn))
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    def result(self):
        metric1 = self.calc_metric(
            self.metric1,
            tp=self.true_positives,
            fp=self.false_positives,
            tn=self.true_negatives,
            fn=self.false_negatives,
        )
        metric2 = self.calc_metric(
            self.metric2,
            tp=self.true_positives,
            fp=self.false_positives,
            tn=self.true_negatives,
            fn=self.false_negatives,
        )

        # see definition here:
        # https://github.com/tensorflow/tensorboard/blob/806dd3e4dab88123efdeb1eece539efd4d0bbea0/tensorboard/plugins/pr_curve/summary.py#L557
        return tf.stack(
            [
                tf.cast(self.true_positives, tf.float32),
                tf.cast(self.false_positives, tf.float32),
                tf.cast(self.true_negatives, tf.float32),
                tf.cast(self.false_negatives, tf.float32),
                tf.cast(metric1, tf.float32),
                tf.cast(metric2, tf.float32),
            ]
        )


class PRCurve(ClassifierCurve):
    def __init__(
        self, metric1="precision", metric2="recall", name="pr_curve", **kwargs
    ):
        super().__init__(metric1=metric1, metric2=metric2, name=name, **kwargs)


class ROCCurve(ClassifierCurve):
    def __init__(self, metric1="tpr", metric2="fpr", name="roc_curve", **kwargs):
        super().__init__(metric1=metric1, metric2=metric2, name=name, **kwargs)


class TensorBoardPRCurves(TensorBoard):
    """
    Extensions to tf.keras.callbacks.TensorBoard to write summaries
    for the TensorBoard PR Curves plugin.
    """

    def __init__(self, *args, **kwargs):
        # whether or not to use the PR curve summary and what log suffix is used
        self.pr_curve_names = kwargs.pop("pr_curve_names", [])
        # optional AUC metric names to add to the description in tensorboard
        # (one for each entry in pr_curve_names if set)
        self.auc_names = kwargs.pop("auc_names", None)
        if not self.auc_names:
            self.auc_names = [None] * len(self.pr_curve_names)
        elif len(self.auc_names) != len(self.pr_curve_names):
            raise ValueError(
                "length of auc_names must match length of pr_curve_names if set"
            )
        super().__init__(*args, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        # separate the logs with PR curves to avoid errors
        # when the base class expects metrics to be scalars
        regular_logs, pr_logs = self.separate_pr_logs(logs)
        super().on_epoch_end(epoch, regular_logs)

        # run additional logic to write the epoch PR curves to tensorboard
        self._log_epoch_curves(epoch, pr_logs)

    def on_test_end(self, logs):
        # no apparent need to run anything for PR curves here
        # so just filter the logs out
        regular_logs, _ = self.separate_pr_logs(logs)
        super().on_test_end(regular_logs)

    def separate_pr_logs(self, logs):
        """
        Filter out any metrics in the logs that end with
        one of the strings in `self.pr_curve_names` and
        add them to a separate dict of curves to be processed separately.
        If `self.auc_names` was set, AUC metrics will be
        stored with the curve if found.
        """
        if self.pr_curve_names is None:
            return logs, []
        logs = logs.copy()  # take a shallow copy
        pr_logs = {}
        for suffix, auc_name in zip(self.pr_curve_names, self.auc_names):
            for k in list(logs.keys()):
                if k.endswith(suffix):
                    auc_log = None
                    if auc_name:
                        prefix = k[: -len(suffix)]
                        auc_key = prefix + auc_name
                        auc_log = logs.get(prefix + auc_name, None)

                    # create a prefix followed by a slash for grouping in tensorboard
                    # e.g. 'x_pr_curve' -> 'pr_curve/x_pr_curve'
                    # 'val_x_pr_curve' -> 'val_pr_curve/x_pr_curve'
                    if k.startswith("val_"):
                        new_key = "val_" + suffix + "/" + k[4:]
                    else:
                        new_key = suffix + "/" + k
                    pr_logs[new_key] = (logs[k], auc_name, auc_log)

                    # ok to delete while iterating because
                    # list() takes a copy of keys
                    del logs[k]

        return logs, pr_logs

    def _log_epoch_curves(self, epoch, logs):
        if not logs:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
        val_logs = {k: v for k, v in logs.items() if k.startswith("val_")}

        with tf.summary.record_if(True):
            if train_logs:
                with self._train_writer.as_default():
                    for name, value in train_logs.items():
                        # AUC metric is rendered as a string in the description
                        # if `auc_names` was set and a metric with a matching name was found
                        curve, auc_name, auc = value

                        # the summary metadata is required for the summary to be found
                        # by the PR Curve plugin in Tensorboard
                        summary_metadata = metadata.create_summary_metadata(
                            display_name="epoch_" + name,
                            description=f"{auc_name}={auc}" if auc else None,
                            num_thresholds=len(curve[0]),
                        )
                        tf.summary.write(
                            tag=name,
                            tensor=curve,
                            metadata=summary_metadata,
                            step=epoch,
                        )
            if val_logs:
                with self._val_writer.as_default():
                    for name, value in val_logs.items():
                        curve, auc_name, auc = value
                        name = name[4:]  # Remove 'val_' prefix.
                        summary_metadata = metadata.create_summary_metadata(
                            display_name="epoch_" + name,
                            description=f"{auc_name}={auc}" if auc else None,
                            num_thresholds=len(curve[0]),
                        )
                        tf.summary.write(
                            tag=name,
                            tensor=curve,
                            metadata=summary_metadata,
                            step=epoch,
                        )