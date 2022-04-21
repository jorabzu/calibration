from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import typing
import warnings
import feyn
from typing import Dict, Optional
from matplotlib.axes import Axes
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


from betacal import BetaCalibration


##############################################################

## Extra functions
def binarize_labels(y_true: typing.Iterable, positive_class: typing.Union[str, int, float]) -> typing.Iterable:
    """
    Arguments:
         y_true -- true labels for the observations.
        positive_class -- class to be considered as positive.

    Returns:
        Binarized labels. Those observations considered as positive are relabelled as 1, and the
        rest as 0.
    """

    if positive_class not in y_true:
        raise ValueError("The given positive class is not in y_true.")

    y_true_binarized = np.where(y_true == positive_class, 1, 0).ravel()

    return y_true_binarized


###############################################################


class Calibrator(ABC):
    @classmethod
    @abstractmethod
    def train(
        cls,
        y_pred_scores_uncalibrated: typing.Iterable,
        y_true: typing.Iterable,
        positive_class: typing.Union[str, int, float],
        sample_weight: typing.Optional[typing.Iterable] = None,
    ):
        pass

    @abstractmethod
    def calibrate(self, y_pred_scores_uncalibrated: typing.Iterable) -> typing.Iterable:
        pass


## Calibrator classes


class LogisticCalibrator(Calibrator):
    @classmethod
    def train(
        cls,
        y_pred_scores_uncalibrated: typing.Iterable,
        y_true: typing.Iterable,
        positive_class: typing.Union[str, int, float],
        sample_weight: typing.Optional[typing.Iterable] = None,
    ):
        """
        Trains a logistic regression for calibration.

        Arguments:

            y_pred_scores_uncalibrated -- uncalibrated scores predicted by a model.
            y_true -- true labels for the observations. It is assumed that there are two possible true labels.
            positive_class -- class to be considered as positive.
            sample_weight -- weight of the samples

        Returns:
            Trained Logistic calibrator
        """

        if len(y_pred_scores_uncalibrated) != len(y_true):
            raise ValueError("y_pred_scores_uncalibrated and y_true do not have the same length")

        if positive_class not in y_true:
            raise ValueError("The given positive class is not in y_true.")

        if sample_weight is not None:
            if len(sample_weight) != len(y_true):
                raise ValueError("sample_weigth,y_true and y_pred_scores are not of the same length")

        y_true_binarized = binarize_labels(y_true, positive_class)
        calibrator = LogisticRegression()
        calibrator.fit(
            y_pred_scores_uncalibrated.reshape(-1, 1), y_true_binarized.reshape(-1, 1), sample_weight=sample_weight
        )

        return cls(calibrator)

    def __init__(self, calibrator):
        self.calibrator = calibrator

    def calibrate(self, y_pred_scores_uncalibrated: typing.Iterable) -> typing.Iterable:

        """
        Calibrates the given uncalibrated probabilities.

        Arguments:

            y_pred_scores_uncalibrated -- uncalibrated scores predicted by a model.

        Returns:
            Calibrated scores
        """

        return self.calibrator.predict_proba(y_pred_scores_uncalibrated.reshape(-1, 1))[:, 1]

    def plot_calibration_map(
        self,
        min_score: float = 0,
        max_score: float = 1 + 1e-08,
        ax: Optional[Axes] = None,
        figsize: Optional[tuple] = None,
    ) -> Axes:

        """
        Plots the calibration map for the trained Logistic calibrator.

        Arguments:

            min_score -- default=0, bottom of the mapping interval.
            max_score -- default = 1+e-8, top of the mapping interval.


            These arguments can be changed if we pass uncalibrated scores outside the [0,1] interval.

        Returns:
            Axes for plotting
        """
        scores = np.linspace(min_score, max_score, 1000).reshape(-1, 1)
        mapped_scores = self.calibrate(scores)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax: Axes = fig.add_subplot()

        ax.plot(scores, mapped_scores, label="Logistic")
        ax.set_title(f"Calibration map", fontsize=14)
        ax.set_xlabel("Uncalibrated score", fontsize=14)
        ax.set_ylabel("Calibrated score", fontsize=14)
        ax.legend(loc="upper left", fontsize=12)
        return ax


class IsotonicCalibrator:
    @classmethod
    def train(
        cls,
        y_pred_scores_uncalibrated: typing.Iterable,
        y_true: typing.Iterable,
        positive_class: typing.Union[str, int, float],
        sample_weight: typing.Optional[typing.Iterable] = None,
    ):

        """
        Trains a logistic regression for calibration.

        Arguments:

            y_pred_scores_uncalibrated -- uncalibrated scores predicted by a model.
            y_true -- true labels for the observations. It is assumed that there are two possible true labels.
            positive_class -- class to be considered as positive.
            sample_weight -- weight of the samples

        Returns:
            Trained Isotonic calibrator
        """

        if positive_class not in y_true:
            raise ValueError("The given positive class is not in y_true.")

        if len(y_pred_scores_uncalibrated) != len(y_true):
            raise ValueError("y_pred_scores_uncalibrated and y_true do not have the same length")

        if sample_weight is not None:
            if len(sample_weight) != len(y_true):
                raise ValueError("sample_weigth,y_true and y_pred_scores are not of the same length")

        y_true_binarized = binarize_labels(y_true, positive_class)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(y_pred_scores_uncalibrated, y_true_binarized, sample_weight=sample_weight)

        return cls(calibrator)

    def __init__(self, calibrator):
        self.calibrator = calibrator

    def calibrate(self, y_pred_scores_uncalibrated: typing.Iterable) -> typing.Iterable:

        """
        Calibrates the given uncalibrated probabilities.

        Arguments:

            y_pred_scores_uncalibrated --  uncalibrated scores predicted by a model.

        Returns:
            Calibrated scores
        """

        return self.calibrator.predict(y_pred_scores_uncalibrated)

    def plot_calibration_map(
        self,
        min_score: float = 0,
        max_score: float = 1 + 1e-08,
        ax: Optional[Axes] = None,
        figsize: Optional[tuple] = None,
    ) -> Axes:

        """
        Plots the calibration map for the trained Isotonic calibrator.

        Arguments:

            min_score -- default=0, bottom of the mapping interval.
            max_score -- default = 1+e-8, top of the mapping interval.


            These arguments can be changed if we pass uncalibrated scores outside the [0,1] interval.

        Returns:
            Axes for plotting
        """

        scores = np.linspace(min_score, max_score, 1000).reshape(-1, 1)
        mapped_scores = self.calibrate(scores)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax: Axes = fig.add_subplot()

        ax.plot(scores, mapped_scores, label="Isotonic")
        ax.set_title(f"Calibration map", fontsize=14)
        ax.set_xlabel("Uncalibrated score", fontsize=14)
        ax.set_ylabel("Calibrated score", fontsize=14)
        ax.legend(loc="upper left", fontsize=12)
        return ax


class BetaCalibrator:
    @classmethod
    def train(
        cls,
        y_pred_scores_uncalibrated: typing.Iterable,
        y_true: typing.Iterable,
        positive_class: typing.Union[str, int, float],
        sample_weight: typing.Optional[typing.Iterable] = None,
    ):

        """
        Trains a logistic regression for calibration.

        Arguments:
            y_pred_scores_uncalibrated -- uncalibrated scores predicted by a model.
            y_true -- true labels for the observations. It is assumed that there are two possible true labels.
            positive_class -- class to be considered as positive.
            sample_weight -- weight of the samples
        Returns:

            Trained Beta calibrator
        """

        if positive_class not in y_true:
            raise ValueError("The given positive class is not in y_true.")

        if len(y_pred_scores_uncalibrated) != len(y_true):
            raise ValueError("y_pred_scores_uncalibrated and y_true do not have the same length")

        if sample_weight is not None:
            if len(sample_weight) != len(y_true):
                raise ValueError("sample_weigth,y_true and y_pred_scores are not of the same length")

        y_true_binarized = binarize_labels(y_true, positive_class)
        calibrator = BetaCalibration()
        calibrator.fit(
            y_pred_scores_uncalibrated.reshape(-1, 1), y_true_binarized.reshape(-1, 1), sample_weight=sample_weight
        )

        return cls(calibrator)

    def __init__(self, calibrator):
        self.calibrator = calibrator

    def calibrate(self, y_pred_scores_uncalibrated: typing.Iterable) -> typing.Iterable:

        """
        Calibrates the given uncalibrated probabilities.

        Arguments:

            y_pred_scores_uncalibrated -- uncalibrated scores predicted by a model.

        Returns:
            Calibrated scores
        """

        return self.calibrator.predict(y_pred_scores_uncalibrated.reshape(-1, 1))

    def plot_calibration_map(
        self,
        min_score: float = 0,
        max_score: float = 1 + 1e-08,
        ax: Optional[Axes] = None,
        figsize: Optional[tuple] = None,
    ) -> Axes:

        """
        Plots the calibration map for the trained Beta calibrator.

        Arguments:

            min_score -- default=0, bottom of the mapping interval.
            max_score -- default = 1+e-8, top of the mapping interval.

            These arguments can be changed if we pass uncalibrated scores outside the [0,1] interval.

        Returns:
            Axes for plotting
        """
        scores = np.linspace(min_score, max_score, 1000).reshape(-1, 1)
        mapped_scores = self.calibrate(scores)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax: Axes = fig.add_subplot()

        ax.plot(scores, mapped_scores, label="Beta")
        ax.set_title(f"Calibration map", fontsize=14)
        ax.set_xlabel("Uncalibrated score", fontsize=14)
        ax.set_ylabel("Calibrated score", fontsize=14)
        ax.legend(loc="upper left", fontsize=12)
        return ax


###########################################################################

### Reliability diagram function


def reliability_diagram(
    y_true: typing.Iterable,
    y_pred_scores: typing.Iterable,
    positive_class: typing.Union[str, int, float],
    n_bins: int = 5,
    bin_type: str = "uniform",
) -> typing.Tuple[typing.Iterable, typing.Iterable]:

    """
    Computes the necessary values to plot a reliability diagram/calibration curve.

    Arguments:
        y_true -- true labels for the observations. It is assumed that there are two possible true labels.
        y_pred_scores -- scores predicted by the model.
        positive_class -- class to be considered as positive.
        n_bins --  default=5, number of bins
        bin_type --  default = "uniform", binning method applied. If "quantile", it generates the bins taking into account n_bins quantiles.
        If "uniform", it generates n_bins of equal width in the interval [0,1].

    Returns:
        Fraction of true positive samples in each bin, mean predicted scores in each bin.
    """

    if len(y_true) != len(y_pred_scores):
        raise ValueError("y_true and y_pred_scores are not of the same length")

    if (max(y_pred_scores) > 1) or (min(y_pred_scores) < 0):
        raise ValueError("y_pred_scores has values outside the [0,1] interval")

    y_true_binarized = binarize_labels(y_true, positive_class=positive_class)
    # One line,extracted for sklearn :)
    frac_true, mean_pred_scores = calibration_curve(y_true_binarized, y_pred_scores, n_bins=n_bins, strategy=bin_type)
    return frac_true, mean_pred_scores


def plot_reliability_diagram(
    frac_true: typing.Iterable,
    mean_pred_scores: typing.Iterable,
    plot_ideal_calibration: Optional[bool] = False,
    label: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Optional[tuple] = None,
) -> Axes:

    """
    Plots a reliability diagram. The function assumes that there are two possible classes.

    Arguments:

        frac_true -- fraction of true positive samples in each bin
        mean_pred_scores -- mean predicted scores for the samples in each bin

    Returns:

        Axes for plotting.
    """

    if len(frac_true) != len(mean_pred_scores):
        raise ValueError("frac_true and mean_pred_scores are not of the same length")

    if (max(frac_true) > 1) or (min(frac_true) < 0):
        raise ValueError("frac_true has values outside the [0,1] interval")

    if (max(mean_pred_scores) > 1) or (min(mean_pred_scores) < 0):
        raise ValueError("mean_pred_scores has values outside the [0,1] interval")

    if len(mean_pred_scores) != len(list(set(mean_pred_scores))):
        raise ValueError("mean_pred_scores has repeated values.")

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax: Axes = fig.add_subplot()

    ax.plot(mean_pred_scores, frac_true, label=label)
    ax.set_xlabel("Mean predicted score", fontsize=14)
    ax.set_ylabel("True positive fraction", fontsize=14)
    

    if plot_ideal_calibration == True:
        x = np.linspace(0, 1, 100)
        ax.plot(x, x, linestyle="--", color="black",label="Perfect calibration")

    if label != None:
        ax.legend(loc="upper left", fontsize=12)
        

    return ax


####################################################################

### Scores



def log_loss(
    y_true: typing.Iterable,
    y_pred_scores: typing.Iterable,
    positive_class: typing.Union[str, int, float],
    sample_weight: typing.Optional[typing.Iterable] = None,
) -> float:

    """
    Computes log loss for the binary case 

    Arguments:
        y_true -- true labels for the observations. It is assumed that there are two possible true labels.
        y_pred_scores -- scores predicted by the model.
        positive_class -- class to be considered as positive.
        sample_weight -- weight of the samples

    Returns:
        Value of log loss.
    """

    if len(y_true) != len(y_pred_scores):
        raise ValueError("y_true and y_pred_scores are not of the same length")

    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weigth,y_true and y_pred_scores are not of the same length")

    if (max(y_pred_scores) > 1) or (min(y_pred_scores) < 0):
        raise ValueError("y_pred_scores has values outside the [0,1] interval")

    y_true_binarized = binarize_labels(y_true, positive_class)

    return np.average(feyn.losses.binary_cross_entropy(y_true_binarized,y_pred_scores),weights=sample_weight)


def brier_score_loss(
    y_true: typing.Iterable,
    y_pred_scores: typing.Iterable,
    positive_class: typing.Union[str, int, float],
    sample_weight: typing.Optional[typing.Iterable] = None,
) -> float:

    """
    Computes Brier Score for the binary case (i.e. mean squared error)

    Arguments:
        y_true -- true labels for the observations. It is assumed that there are two possible true labels.
        y_pred_scores -- scores predicted by the model.
        positive_class -- class to be considered as positive.
        sample_weight -- weight of the samples

    Returns:
        Value of Brier score.
    """
    if len(y_true) != len(y_pred_scores):
        raise ValueError("y_true and y_pred_scores are not of the same length")

    if sample_weight is not None:
        if len(sample_weight) != len(y_true):
            raise ValueError("sample_weigth,y_true and y_pred_scores are not of the same length")

    if (max(y_pred_scores) > 1) or (min(y_pred_scores) < 0):
        raise ValueError("y_pred_scores has values outside the [0,1] interval")

    y_true_binarized = binarize_labels(y_true, positive_class)

    return np.average((y_true_binarized - y_pred_scores) ** 2, weights=sample_weight)


def binary_ece(
    y_true: typing.Iterable,
    y_pred_scores: typing.Iterable,
    positive_class: typing.Union[str, int, float],
    n_bins: int = 5,
    bin_type: str = "uniform",
) -> float:

    """
    Computes the binary ECE.
    Arguments:
        y_true -- true labels for the observations. It is assumed that there are two possible true labels.
        y_pred_scores -- scores predicted by the model.
        positive_class -- class to be considered as positive.
        n_bins --  default=5, number of bins
        bin_type --  default = "uniform", binning method applied. If "quantile", it generates the bins taking into account n_bins quantiles.
        If "uniform", it generally n_bins of equal width in the interval [0,1].

    Returns:
        Binary ECE
    """

    if len(y_true) != len(y_pred_scores):
        raise ValueError("y_true and y_pred_scores are not of the same length")

    if (max(y_pred_scores) > 1) or (min(y_pred_scores) < 0):
        raise ValueError("y_pred_scores has values outside the [0,1] interval")

    y_true_binarized = binarize_labels(y_true, positive_class)

    if bin_type == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_pred_scores, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif bin_type == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    binids = np.digitize(y_pred_scores, bins) - 1
    # sums the predicted scores/ true positives in each bin to later compute the mean dividing by bin_total
    bin_true = np.bincount(binids, weights=y_true_binarized, minlength=len(bins))
    bin_pred_scores = np.bincount(binids, weights=y_pred_scores, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    frac_true = bin_true[nonzero] / bin_total[nonzero]
    mean_pred_scores = bin_pred_scores[nonzero] / (bin_total[nonzero])
    binary_ece_value = np.sum(np.abs(frac_true - mean_pred_scores) * (bin_total[nonzero] / len(y_true)))
    return binary_ece_value


def binary_mce(
    y_true: typing.Iterable,
    y_pred_scores: typing.Iterable,
    positive_class: typing.Union[str, int, float],
    n_bins: int = 5,
    bin_type: str = "uniform",
) -> float:

    """
    Computes binary MCE.

    Arguments:
        y_true -- true labels for the observations. It is assumed that there are two possible true labels.
        y_pred_scores -- scores predicted by the model.
        positive_class -- class to be considered as positive.
        n_bins --  default=5, number of bins
        bin_type --  default = "uniform", binning method applied. If "quantile", it generates the bins taking into account n_bins quantiles.
        If "uniform", it generally n_bins of equal width in the interval [0,1].

    Returns:
        Binary MCE.
    """

    if len(y_true) != len(y_pred_scores):
        raise ValueError("y_true and y_pred_scores are not of the same length")

    if (max(y_pred_scores) > 1) or (min(y_pred_scores) < 0):
        raise ValueError("y_pred_scores has values outside the [0,1] interval")

    y_true_binarized = binarize_labels(y_true, positive_class)

    if bin_type == "quantile":  
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_pred_scores, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
    elif bin_type == "uniform":
        bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)

    binids = np.digitize(y_pred_scores, bins) - 1
    # sums the predicted scores/ true positives in each bin to later compute the mean dividing by bin_total
    bin_true = np.bincount(binids, weights=y_true_binarized, minlength=len(bins))
    bin_pred_scores = np.bincount(binids, weights=y_pred_scores, minlength=len(bins))

    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    frac_true = bin_true[nonzero] / bin_total[nonzero]
    mean_pred_prob = bin_pred_scores[nonzero] / bin_total[nonzero]
    binary_mce_value = np.max(np.abs(frac_true - mean_pred_prob))
    return binary_mce_value
