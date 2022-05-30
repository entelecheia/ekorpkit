import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .base import set_figure
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def confusion_matrix(ax=None, x=None, y=None, data=None, **kwargs):
    _parms_ = {} or kwargs.get(eKonf.Keys.PARMS)
    _figure = {} or kwargs.get("figure")
    if ax is None:
        ax = plt.gca()

    log.info(f"Confusion matrix: {kwargs}")

    display_labels = kwargs.get("display_labels") or "auto"
    matrix_labels = kwargs.get("matrix_labels")
    include_values = kwargs.get("include_values")
    include_percentages = kwargs.get("include_percentages")
    summary_stats = kwargs.get("summary_stats")

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(data.size)]

    if matrix_labels and len(matrix_labels) == data.size:
        matrix_labels = ["{}\n".format(value) for value in matrix_labels]
    else:
        matrix_labels = blanks

    if include_values:
        matrix_values = ["{0:0.0f}\n".format(value) for value in data.flatten()]
    else:
        matrix_values = blanks

    if include_percentages:
        matrix_percentages = [
            "{0:.2%}".format(value) for value in data.flatten() / np.sum(data)
        ]
    else:
        matrix_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(matrix_labels, matrix_values, matrix_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(data.shape[0], data.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if summary_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(data) / float(np.sum(data))

        # if it is a binary confusion matrix, show some more stats
        if len(data) == 2:
            # Metrics for Binary Confusion Matrices
            precision = data[1, 1] / sum(data[:, 1])
            recall = data[1, 1] / sum(data[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    sns.heatmap(
        data,
        annot=box_labels,
        fmt="",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
        **_parms_,
    )

    xlabel = _figure.get("xlabel")
    if isinstance(xlabel, str):
        _figure["xlabel"] = xlabel + stats_text
    else:
        _figure["xlabel"] = stats_text

    set_figure(ax, **_figure)
