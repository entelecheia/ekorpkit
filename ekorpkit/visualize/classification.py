import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from .base import set_style, set_figure


def plot_confusion_matrix(
    confusion_matrix,
    display_labels="auto",
    matrix_labels=None,
    include_values=True,
    include_percentages=True,
    summary_stats=True,
    cbar=True,
    cmap="Blues",
    savefig={},
    plot={},
    figure={},
    verbose=False,
    **kwargs,
):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    confusion_matrix:       confusion matrix to be passed in
    display_labels:         List of strings containing the display_labels to be displayed on the x,y axis. Default is 'auto'
    matrix_labels:          List of strings that represent the labels row by row to be shown in each square.
    include_values:         If True, show the raw number in the confusion matrix. Default is True.
    include_percentages:    If True, show the proportions for each category. Default is True.
    summary_stats:          If True, display summary statistics below the figure. Default is True.
    cbar:                   If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                            Default is True.
    cmap:                   Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
    """
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(confusion_matrix.size)]

    if matrix_labels and len(matrix_labels) == confusion_matrix.size:
        matrix_labels = ["{}\n".format(value) for value in matrix_labels]
    else:
        matrix_labels = blanks

    if include_values:
        matrix_values = [
            "{0:0.0f}\n".format(value) for value in confusion_matrix.flatten()
        ]
    else:
        matrix_values = blanks

    if include_percentages:
        matrix_percentages = [
            "{0:.2%}".format(value)
            for value in confusion_matrix.flatten() / np.sum(confusion_matrix)
        ]
    else:
        matrix_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(matrix_labels, matrix_values, matrix_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(
        confusion_matrix.shape[0], confusion_matrix.shape[1]
    )

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if summary_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))

        # if it is a binary confusion matrix, show some more stats
        if len(confusion_matrix) == 2:
            # Metrics for Binary Confusion Matrices
            precision = confusion_matrix[1, 1] / sum(confusion_matrix[:, 1])
            recall = confusion_matrix[1, 1] / sum(confusion_matrix[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    set_style(**plot)
    figsize = plot.get("figsize", None)
    if figsize is not None and isinstance(figsize, str):
        figsize = eval(figsize)
    plt.figure(figsize=figsize, tight_layout=True)
    ax = plt.gca()

    sns.heatmap(
        confusion_matrix,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=display_labels,
        yticklabels=display_labels,
    )

    fig_args = figure.copy()
    xlabel = fig_args.get("xlabel", {}).get("xlabel")
    if xlabel:
        xlabel = xlabel + stats_text
    else:
        xlabel = stats_text
    fig_args.get("xlabel", {}).update({"xlabel": xlabel})

    set_figure(ax, **fig_args)

    plt.tight_layout()
    fname = savefig.get("fname", None)
    if fname:
        Path(fname).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(**savefig)
        if verbose:
            print(f"Saved figure to {fname}")
