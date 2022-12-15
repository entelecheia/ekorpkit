import logging
from ekorpkit import eKonf
from ekorpkit.models.metrics import evaluate_classification_performance


log = logging.getLogger(__name__)


def eval_classification(
    data=None, labels=None, columns=None, visualize=None, average="weighted", **args
):
    args = eKonf.to_config(args)
    eval_columns = columns.eval
    if labels:
        labels = eKonf.ensure_list(labels)
    if data is None:
        log.warning("No data to evaluate found")
        return
    data = data.dropna(subset=[eval_columns.actual, eval_columns.predicted])
    cm = evaluate_classification_performance(
        data[eval_columns.actual],
        data[eval_columns.predicted],
        labels=labels,
        average=average,
    )
    plot = visualize.plot
    plot.plots[0].display_labels = labels
    eKonf.instantiate(plot, data=cm)
