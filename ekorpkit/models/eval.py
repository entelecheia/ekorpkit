import logging
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def eval_classification(data=None, **args):
    args = eKonf.to_config(args)
    _eval_ = args._eval_
    labels = args.labels
    if labels:
        labels = eKonf.ensure_list(labels)
    eval_metrics = args[eKonf.Keys.FUNC]
    if data is None:
        data = eKonf.load_data(**args.path.data)
    if data is None:
        log.warning(f"No data to evaluate on given path: {args.path.data}")
        return
    eval_metrics = eKonf.partial(eval_metrics)
    data = data.dropna(subset=[_eval_.actual, _eval_.predicted])
    cm = eval_metrics(data[_eval_.actual], data[_eval_.predicted], labels=labels)
    plot = args.visualize.plot
    plot.plots[0].display_labels = labels
    eKonf.instantiate(plot, data=cm)
