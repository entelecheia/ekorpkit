import logging
from ekorpkit import eKonf
from ekorpkit.pipelines.pipe import apply_pipe


log = logging.getLogger(__name__)


def eval_classification(data=None, **args):
    args = eKonf.to_config(args)
    _eval_ = args._eval_
    eval_metrics = args[eKonf.Keys.FUNC]
    if data is None:
        data = eKonf.load_data(**args.path.data)
    eval_metrics = eKonf.partial(eval_metrics)
    cm = eval_metrics(data[_eval_.actual], data[_eval_.predicted])
    plot = args.visualize.plot
    eKonf.instantiate(plot, data=cm)
