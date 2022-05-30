import logging
from ekorpkit import eKonf
from ekorpkit.pipelines.pipe import apply_pipe


log = logging.getLogger(__name__)


def eval_classification(data=None, **args):
    args = eKonf.to_config(args)
    to_eval = args.to_eval
    pipe = args.pipeline.load_dataframe
    eval_metrics = args[eKonf.Keys.FUNC]
    if data is None:
        data = apply_pipe(data, pipe)
    eval_metrics = eKonf.partial(eval_metrics)
    cm = eval_metrics(data[to_eval.actual], data[to_eval.predicted])
    plot = args.visualize.plot
    eKonf.instantiate(plot, data=cm)
