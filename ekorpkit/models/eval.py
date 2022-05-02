import logging
from ekorpkit import eKonf
from ekorpkit.pipelines.pipe import apply_pipe


log = logging.getLogger(__name__)


def eval_classification(data=None, **args):
    args = eKonf.to_config(args)
    column_info = args.column_info
    pipe = args.pipeline.load_dataframe
    eval_metrics = args.method
    if data is None:
        data = apply_pipe(data, pipe)
    eval_metrics = eKonf.instantiate(eval_metrics)
    cm = eval_metrics(data[column_info.actual], data[column_info.predicted])
    plot = args.visualize.plot
    eKonf.instantiate(plot, data=cm)
