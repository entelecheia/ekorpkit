from tqdm.auto import tqdm

from .hydra import SpecialKeys, _partial
from .utils.batch import batcher, decorator_apply
from .utils.logging import getLogger

logger = getLogger(__name__)


def _pipe(data, pipe):
    _func_ = pipe.get(SpecialKeys.FUNC)
    _fn = _partial(_func_)
    logger.info("Applying pipe: %s", _fn)
    if isinstance(data, dict):
        if "concat_dataframes" in str(_fn):
            return _fn(data, pipe)
        dfs = {}
        for df_no, df_name in enumerate(data):
            df_each = data[df_name]

            logger.info(
                "Applying pipe to dataframe [%s], %d/%d", df_name, df_no + 1, len(data)
            )

            pipe[SpecialKeys.SUFFIX.value] = df_name
            dfs[df_name] = _fn(df_each, pipe)
        return dfs
    return _fn(data, pipe)


def _apply(
    func,
    series,
    description=None,
    use_batcher=True,
    minibatch_size=None,
    num_workers=None,
    **kwargs,
):
    batcher_instance = batcher.batcher_instance
    if use_batcher and batcher_instance is not None:
        if batcher_instance is not None:
            batcher_minibatch_size = batcher_instance.minibatch_size
        else:
            batcher_minibatch_size = 1000
        if minibatch_size is None:
            minibatch_size = batcher_minibatch_size
        if num_workers is not None:
            batcher_instance.procs = int(num_workers)
        if batcher_instance.procs > 1:
            batcher_instance.minibatch_size = min(
                int(len(series) / batcher_instance.procs) + 1, minibatch_size
            )
            logger.info(
                f"Using batcher with minibatch size: {batcher_instance.minibatch_size}"
            )
            results = decorator_apply(func, batcher_instance, description=description)(
                series
            )
            if batcher_instance is not None:
                batcher_instance.minibatch_size = batcher_minibatch_size
            return results

    if batcher_instance is None:
        logger.info("Warning: batcher not initialized")
    tqdm.pandas(desc=description)
    return series.progress_apply(func)
