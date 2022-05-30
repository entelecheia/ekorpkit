import logging
from ekorpkit import eKonf
from ekorpkit.utils.func import elapsed_timer
from ekorpkit.pipelines.pipe import apply_pipeline


log = logging.getLogger(__name__)


def topic_tasks(**cfg):
    args = eKonf.to_config(cfg)
    corpus_cfg = args.corpus
    model_cfg = args.model.topic
    subtasks = args.task.topic
    _subtasks_ = args.task.topic._subtasks_

    with elapsed_timer(format_time=True) as elapsed:
        model = eKonf.instantiate(model_cfg)
        model.corpora = eKonf.instantiate(corpus_cfg)

        for subtask in _subtasks_:
            subtask_cfg = subtasks[subtask]
            if eKonf.Keys.TARGET in subtask_cfg:
                log.info(f"Instantiate {subtask} ...")
                eKonf.instantiate(subtask_cfg, model=model)
            elif eKonf.Keys.NAME in subtask_cfg:
                log.info(f"Running model.{subtask} ...")
                getattr(model, subtask)(**subtask_cfg)
            else:
                log.warning(f"{subtask} is not a valid subtask")

        log.info(f">>> Elapsed time: {elapsed()} <<< ")


def corpora_tasks(**cfg):
    args = eKonf.to_config(cfg)
    corpus_cfg = args.corpus
    pipeline = args.get("pipeline", {})
    if pipeline is None:
        raise Exception("Pipeline is missing")

    with elapsed_timer(format_time=True) as elapsed:
        corpora = eKonf.instantiate(corpus_cfg)
        corpora.load()
        corpora.concat_corpora()

        _pipeline_ = pipeline.get(eKonf.Keys.PIPELINE, {})
        df = apply_pipeline(corpora._data, _pipeline_, pipeline)
        log.info(f">>> Elapsed time: {elapsed()} <<< ")

    return df


def corpus_tasks(**cfg):
    from ekorpkit.corpora import Corpus

    args = eKonf.to_config(cfg)
    corpus_cfg = args.corpus
    merge_metadata = args.get("merge_metadata", False)
    pipeline = args.get("pipeline", {})
    if pipeline is None:
        raise Exception("Pipeline is missing")

    with elapsed_timer(format_time=True) as elapsed:
        corpora = eKonf.instantiate(corpus_cfg)
        if isinstance(corpora, Corpus):
            corpora = [corpora]

        for corpus in corpora:
            log.info(f"::: processing {corpus.name}")
            if merge_metadata:
                corpus.merge_metadata()
            update_args = {"corpus_name": corpus.name}
            _pipeline_ = pipeline.get(eKonf.Keys.PIPELINE, {})
            df = apply_pipeline(
                corpus._data, _pipeline_, pipeline, update_args=update_args
            )
        log.info(f">>> Elapsed time: {elapsed()} <<< ")

    return df


def transfomer_finetune(**cfg):
    args = eKonf.to_config(cfg)
    model_cfg = args.model.transformer.finetune
    dataset_cfg = args.dataset

    if model_cfg._target_:
        model = eKonf.instantiate(model_cfg, dataset_cfg=dataset_cfg)
        model.apply_pipeline()
    else:
        raise Exception("Model instantiation target is missing")


def dataframe_tasks(**cfg):
    args = eKonf.to_config(cfg)
    pipeline_args = args.get("pipeline", {})

    if pipeline_args._target_:
        df = eKonf.instantiate(pipeline_args)
        if df is None:
            raise Exception("No dataframe found")
    else:
        raise Exception("Dataframe instantiation target is missing")

    return df
