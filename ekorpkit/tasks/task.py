from ekorpkit import eKonf
from hydra.utils import instantiate
from wasabi import msg
from ekorpkit.utils.func import elapsed_timer
from ekorpkit.pipelines.pipe import apply_pipeline


def topic_tasks(**cfg):
    args = eKonf.to_config(cfg)
    corpus_cfg = args.corpus
    model_cfg = args.model.topic
    subtasks = args.task.topic
    _subtasks_ = args.task.topic._subtasks_

    with elapsed_timer(format_time=True) as elapsed:
        model = instantiate(model_cfg, _recursive_=False)
        model.corpora = instantiate(corpus_cfg, _recursive_=False)

        for subtask in _subtasks_:
            subtask_cfg = subtasks[subtask]
            if "_target_" in subtask_cfg:
                msg.info(f"Instantiate {subtask} ...")
                instantiate(subtask_cfg, model=model, _recursive_=False)
            elif "_name_" in subtask_cfg:
                msg.info(f"Running model.{subtask} ...")
                getattr(model, subtask)(**subtask_cfg)
            else:
                msg.fail(f"{subtask} is not a valid subtask")

        print(f"\n >>> Elapsed time: {elapsed()} <<< ")


def corpora_tasks(**cfg):
    args = eKonf.to_config(cfg)
    corpus_cfg = args.corpus
    merge_metadata = args.get("merge_metadata", False)
    pipeline = args.get("pipeline", {})
    if pipeline is None:
        raise Exception("Pipeline is missing")

    with elapsed_timer(format_time=True) as elapsed:
        corpora = eKonf.instantiate(corpus_cfg)
        corpora.load()
        corpora.concat_corpora()

        if merge_metadata:
            corpora.merge_metadata()
        _pipeline_ = pipeline.get("_pipeline_", {})
        df = apply_pipeline(corpora._data, _pipeline_, pipeline)
        print(f"\n >>> Elapsed time: {elapsed()} <<< ")

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
            print(f"::: processing {corpus.name}")
            if merge_metadata:
                corpus.merge_metadata()
            update_args = {"corpus_name": corpus.name}
            _pipeline_ = pipeline.get("_pipeline_", {})
            df = apply_pipeline(corpus._data, _pipeline_, pipeline, update_args=update_args)
        print(f"\n >>> Elapsed time: {elapsed()} <<< ")

    return df


def transfomer_finetune(**cfg):
    args = eKonf.to_config(cfg)
    model_cfg = args.model.transformer.finetune
    dataset_cfg = args.dataset

    if model_cfg._target_:
        model = instantiate(model_cfg, dataset_cfg=dataset_cfg, _recursive_=False)
        model.apply_pipeline()
    else:
        raise Exception("Model instantiation target is missing")


def dataframe_tasks(**cfg):
    args = eKonf.to_config(cfg)
    pipeline_args = args.get("pipeline", {})

    if pipeline_args._target_:
        df = instantiate(pipeline_args, _recursive_=False)
        if df is None:
            raise Exception("No dataframe found")
    else:
        raise Exception("Dataframe instantiation target is missing")

    return df
