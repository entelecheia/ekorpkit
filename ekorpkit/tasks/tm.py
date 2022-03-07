import pandas as pd
import os
import orjson as json
from tqdm import tqdm
from pathlib import Path
from timeit import default_timer as timer
from omegaconf import OmegaConf
from hydra.utils import instantiate
from ..models.topic import TopicModel, IDF, ONE
# from ..corpora.loader import eKorpkit, load_corpus_paths
# from ..corpora import CORPUS_DF


def topic_model(**args):
    cfg = OmegaConf.create(args)
    task_args = cfg.task

    tm = instantiate(cfg.model.topic, _recursive_=False)

    if "train" in task_args:

        # tm = TopicModel(model_name=model_name,
        #                 model_dir=model_dir, output_dir=args.output_dir,
        #                 ngram_delimiter='',
        #                 stopwords_path=args.stopwords_path,
        #                 word_prior_path=args.word_prior_path,
        #                 corpus_names=corpus_names,
        #                 corpus_dir=corpus_dir,
        #                 num_workers=args.num_workers)
        tm.load_corpus(reload=True)
        tm.train_model(
            model_type=args.model_type,
            save=True,
            k=args.k,
            alpha=0.1,
            eta=0.01,
            tw=IDF,
            rm_top=5,
            eval_coherence=False,
            set_word_prior=True,
            iterations=args.iterations,
            interval=args.interval,
        )
        tm.visualize()
        tm.label_topics()
        tm.topic_wordclouds(ncols=args.wc_ncols, nrows=args.wc_nrows)

    elif "label" in task_args:
        # tm = TopicModel(model_name=model_name, model_dir=model_dir, output_dir=args.output_dir)
        tm.load_model(model_file=args.model_file)
        tm.label_topics()
        tm.topic_wordclouds(ncols=args.wc_ncols, nrows=args.wc_nrows)

    elif "topic_names" in task_args:
        # tm = TopicModel(model_name=model_name, model_dir=model_dir, output_dir=args.output_dir)
        tm.load_model(model_file=args.model_file)
        tm.save_labels(names=args.topic_names)
        tm.topic_wordclouds(ncols=args.wc_ncols, nrows=args.wc_nrows)

    elif "infer" in task_args:
        # tm = TopicModel(model_name=model_name, model_dir=model_dir, output_dir=args.output_dir)
        tm.load_model(model_file=args.model_file)
        tm.infer_corpus(
            args.input_dir,
            args.corpus_names,
            args.export_dir,
            compress=args.compress,
            num_workers=args.num_workers,
        )

    elif "export_samples" in task_args:
        # tm = TopicModel(model_name=model_name, model_dir=model_dir, output_dir=args.output_dir)
        # tm.load_model(model_file=args.model_file)
        tm.export_samples(**cfg.task.export_samples)

    elif "export_corpus_samples" in task_args:
        corpus_args = cfg.corpus
        model_args = cfg.model.topic
        # export_corpus_samples(cfg.task.export_corpus_samples, corpus_args, model_args)

    elif "transform" in task_args:
        column_evals = args.column_evals
        if isinstance(column_evals, str):
            column_evals = [column_evals]
        aggregations = args.aggregations
        if aggregations:
            aggregations = json.loads(aggregations.replace("'", '"'))
            print(aggregations)

        # tm = TopicModel(model_name=model_name, model_dir=model_dir)
        # tm.load_model(model_file=args.model_file)
        tm.transform_topic_dists(
            args.input_dir,
            args.corpus_names,
            args.export_dir,
            compress=args.compress,
            column_evals=column_evals,
            aggregations=aggregations,
            groupby_cols=args.groupby_cols,
        )

    else:
        raise ValueError("Not found any proper subtask. Check the `subtask` argument")
    print("All tasks completed.")


# def export_corpus_samples(args, corpus_args, model_args):
#     os.makedirs(os.path.abspath(args.export_dir), exist_ok=True)

#     sample_ids = {}
#     sample_dfs = {}
#     for sample_type in args.sample_types:
#         filename = '{}-{}-{}.csv'.format(model_args.model_name, args.model_id, sample_type)
#         sample_id_path = f'{model_args.output_dir}/{model_args.model_name}/{filename}'
#         col_names = pd.read_csv(sample_id_path, index_col=0, nrows=0).columns
#         col_types = {col: str for col in col_names}
#         df_id = pd.read_csv(sample_id_path, index_col=None, dtype=col_types)
#         sample_ids[sample_type] = df_id
#         sample_dfs[sample_type] = []

#     corpus_paths = load_corpus_paths(corpus_args.corpus_dir, corpus_args.name,
#         corpus_type=corpus_args.corpus_type, corpus_filetype=corpus_args.corpus_filetype)
#     for i_corpus, (corpus_name, corpus_file) in enumerate(corpus_paths):
#         print('-' * 120)
#         corpus = eKorpkit.load(corpus_name, corpus_dir=corpus_file,
#                                 corpus_type=corpus_args.corpus_type,
#                                 load_light=False,
#                                 load_dataframe=True, id_keys=corpus_args.id_keys)
#         for sample_type in args.sample_types:
#             df_id = sample_ids[sample_type]
#             df = df_id.query('corpus_filename.str.startswith(@corpus_name)', engine='python')
#             df = df.merge(corpus.docs, on=list(args.merge_cols))
#             df.text = df.text.str.replace('\n\n', '\n')
#             if len(df.index) > 0:
#                 df['corpus_filename'] = corpus_name
#                 df.rename(columns={'corpus_filename': 'corpus'}, inplace=True)
#                 print(df.head())
#                 sample_dfs[sample_type].append(df)

#     combined_dfs = []
#     for sample_type in args.sample_types:
#         df_exp = pd.concat(sample_dfs[sample_type])
#         print(df_exp.head())
#         filename = '{}-{}-{}-samples.csv'.format(model_args.model_name, args.model_id, sample_type)
#         export_path = f'{args.export_dir}/{filename}'
#         df_exp.to_csv(export_path, index=False)
#         filename = '{}-{}-{}-samples.json'.format(model_args.model_name, args.model_id, sample_type)
#         export_path = f'{args.export_dir}/{filename}'
#         df_exp.to_json(export_path, orient='records', force_ascii=False, indent=4)
#         df_exp['sample_type'] = sample_type
#         combined_dfs.append(df_exp)

#     combined_df = pd.concat(combined_dfs)
#     filename = '{}-{}-samples.csv'.format(model_args.model_name, args.model_id)
#     export_path = f'{args.export_dir}/{filename}'
#     combined_df.to_csv(export_path, index=False)
#     filename = '{}-{}-samples.json'.format(model_args.model_name, args.model_id)
#     export_path = f'{args.export_dir}/{filename}'
#     combined_df.to_json(export_path, orient='records', force_ascii=False, indent=4)
