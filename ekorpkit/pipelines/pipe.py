import codecs
import os
from collections import OrderedDict
from functools import partial, reduce

import ekorpkit.config as config
import numpy as np
import pandas as pd
from ekorpkit.io.file import get_filepaths, load_dataframe, save_dataframe
from ekorpkit.utils import print_status
from ekorpkit.utils.batch import decorator_apply
from ekorpkit.utils.func import check_max_len, check_min_len, elapsed_timer
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig
from tqdm.auto import tqdm
from wasabi import msg


def apply(
    func,
    series,
    description=None,
    verbose=False,
    use_batcher=True,
    minibatch_size=None,
    **kwargs,
):
    batcher = config.batcher
    if use_batcher and batcher is not None:
        batcher_minibatch_size = batcher.minibatch_size
        if minibatch_size is None:
            minibatch_size = batcher_minibatch_size
        if batcher.procs > 1:
            batcher.minibatch_size = min(
                int(len(series) / batcher.procs) + 1, minibatch_size
            )
            if verbose:
                msg.info(f"Using batcher with minibatch size: {batcher.minibatch_size}")
            results = decorator_apply(func, batcher, description=description)(series)
            batcher.minibatch_size = batcher_minibatch_size
            return results

    if verbose and batcher is None:
        msg.warn("Warning: batcher not initialized")
    tqdm.pandas(desc=description)
    return series.progress_apply(func)


def apply_pipe(df, pipe):
    fn = instantiate(pipe["method"], _recursive_=False)
    print(f"\nApplying pipe: {fn}")
    if isinstance(df, list):
        if "concat_dataframes" in str(fn):
            return fn(df, pipe)
        else:
            dfs = []
            for df_no, df_each in enumerate(df):
                print(f"Applying pipe to dataframe {(df_no+1)}/{len(df)}")
                pipe["dataframe_no"] = df_no
                dfs.append(fn(df_each, pipe))
            return dfs
    else:
        return fn(df, pipe)
    # df = df.pipe(fn, pipe)
    # return df


def apply_pipeline(df, pipeline, pipeline_args, update_args={}, verbose=True):
    pipeline_targets = []
    pipes = OrderedDict()
    if isinstance(pipeline, (list, ListConfig)):
        for pipe in pipeline:
            pipes[pipe] = pipe
    elif isinstance(pipeline, str):
        pipes[pipeline] = pipeline
    else:
        pipes = pipeline
    if pipes is None or len(pipes) == 0:
        if verbose:
            print("No pipeline specified")
        return df
    if verbose:
        print(f"Applying pipeline: {pipes}")
    for pipe, pipe_arg_name in pipes.items():
        args = dict(pipeline_args.get(pipe_arg_name, {}))
        if pipe != pipe_arg_name:
            args_override = dict(pipeline_args.get(pipe, {}))
            args.update(args_override)
        if args and isinstance(args, dict):
            args.update(update_args)
            pipeline_targets.append(args)

    return reduce(apply_pipe, pipeline_targets, df)


def eval_columns(df, args):
    verbose = args.get("verbose", False)
    expressions = args.get("expressions", None)
    engine = args.get("engine", None)
    eval_at = args.get("eval_at", "dataframe")
    if expressions is None:
        if verbose:
            print("No expressions specified")
        return df
    if verbose:
        print(f"Eval columns: {args}")
    if eval_at == "dataframe":
        if isinstance(expressions, (list, ListConfig)):
            for expr in expressions:
                df.eval(expr, engine=engine, inplace=True)
        else:
            for col, expr in expressions.items():
                df[col] = df.eval(expr, engine=engine)
    else:
        if isinstance(expressions, (list, ListConfig)):
            for expr in expressions:
                pd.eval(expr, engine=engine, inplace=True, target=df)
        else:
            for col, expr in expressions.items():
                df[col] = pd.eval(expr, engine=engine)

    if verbose:
        print(df.tail())
    return df


def combine_columns(df, args):
    verbose = args.get("verbose", False)
    columns = args.get("columns", None)
    if columns is None:
        if verbose:
            print("No columns specified")
        return df
    if verbose:
        print(f"Combining columns: {args}")
    for col in columns:
        df[col].fillna("", inplace=True)
        df[col] = df[col].astype(str)
    separator = codecs.decode(args["separator"], "unicode_escape")
    df[args["into"]] = df[columns].agg(separator.join, axis=1)
    return df


def drop(df, args):
    verbose = args.get("verbose", False)
    labels = args.get("labels", None)
    axis = args.get("axis", 1)
    columns = args.get("columns", None)
    index = args.get("index", None)
    level = args.get("level", None)
    errors = args.get("errors", "ignore")
    if verbose:
        print(f"droping: {args}")
    df = df.drop(
        columns=columns,
        axis=axis,
        labels=labels,
        index=index,
        level=level,
        errors=errors,
    )
    if verbose:
        print(df.head())
    return df


def melt(df, args):
    verbose = args.get("verbose", False)
    id_vars = args.get("id_vars", None)
    value_vars = args.get("value_vars", None)
    var_name = args.get("var_name", None)
    value_name = args.get("value_name", None)
    col_level = args.get("col_level", None)
    ignore_index = args.get("ignore_index", True)
    if id_vars is None:
        if verbose:
            print("No id_vars specified")
        return df
    if verbose:
        print(f"Melting columns: {args}")
    id_vars = list(id_vars)
    if value_vars:
        if isinstance(value_vars, str):
            value_vars = eval(value_vars)
        else:
            value_vars = list(value_vars)
    df = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
        col_level=col_level,
        ignore_index=ignore_index,
    )
    if verbose:
        print(df.head())
    return df


def plot(df, args):
    verbose = args.get("verbose", False)
    plot_cfg = args.get("visualize", {}).get("plot", {})
    if "_target_" not in plot_cfg:
        print("No target specified")
        return df
    if verbose:
        print(f"Plotting: {plot_cfg}")
    subset = args.get("subset", {})
    col = subset.get("column", None)
    values = subset.get("values", None)
    titles = subset.get("titles", None)
    output_file = args.get("output_file", None)
    output_dir = args.get("output_dir", ".")
    if subset and col:
        title_prefix = plot_cfg["figure"].get("title", None)
        title_prefix = title_prefix + " - " if title_prefix else ""
        if isinstance(values, str):
            values = [values]
        if titles and isinstance(titles, str):
            titles = [titles]
        for i, val in enumerate(values):
            if verbose:
                print(f"Plotting subset: {col} == {val}")
            df_sub = df[df[col] == val]
            if output_file and "{}" in output_file:
                output_path = output_file.replace("{}", str(val))
                output_path = f"{output_dir}/{output_path}"
                plot_cfg["savefig"]["fname"] = output_path
            if titles and len(titles) > i:
                plot_cfg["figure"]["title"] = title_prefix + titles[i]
            if verbose:
                print(f"Plotting: {plot_cfg}")
                # print(df_sub.head())
            instantiate(plot_cfg, df=df_sub, _recursive_=False)
    else:
        instantiate(plot_cfg, df=df, _recursive_=False)

    return df


def pivot(df, args):
    verbose = args.get("verbose", False)
    index = args.get("index", None)
    columns = args.get("columns", None)
    values = args.get("values", None)
    fillna = args.get("fillna", None)
    reset_index = args.get("reset_index", True)
    if index is None:
        print("No index specified")
        return df
    if columns is None:
        print("No columns specified")
        return df
    if values is None:
        print("No values specified")
        return df
    if verbose:
        print(f"Pivoting columns: {args}")
    if isinstance(index, (list, ListConfig)):
        index = list(index)
    if isinstance(columns, (list, ListConfig)):
        columns = list(columns)
    if isinstance(values, (list, ListConfig)):
        values = list(values)
    if len(values) == 1:
        values = values[0]
    df = df.pivot(index=index, columns=columns, values=values)
    # df = df.set_index(index)[values].unstack()
    if fillna is not None:
        df.fillna(fillna, inplace=True)
    if reset_index:
        if verbose:
            print(f"Resetting index, nlevels of columns: {df.columns.nlevels}")
        df.reset_index(inplace=True)
        if df.columns.nlevels  > 1:
            df.columns = ["_".join(a).strip("_") for a in df.columns.to_flat_index()]
    if verbose:
        print(df.head())
    return df


def split_sampling(df, args):
    from sklearn.model_selection import train_test_split

    verbose = args.get("verbose", False)
    stratify_on = args.get("stratify_on", None)
    random_state = args.get("random_state", 123)
    groupby = args.get("groupby", stratify_on)
    unique_key = args.get("unique_key", "id")
    test_size = args.get("test_size", 0.1)
    dev_size = args.get("dev_size", None)
    if isinstance(stratify_on, ListConfig):
        stratify_on = list(stratify_on)
    if isinstance(groupby, ListConfig):
        groupby = list(groupby)
    if verbose:
        print(f"Split sampling: {args}")

    train, dev, test = None, None, None
    if stratify_on is None:
        train, test = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        if dev_size:
            train, dev = train_test_split(
                train, test_size=dev_size, random_state=random_state
            )
    else:
        train, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[stratify_on]
        )
        if dev_size:
            train, dev = train_test_split(
                train,
                test_size=dev_size,
                random_state=random_state,
                stratify=train[stratify_on],
            )
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    if dev_size:
        dev.reset_index(drop=True, inplace=True)
    if verbose:
        print(f"Total rows: {len(df)}")
        print(f"Train: {len(train)}")
        print(f"Test: {len(test)}")
        if dev_size:
            print(f"Dev: {len(dev)}")

        grp_all = (
            df.groupby(groupby)[unique_key]
            .count()
            .rename("population")
            .transform(lambda x: x / x.sum() * 100)
        )
        grp_train = (
            train.groupby(groupby)[unique_key]
            .count()
            .rename("train")
            .transform(lambda x: x / x.sum() * 100)
        )
        grp_test = (
            test.groupby(groupby)[unique_key]
            .count()
            .rename("test")
            .transform(lambda x: x / x.sum() * 100)
        )
        grp_dists = pd.concat([grp_all, grp_train, grp_test], axis=1)
        if dev_size:
            grp_dev = (
                dev.groupby(groupby)[unique_key]
                .count()
                .rename("dev")
                .transform(lambda x: x / x.sum() * 100)
            )
            grp_dists = pd.concat([grp_dists, grp_dev], axis=1)
        print(grp_dists)

    output_dir = args.get("output_dir", ".")
    train_file = args.get("train_file", None)
    test_file = args.get("test_file", None)
    dev_file = args.get("dev_file", None)
    if train_file and train is not None:
        filepath = f"{output_dir}/{train_file}"
        save_dataframe(train, filepath, verbose=verbose)
    if test_file and test is not None:
        filepath = f"{output_dir}/{test_file}"
        save_dataframe(test, filepath, verbose=verbose)
    if dev_file and dev is not None:
        filepath = f"{output_dir}/{dev_file}"
        save_dataframe(dev, filepath, verbose=verbose)

    return df


def top_values(df, args):
    verbose = args.get("verbose", False)
    groupby = args.get("groupby", None)
    value_var = args.get("value_var", None)
    value_label = args.get("value_label", None)
    value_name = args.get("value_name", None)
    value_separator = args.get("value_separator", ", ")
    top_n = args.get("top_n", 5)
    columns_to_keep = args.get("columns_to_keep", None)
    use_batcher = args.get("use_batcher", True)
    minibatch_size = args.get("minibatch_size", None)

    if isinstance(groupby, ListConfig):
        groupby = list(groupby)
    if verbose:
        print(f"Split sampling: {args}")

    def label(row):
        return f"{row[value_label]}[{round(row[value_var]*100,0):.0f}%]"

    if value_label is not None and value_name is not None:
        # df[value_name] = df.apply(label, axis=1)
        df[value_name] = apply(
            label,
            df,
            description="labeling",
            verbose=verbose,
            use_batcher=use_batcher,
            minibatch_size=minibatch_size,
        )

    top_n_grp = (
        df.sort_values(by=value_var, ascending=False)
        .groupby(groupby)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_n_grp = top_n_grp.sort_values(by=groupby)
    top_n_grp = top_n_grp.groupby(groupby, as_index=False).agg(
        {value_name: value_separator.join}
    )
    top_n_grp = top_n_grp[groupby + [value_name]]
    df.drop(columns=value_name, inplace=True)
    df = pd.merge(df, top_n_grp, on=groupby, how="left")
    if columns_to_keep:
        df = df[columns_to_keep]
    if verbose:
        print(df.head())

    return df


def sampling(df, args):
    verbose = args.get("verbose", False)
    random_state = args.get("random_state", 123)
    groupby = args.get("groupby", None)
    sample_size_per_group = args.get("sample_size_per_group", None)
    value_var = args.get("value_var", None)
    columns_to_keep = args.get("columns_to_keep", None)

    if isinstance(groupby, ListConfig):
        groupby = list(groupby)
    if verbose:
        print(f"Split sampling: {args}")

    if groupby is None:
        if sample_size_per_group < 1:
            df_sample = df.sample(frac=sample_size_per_group, random_state=random_state)
        else:
            df_sample = df.sample(n=sample_size_per_group, random_state=random_state)
    else:
        if sample_size_per_group < 1:
            df_sample = df.groupby(groupby, group_keys=False).apply(
                lambda x: x.sample(
                    frac=sample_size_per_group, random_state=random_state
                )
            )
        else:
            df_sample = df.groupby(groupby, group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), sample_size_per_group), random_state=random_state
                )
            )
    df_sample.reset_index(drop=True, inplace=True)
    if columns_to_keep:
        df_sample = df_sample[columns_to_keep]
    if verbose:
        print(f"Total rows: {len(df)}")
        print(f"Sample: {len(df_sample)}")
        print(df_sample.head())

    if groupby is not None and verbose:
        grp_all = df.groupby(groupby)[value_var].count().rename("population")
        grp_sample = df_sample.groupby(groupby)[value_var].count().rename("sample")
        grp_dists = pd.concat([grp_all, grp_sample], axis=1)
        print(grp_dists)

    output_dir = args.get("output_dir", ".")
    output_file = args.get("output_file", None)
    if output_file:
        filepath = f"{output_dir}/{output_file}"
        save_dataframe(
            df_sample, filepath, verbose=verbose, columns_to_keep=columns_to_keep
        )

    return df


def aggregate_columns(df, args):
    verbose = args.get("verbose", False)
    onto_column = args.get("onto", None)
    aggregations = args.get("aggregations", None)
    reset_index = args.get("reset_index", False)
    if onto_column is None and aggregations is None:
        if verbose:
            print("No columns or aggregations are specified")
        return df
    groupby_cloumns = args["groupby"]
    if groupby_cloumns is None:
        if verbose:
            print("No groupby specified")
        return df
    if isinstance(groupby_cloumns, ListConfig):
        groupby_cloumns = list(groupby_cloumns)
    separator = codecs.decode(args["separator"], "unicode_escape")

    num_docs = df.shape[0]
    if verbose:
        print(f"Aggregating columns: {args}")
    if aggregations:
        if isinstance(aggregations, DictConfig):
            aggregations = dict(aggregations)
        df = df.groupby(groupby_cloumns, as_index=False).agg(aggregations)
    else:
        df[onto_column].fillna("", inplace=True)
        df = df.groupby(groupby_cloumns, as_index=False).agg(
            {onto_column: separator.join}
        )
    if reset_index:
        if verbose:
            print(f"Resetting index, nlevels of columns: {df.columns.nlevels}")
        df.reset_index(inplace=True)
        if df.columns.nlevels  > 1:
            df.columns = ["_".join(a).strip("_") for a in df.columns.to_flat_index()]
    n_docs = df.shape[0]
    if verbose:
        print(df.tail())
        print(f"{num_docs} documents aggregated into {n_docs} documents")
    return df


def explode_splits(df, args):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    separator = codecs.decode(args["separator"], "unicode_escape")
    id_key = args.get("id_key", "id")
    split_key = args.get("split_key", "seg_id")
    if isinstance(id_key, ListConfig):
        id_key = list(id_key)
    if verbose:
        print(f"Exploding column: {args}")

    num_docs = df.shape[0]
    df[apply_to] = df[apply_to].str.split(separator)
    df = df.explode(apply_to)
    df[split_key] = df.groupby(id_key).cumcount()
    n_docs = df.shape[0]
    if verbose:
        print(f"{num_docs} documents exploded into {n_docs} documents")
    return df


def rename_columns(df, args):
    verbose = args.get("verbose", False)
    new_names = args.get("new_names", None)
    if new_names is None:
        if verbose:
            print("No columns specified")
        return df
    if verbose:
        print(f"Renaming columns: {args}")
    if new_names is not None:
        df.rename(columns=new_names, inplace=True)
    if verbose:
        print(df.head())
    return df


def reset_index(df, args):
    verbose = args.get("verbose", False)
    index_column_name = args.get("index_column_name", None)
    if index_column_name is None:
        index_column_name = "index"
    drop_index = args.get("drop_index", False)
    if verbose:
        print(f"Resetting index: {args}")
    df = df.reset_index(drop=drop_index)
    if not drop_index and index_column_name != "index":
        df.rename(columns={"index": index_column_name}, inplace=True)
    if verbose:
        print(df.head())
    return df


def normalize(df, args):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    normalizer = args.get("normalizer", None)
    if normalizer is None:
        if verbose:
            print("No normalizer specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        print(f"Normalizing text: {args}")
    if verbose:
        print("instantiating normalizer")
    normalizer = instantiate(normalizer)
    for key in apply_to:
        if verbose:
            print(f"\nPreprocessing column: {key}")
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = apply(
                normalizer.normalize,
                df[key],
                description=f"Normalizing column: {key}",
                verbose=verbose,
            )
            if verbose:
                msg.good("\n >> elapsed time to normalize: {}\n".format(elapsed()))
    return df


def fillna(df, args):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    fill_with = args.get("fill_with", "")
    if verbose:
        print(f"Filling missing values: {args}")
    for key in apply_to:
        if verbose:
            print(f"\nPreprocessing column: {key}")
        df[key].fillna(fill_with, inplace=True)
    return df


def segment(df, args):
    verbose = args.get("verbose", False)
    use_batcher = args.get("use_batcher", True)
    minibatch_size = args.get("minibatch_size", None)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    segmenter = args.get("segmenter", None)
    if segmenter is None:
        if verbose:
            print("No segmenter specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        print(f"Splitting text: {args}")
        print("instantiating segmenter")
    segmenter = instantiate(segmenter)
    for key in apply_to:
        if verbose:
            print(f"\nPreprocessing column: {key}")
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = apply(
                segmenter.segment_article,
                df[key],
                description=f"Splitting column: {key}",
                verbose=verbose,
                use_batcher=use_batcher,
                minibatch_size=minibatch_size,
            )
            if verbose:
                msg.good("\n >> elapsed time to segment: {}\n".format(elapsed()))
    return df


def tokenize(df, args):
    verbose = args.get("verbose", False)
    use_batcher = args.get("use_batcher", True)
    minibatch_size = args.get("minibatch_size", None)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    tokenizer = args.get("preprocessor", {}).get("tokenizer", None)
    if tokenizer is None:
        if verbose:
            print("No tokenizer specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        print(f"Tokenizing text: {args}")
        print("instantiating tokenizer")
    tokenizer = instantiate(tokenizer)
    for key in apply_to:
        if verbose:
            print(f"\nPreprocessing column: {key}")
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = apply(
                tokenizer.tokenize_article,
                df[key],
                description=f"Tokenizing column: {key}",
                verbose=verbose,
                use_batcher=use_batcher,
                minibatch_size=minibatch_size,
            )
            if verbose:
                msg.good("\n >> elapsed time to segment: {}\n".format(elapsed()))
    return df


def extract_tokens(df, args):
    verbose = args.get("verbose", False)
    use_batcher = args.get("use_batcher", True)
    nouns_ony = args.get("nouns_ony", True)
    filter_stopwords_only = args.get("filter_stopwords_only", False)
    minibatch_size = args.get("minibatch_size", None)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    tokenizer = args.get("preprocessor", {}).get("tokenizer", None)
    if tokenizer is None:
        if verbose:
            print("No tokenizer specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        print(f"Extracting tokens: {args}")
        print("instantiating tokenizer")
    tokenizer = instantiate(tokenizer)
    if filter_stopwords_only:
        extract_func = tokenizer.filter_article_stopwords
    elif nouns_ony:
        extract_func = tokenizer.extract_nouns
    else:
        extract_func = tokenizer.extract_tokens

    for key in apply_to:
        if verbose:
            print(f"\nPreprocessing column: {key}")
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = apply(
                extract_func,
                df[key],
                description=f"Extracting column: {key}",
                verbose=verbose,
                use_batcher=use_batcher,
                minibatch_size=minibatch_size,
            )
            if verbose:
                msg.good("\n >> elapsed time to segment: {}\n".format(elapsed()))
    return df


def chunk(df, args):
    verbose = args.get("verbose", False)
    use_batcher = args.get("use_batcher", True)
    minibatch_size = args.get("minibatch_size", None)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    segmenter = args.get("preprocessor", {}).get("segmenter", None)
    if segmenter is None:
        if verbose:
            print("No segmenter specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        print(f"Chunking text: {args}")
        print("instantiating segmenter")
    segmenter = instantiate(segmenter)
    for key in apply_to:
        if verbose:
            print(f"\nPreprocessing column: {key}")
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = apply(
                segmenter.chunk_article,
                df[key],
                description=f"Chunking column: {key}",
                verbose=verbose,
                use_batcher=use_batcher,
                minibatch_size=minibatch_size,
            )
            if verbose:
                msg.good("\n >> elapsed time to segment: {}\n".format(elapsed()))
    return df


def replace_whitespace(df, args):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    replace_with = args.get("replace_with", " ")
    if verbose:
        print(f"Replacing whitespace with [{replace_with}]")
    for key in apply_to:
        if verbose:
            print(f"\nPreprocessing column: {key}")
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = df[key].str.replace(r"\s+", replace_with)
            if verbose:
                msg.good(
                    "\n >> elapsed time to replace whitespace: {}\n".format(elapsed())
                )
    return df


def replace_regex(df, args):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    patterns = args.get("patterns", {})
    if patterns is None:
        if verbose:
            print("No patterns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        print(f"Replacing regex: {args}")
    for key in apply_to:
        if verbose:
            print(f"\nPreprocessing column: {key}")
        with elapsed_timer(format_time=True) as elapsed:
            for pat, repl in patterns.items():
                df[key] = df[key].str.replace(pat, repl, regex=True).str.strip()
            if verbose:
                msg.good("\n >> elapsed time to replace regex: {}\n".format(elapsed()))
    return df


def remove_startswith(df, args):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    startswith = args.get("startswith", {})
    if startswith is None:
        if verbose:
            print("No startswith text specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        print(f"Remove startswith: {args}")
    for key in apply_to:
        with elapsed_timer(format_time=True) as elapsed:
            for starting_text in startswith:
                print(f"Remove text starting with {starting_text} from [{key}]")
                idx = df[key].str.lower().str.startswith(starting_text, na=False)
                start_pos = len(starting_text)
                df.loc[idx, key] = df.loc[idx, key].str[start_pos:].str.strip()
            if verbose:
                msg.good(
                    "\n >> elapsed time to remove startswith: {}\n".format(elapsed())
                )
    return df


def filter_length(df, args, **kwargs):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    min_length = args.get("min_length", None)
    max_length = args.get("max_length", None)
    if min_length is None and max_length is None:
        if verbose:
            print("No length specified")
        return df
    len_func = args["method"].get("len_bytes", None)
    len_func = instantiate(len_func, _recursive_=False)
    _check_max_len = partial(check_max_len, max_len=max_length, len_func=len_func)
    _check_min_len = partial(check_min_len, min_len=min_length, len_func=len_func)

    if verbose:
        print(f"Filtering by length: {args}")
    for key in apply_to:
        if verbose:
            print(f"\nPreprocessing column: {key}")
        with elapsed_timer(format_time=True) as elapsed:
            if min_length and min_length > 0:
                n_docs = df.shape[0]
                idx = apply(
                    _check_min_len,
                    df[key].astype(str),
                    verbose=verbose,
                    description=f"min length: {min_length}",
                )
                df = df[idx]
                if verbose:
                    print(
                        f"{(n_docs-df.shape[0])} documents removed due to length is less than {min_length}"
                    )
            if max_length and max_length > 0:
                n_docs = df.shape[0]
                idx = apply(
                    _check_max_len,
                    df[key].astype(str),
                    verbose=verbose,
                    description=f"max length: {max_length}",
                )
                df = df[idx]
                if verbose:
                    print(
                        f"{(n_docs-df.shape[0])} documents removed due to length is greater than {max_length}"
                    )
            if verbose:
                msg.good("\n >> elapsed time to filter length: {}\n".format(elapsed()))
    return df


def filter_query(df, args):
    verbose = args.get("verbose", False)
    query = args.get("query", None)
    if query is None:
        if verbose:
            print("No query specified")
        return df
    if isinstance(query, str):
        query = [query]

    if verbose:
        print(f"Filtering by qeury: {args}")
    with elapsed_timer(format_time=True) as elapsed:
        for qry in query:
            if verbose:
                print(f"\nPreprocessing query: {qry}")
            n_docs = df.shape[0]
            df = df.query(qry, engine="python")
            if verbose:
                print(f"filtered {df.shape[0]} out of {n_docs} documents by {qry}")
        if verbose:
            msg.good("\n >> elapsed time to filter query: {}\n".format(elapsed()))
    return df


def drop_duplicates(df, args):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", None)
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        print(f"Dropping duplicates: {args}")
    with elapsed_timer(format_time=True) as elapsed:
        for key in apply_to:
            num_docs = df.shape[0]
            df = df.drop_duplicates(subset=[key])
            n_docs = df.shape[0]
            if verbose:
                print(
                    f"{n_docs} documents after dropping {(num_docs-n_docs)} duplicates from [{key}]"
                )
        if verbose:
            msg.good("\n >> elapsed time to drop duplicates: {}\n".format(elapsed()))
    return df


def save_samples(df, args):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    sample_length_to_print = args.get("sample_length_to_print", 1000)
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    num_samples_to_save = args.get("num_samples_to_save", None)
    smaple_file_prefix = args.get("sample_file_prefix", "sample")
    if verbose:
        print(f"Saving samples: {args}")

    sample_separator = "-" * 100 + "\n"
    df_sample = df.sample(num_samples_to_save)[apply_to]
    sample_text = ""
    print_text = ""
    for i, row in df_sample.iterrows():
        for key in apply_to:
            stext = row[key]
            if len(stext) > sample_length_to_print:
                ptext = stext[:sample_length_to_print] + "..."
            else:
                ptext = stext
            sample_text += key + ": \n" + stext + "\n\n"
            print_text += key + ": \n" + ptext + "\n\n"
        sample_text += sample_separator
        print_text += sample_separator
    sample_text = sample_text.strip()
    print_text = print_text.strip()
    sample_file = smaple_file_prefix + ".txt"
    open(sample_file, "w", encoding="utf-8").write(sample_text)

    if verbose:
        print(sample_separator)
        print(print_text)
        print(f"Saved {num_samples_to_save} samples to {sample_file}")

    return df


def stdout_samples(df, args):
    from contextlib import redirect_stdout

    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    sample_length_to_print = args.get("sample_length_to_print", 1000)
    if apply_to is None:
        if verbose:
            print("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    num_samples = args.get("num_samples", None)
    output_dir = args.get("output_dir", None)
    if output_dir is None:
        output_dir = "."
    output_file = args.get("output_file", None)
    head = args.get("head", None)
    tail = args.get("tail", None)
    if head is None:
        head = 5
    if tail is None:
        tail = 5

    if verbose:
        print(f"Print samples: {args}")

    sample_separator = "-" * 100 + "\n"
    df_sample = df.sample(num_samples)[apply_to]

    print_text = ""
    for i, row in df_sample.iterrows():
        for key in apply_to:
            ptext = row[key]
            print_text += key + ": \n" + ptext + "\n\n"
        print_text += sample_separator
    print_text = print_text.strip()

    if output_file is not None:
        output_file = os.path.join(output_dir, output_file)
        with open(output_file, "w", encoding="utf-8") as f:
            with redirect_stdout(f):
                print(print_text)
                print(sample_separator)
                print(df.head(head))
                print(sample_separator)
                print(df.tail(tail))
    print(print_text)
    print(sample_separator)
    print(df.head(head))
    print(sample_separator)
    print(df.tail(tail))

    if verbose:
        print(f"Saved {num_samples} samples to {output_file}")

    return df


def save_as_text(df, args):
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    corpus_name = args.get("corpus_name", "corpus")
    output_dir = args.get("output_dir", ".")
    output_file = args.get("output_file", None)
    doc_separator = args.get("doc_separator", "\n\n")
    if isinstance(apply_to, (list, ListConfig)):
        apply_to = apply_to[0]

    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    if output_file:
        filename = output_file
    else:
        filename = f"{corpus_name}.txt"
    output_file_path = f"{output_dir}/{filename}"

    with elapsed_timer(format_time=True) as elapsed:
        n_loaded = len(df.index)
        doc_separator = str(doc_separator).encode("utf-8").decode("unicode_escape")
        with open(output_file_path, "w") as fo:
            fo.write(doc_separator.join(df[apply_to].dropna().tolist()))
        print(f"Corpus is exported to {output_file_path}")
        n_sampled = len(df.index)
        status = [[" x ", corpus_name, n_loaded, n_sampled, elapsed(), filename]]

    if verbose:
        print_status(status)
    return df


def split_dataframe(df, args):
    verbose = args.get("verbose", False)
    num_splits = args.get("num_splits", 1)
    if num_splits <= 1:
        return df
    if verbose:
        print(f"Splitting dataframe into {num_splits} splits")
    return np.array_split(df, num_splits)


def concat_dataframes(dfs, args):
    verbose = args.get("verbose", False)
    if isinstance(dfs, list):
        if verbose:
            print(f"Concatenating {len(dfs)} dataframes")
        return pd.concat(dfs)
    else:
        if verbose:
            print("Returning original dataframe")
        return dfs


def merge_dataframe(df=None, args=None):
    if args is None:
        raise ValueError("args must be specified")
    verbose = args.get("verbose", False)
    filepath = args.get("filepath", None)
    data_dir = args.get("data_dir", None)
    data_file = args.get("data_file", None)
    how = args.get("how", "inner")
    merge_on = args.get("merge_on", None)
    left_on = args.get("left_on", None)
    right_on = args.get("right_on", None)
    if merge_on is None and (right_on is None or left_on is None):
        raise ValueError("merge_on or (left_on and right_on) must be specified")
    if isinstance(merge_on, str):
        merge_on = [merge_on]
    elif isinstance(merge_on, ListConfig):
        merge_on = list(merge_on)
    if isinstance(left_on, str):
        left_on = [left_on]
    elif isinstance(left_on, ListConfig):
        left_on = list(left_on)
    if isinstance(right_on, str):
        right_on = [right_on]
    elif isinstance(right_on, ListConfig):
        right_on = list(right_on)

    if filepath:
        filepaths = get_filepaths(filepath)
    else:
        filepaths = get_filepaths(data_file, data_dir)
    if verbose:
        print(f"Loading {len(filepaths)} dataframes from {filepaths}")
    if len(filepaths) == 1:
        df_to_merge = load_dataframe(filepaths[0], verbose=verbose)
    else:
        df_to_merge = pd.concat([load_dataframe(f, verbose=verbose) for f in filepaths])
    if merge_on:
        df = df.merge(df_to_merge, how=how, on=merge_on)
    else:
        df = df.merge(df_to_merge, how=how, left_on=left_on, right_on=right_on)
    if verbose:
        print(df.tail())
    return df


def save_metadata(df, args):
    verbose = args.get("verbose", False)
    filepath = args.get("filepath", None)
    filetype = args.get("filetype", None)
    column_info = args.get("column_info", None)
    split_name = args.get("split_name", None)

    if verbose:
        print(f"Saving metadata: {args}")

    meta_info = column_info.get("meta", None)
    if isinstance(meta_info, (dict, DictConfig)):
        meta_columns = list(meta_info.keys())
        if "split" in meta_columns and "split" not in df.columns:
            df["split"] = split_name
        df_meta = df[meta_columns]
        save_dataframe(df_meta, filepath, filetype, verbose)

    data_info = column_info.get("data", None)
    if isinstance(data_info, (dict, DictConfig)):
        data_columns = list(data_info.keys())
        if "split" in data_columns and "split" not in df.columns:
            df["split"] = split_name
        df = df[data_columns]

    return df


def save_dataframe_pipe(df, args):
    verbose = args.get("verbose", False)
    filepath = args.get("filepath", None)
    filetype = args.get("filetype", None)
    corpus_name = args.get("corpus_name", "corpus")
    output_dir = args.get("output_dir", ".")
    output_file = args.get("output_file", None)
    dataframe_no = args.get("dataframe_no", None)
    columns_to_keep = args.get("columns_to_keep", None)

    if df is None:
        msg.warn("Dataframe is None")
        return df
    if verbose:
        print(f"Saving dataframe: {args}")

    if filepath:
        output_dir = os.path.dirname(filepath)
        output_file = os.path.basename(filepath)
    if output_file:
        fileinfo = os.path.splitext(output_file)
        filename = fileinfo[0]
        if not filetype:
            filetype = fileinfo[1] if len(fileinfo) > 1 else "csv"
    else:
        filename = f"{corpus_name}"
        if not filetype:
            filetype = "csv"
    filetype = "." + filetype.replace(".", "")
    if dataframe_no is not None:
        filename = f"{filename}-{dataframe_no:0>3d}{filetype}"
    else:
        filename = f"{filename}{filetype}"
    filepath = f"{output_dir}/{filename}"

    save_dataframe(df, filepath, filetype, verbose, columns_to_keep=columns_to_keep)
    return df


def load_dataframe_pipe(df=None, args=None):
    if args is None:
        raise ValueError("args must be specified")
    verbose = args.get("verbose", False)
    filepath = args.get("filepath", None)
    data_dir = args.get("data_dir", None)
    data_file = args.get("data_file", None)
    dtype = args.get("dtype", None)
    if isinstance(dtype, DictConfig):
        dtype = dict(dtype)
    elif isinstance(dtype, (list, ListConfig)):
        dtype = {k: "str" for k in dtype}
    parse_dates = args.get("parse_dates", False)
    if isinstance(parse_dates, DictConfig):
        parse_dates = dict(parse_dates)
    elif isinstance(parse_dates, ListConfig):
        parse_dates = list(parse_dates)

    if filepath:
        filepaths = get_filepaths(filepath)
    else:
        filepaths = get_filepaths(data_file, data_dir)
    if verbose:
        print(f"Loading {len(filepaths)} dataframes from {filepaths}")
    if len(filepaths) == 1:
        return load_dataframe(
            filepaths[0], verbose=verbose, dtype=dtype, parse_dates=parse_dates
        )
    else:
        df = pd.concat(
            [
                load_dataframe(f, verbose=verbose, dtype=dtype, parse_dates=parse_dates)
                for f in filepaths
            ]
        )
        return df


def save_as_json(df, args):
    verbose = args.get("verbose", False)
    corpus_name = args.get("corpus_name", "corpus")
    output_dir = args.get("output_dir", ".")
    output_file = args.get("output_file", None)
    force_ascii = args.get("force_ascii", False)

    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    if output_file:
        filename = output_file
    else:
        filename = f"{corpus_name}.json"
    output_file_path = f"{output_dir}/{filename}"

    df.to_json(output_file_path, orient="records", lines=True, force_ascii=force_ascii)
    if verbose:
        print(f"Corpus is exported to {output_file_path}")
    return df


def process_dataframe(**cfg):
    args = OmegaConf.create(cfg)
    verbose = args.get("verbose", False)
    process_pipeline = args.get("_pipeline_", [])
    if process_pipeline is None:
        process_pipeline = []
    df = None
    if len(process_pipeline) > 0:
        df = apply_pipeline(None, process_pipeline, args)
        if df is not None:
            if isinstance(df, list):
                df = pd.concat(df)
            if verbose:
                print(df.tail())
        else:
            print("No dataframe returned")

    return df
