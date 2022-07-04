import logging
import codecs
import os
import numpy as np
import pandas as pd
from collections import OrderedDict
from functools import reduce
from ekorpkit.utils import print_status
from ekorpkit.utils.func import elapsed_timer
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def apply_pipeline(df, pipeline, pipeline_args, update_args={}, verbose=True):
    pipeline = eKonf.to_dict(pipeline)
    pipeline_args = eKonf.to_dict(pipeline_args)

    pipeline_targets = []
    pipes = OrderedDict()
    if isinstance(pipeline, list):
        for pipe in pipeline:
            pipes[pipe] = pipe
    elif isinstance(pipeline, str):
        pipes[pipeline] = pipeline
    else:
        pipes = pipeline

    if pipes is None or len(pipes) == 0:
        log.warning("No pipeline specified")
        return df

    log.info(f"Applying pipeline: {pipes}")
    for pipe, pipe_arg_name in pipes.items():
        args = pipeline_args.get(pipe_arg_name, {}).copy()
        if pipe != pipe_arg_name:
            args_override = pipeline_args.get(pipe, {})
            args.update(args_override)
        if args and isinstance(args, dict):
            args.update(update_args)
            pipeline_targets.append(args)

    return reduce(eKonf.pipe, pipeline_targets, df)


def split_column(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    source = args.get("source")
    target = args.get("target")
    _split = args.get(eKonf.Keys.SPLIT)
    if source is None:
        log.warning("No source specified")
        return df
    log.info(f"Split column: {args}")
    df[target] = df[source].str.split(**_split)

    if verbose:
        print(df.tail())
    return df


def eval_columns(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    expressions = args.get("expressions", None)
    engine = args.get("engine", None)
    eval_at = args.get("eval_at", "dataframe")
    if expressions is None:
        log.warning("No expressions specified")
        return df
    log.info(f"Eval columns: {args}")
    if eval_at == "dataframe":
        if isinstance(expressions, list):
            for expr in expressions:
                df.eval(expr, engine=engine, inplace=True)
        else:
            for col, expr in expressions.items():
                df[col] = df.eval(expr, engine=engine)
    elif eval_at == "python":
        if isinstance(expressions, dict):
            for col, expr in expressions.items():
                df[col] = eval(expr)
        else:
            log.warning("Expressions must be a dict to eval at python")
    else:
        if isinstance(expressions, list):
            for expr in expressions:
                pd.eval(expr, engine=engine, inplace=True, target=df)
        else:
            for col, expr in expressions.items():
                df[col] = pd.eval(expr, engine=engine)

    if verbose:
        print(df.tail())
    return df


def combine_columns(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    columns = args.get("columns", None)
    if columns is None:
        log.warning("No columns specified")
        return df
    log.info(f"Combining columns: {args}")
    for col in columns:
        df[col].fillna("", inplace=True)
        df[col] = df[col].astype(str)
    separator = codecs.decode(args["separator"], "unicode_escape")
    df[args["into"]] = df[columns].agg(separator.join, axis=1)
    return df


def drop(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    labels = args.get("labels", None)
    axis = args.get("axis", 1)
    columns = args.get("columns", None)
    index = args.get("index", None)
    level = args.get("level", None)
    errors = args.get("errors", "ignore")
    if verbose:
        log.info(f"droping: {args}")
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
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    id_vars = args.get("id_vars", None)
    value_vars = args.get("value_vars", None)
    var_name = args.get("var_name", None)
    value_name = args.get("value_name", None)
    col_level = args.get("col_level", None)
    ignore_index = args.get("ignore_index", True)
    if id_vars is None:
        if verbose:
            log.warning("No id_vars specified")
        return df
    if verbose:
        log.info(f"Melting columns: {args}")
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
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    plot_cfg = args.get("visualize", {}).get("plot", {})
    if "_target_" not in plot_cfg:
        log.warning("No target specified")
        return df
    if verbose:
        log.info(f"Plotting: {plot_cfg}")
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
                log.info(f"Plotting subset: {col} == {val}")
            df_sub = df[df[col] == val]
            if output_file and "{}" in output_file:
                output_path = output_file.replace("{}", str(val))
                output_path = f"{output_dir}/{output_path}"
                plot_cfg["savefig"]["fname"] = output_path
            if titles and len(titles) > i:
                plot_cfg["figure"]["title"] = title_prefix + titles[i]
            if verbose:
                log.info(f"Plotting: {plot_cfg}")
                # print(df_sub.head())
            eKonf.instantiate(plot_cfg, df=df_sub)
    else:
        eKonf.instantiate(plot_cfg, df=df)

    return df


def pivot(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    index = args.get("index", None)
    columns = args.get("columns", None)
    values = args.get("values", None)
    fillna = args.get("fillna", None)
    reset_index = args.get("reset_index", True)
    if index is None:
        log.warning("No index specified")
        return df
    if columns is None:
        log.warning("No columns specified")
        return df
    if values is None:
        log.warning("No values specified")
        return df
    if verbose:
        log.info(f"Pivoting columns: {args}")
    if len(values) == 1:
        values = values[0]
    df = df.pivot(index=index, columns=columns, values=values)
    # df = df.set_index(index)[values].unstack()
    if fillna is not None:
        df.fillna(fillna, inplace=True)
    if reset_index:
        if verbose:
            log.info(f"Resetting index, nlevels of columns: {df.columns.nlevels}")
        df.reset_index(inplace=True)
        if df.columns.nlevels > 1:
            df.columns = ["_".join(a).strip("_") for a in df.columns.to_flat_index()]
    if verbose:
        print(df.head())
    return df


def split_sampling(df, args):
    from sklearn.model_selection import train_test_split

    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    stratify_on = args.get("stratify_on", None)
    random_state = args.get("random_state", 123)
    shuffle = args.get("shuffle", True)
    groupby = args.get("groupby", stratify_on)
    unique_key = args.get("unique_key", "id")
    test_size = args.get("test_size", 0.1)
    dev_size = args.get("dev_size", None)
    if verbose:
        log.info(f"Split sampling: {args}")

    train, dev, test = None, None, None
    if stratify_on is None:
        train, test = train_test_split(
            df, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        if dev_size:
            train, dev = train_test_split(
                train, test_size=dev_size, random_state=random_state, shuffle=shuffle
            )
    else:
        train, test = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[stratify_on],
            shuffle=shuffle,
        )
        if dev_size:
            train, dev = train_test_split(
                train,
                test_size=dev_size,
                random_state=random_state,
                stratify=train[stratify_on],
                shuffle=shuffle,
            )
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    if dev_size:
        dev.reset_index(drop=True, inplace=True)
    if verbose:
        log.info(f"Total rows: {len(df)}")
        log.info(f"Train: {len(train)}")
        log.info(f"Test: {len(test)}")
        if dev_size:
            log.info(f"Dev: {len(dev)}")

        if groupby in df.columns and unique_key in df.columns:
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
        eKonf.save_data(train, train_file, base_dir=output_dir, verbose=verbose)
    if test_file and test is not None:
        eKonf.save_data(test, test_file, base_dir=output_dir, verbose=verbose)
    if dev_file and dev is not None:
        eKonf.save_data(dev, dev_file, base_dir=output_dir, verbose=verbose)

    return df


def top_values(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    groupby = args.get("groupby", None)
    value_var = args.get("value_var", None)
    value_label = args.get("value_label", None)
    value_name = args.get("value_name", None)
    value_separator = args.get("value_separator", ", ")
    top_n = args.get("top_n", 5)
    columns = args.get("columns", None)
    use_batcher = args.get("use_batcher", True)
    minibatch_size = args.get("minibatch_size", None)

    if verbose:
        log.info(f"Split sampling: {args}")

    def label(row):
        return f"{row[value_label]}[{round(row[value_var]*100,0):.0f}%]"

    if value_label is not None and value_name is not None:
        # df[value_name] = df.apply(label, axis=1)
        df[value_name] = eKonf.apply(
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
    if columns:
        df = df[columns:]
    if verbose:
        print(df.head())

    return df


def subset(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    random_state = args.get("random_state", 123)
    head_n = args.get("head_n", None)
    tail_n = args.get("tail_n", None)
    sample_n = args.get("sample_n", None)
    sample_frac = args.get("sample_frac", None)
    if not (head_n or tail_n or sample_n or sample_frac):
        log.error("Must specify one of head_n, tail_n, sample_n, sample_frac")
        return df

    if verbose:
        log.info(f"Subsetting: {args}")
    dfs = []
    if head_n:
        dfs.append(df.head(head_n))
    if tail_n:
        dfs.append(df.tail(tail_n))
    if sample_n:
        dfs.append(df.sample(sample_n, random_state=random_state))
    if sample_frac:
        dfs.append(df.sample(frac=sample_frac, random_state=random_state))
    if verbose:
        log.info(f"Total rows: {len(df)}")

    if len(dfs) > 0:
        df = pd.concat(dfs)

    if verbose:
        log.info(f"Subset rows: {len(df)}")

    return df


def sampling(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    random_state = args.get("random_state", 123)
    groupby = args.get("groupby", None)
    sample_size_per_group = args.get("sample_size_per_group", None)
    value_var = args.get("value_var", None)
    columns = args.get("columns", None)

    if verbose:
        log.info(f"Split sampling: {args}")

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
    if columns:
        df_sample = df_sample[columns:]
    if verbose:
        log.info(f"Total rows: {len(df)}")
        log.info(f"Sample: {len(df_sample)}")
        print(df_sample.head())

    if groupby is not None and verbose:
        grp_all = df.groupby(groupby)[value_var].count().rename("population")
        grp_sample = df_sample.groupby(groupby)[value_var].count().rename("sample")
        grp_dists = pd.concat([grp_all, grp_sample], axis=1)
        print(grp_dists)

    save_dataframe(df_sample, args)

    return df


def aggregate_columns(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    onto_column = args.get("onto", None)
    aggregations = args.get("aggregations", None)
    reset_index = args.get("reset_index", False)
    if onto_column is None and aggregations is None:
        if verbose:
            log.warning("No columns or aggregations are specified")
        return df
    groupby_cloumns = args["groupby"]
    if groupby_cloumns is None:
        if verbose:
            log.warning("No groupby specified")
        return df
    separator = codecs.decode(args["separator"], "unicode_escape")

    num_docs = df.shape[0]
    if verbose:
        log.info(f"Aggregating columns: {args}")
    if aggregations:
        df = df.groupby(groupby_cloumns, as_index=False).agg(aggregations)
    else:
        df[onto_column].fillna("", inplace=True)
        df = df.groupby(groupby_cloumns, as_index=False).agg(
            {onto_column: separator.join}
        )
    if reset_index:
        if verbose:
            log.info(f"Resetting index, nlevels of columns: {df.columns.nlevels}")
        df.reset_index(inplace=True)
        if df.columns.nlevels > 1:
            df.columns = ["_".join(a).strip("_") for a in df.columns.to_flat_index()]
    n_docs = df.shape[0]
    if verbose:
        print(df.tail())
        log.info(f"{num_docs} documents aggregated into {n_docs} documents")
    return df


def explode_splits(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            log.warning("No columns specified")
        return df
    separator = codecs.decode(args["separator"], "unicode_escape")
    id_key = args.get("id_key", "id")
    split_key = args.get("split_key", "seg_id")
    if verbose:
        log.info(f"Exploding column: {args}")

    num_docs = df.shape[0]
    df[apply_to] = df[apply_to].str.split(separator)
    df = df.explode(apply_to)
    df[split_key] = df.groupby(id_key).cumcount()
    n_docs = df.shape[0]
    if verbose:
        log.info(f"{num_docs} documents exploded into {n_docs} documents")
    return df


def rename_columns(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    new_names = args.get("new_names", None)
    if new_names is None:
        if verbose:
            log.warning("No columns specified")
        return df
    if verbose:
        log.info(f"Renaming columns: {args}")
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
        log.info(f"Resetting index: {args}")
    df = df.reset_index(drop=drop_index)
    if not drop_index and index_column_name != "index":
        df.rename(columns={"index": index_column_name}, inplace=True)
    if verbose:
        print(df.head())
    return df


def normalize(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        log.warning("No columns specified")
        return df
    normalizer = args.get("preprocessor", {}).get("normalizer", None)
    if normalizer is None:
        log.warning("No normalizer specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    log.info("instantiating normalizer")
    normalizer = eKonf.instantiate(normalizer)
    for key in apply_to:
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = eKonf.apply(
                normalizer.normalize,
                df[key],
                description=f"Normalizing column: {key}",
                verbose=verbose,
            )
            log.info(" >> elapsed time to normalize: {}".format(elapsed()))
    return df


def fillna(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            log.warning("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    fill_with = args.get("fill_with", "")
    if verbose:
        log.info(f"Filling missing values: {args}")
    for key in apply_to:
        df[key].fillna(fill_with, inplace=True)
    return df


def predict(df, args):
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        log.warning("No columns specified")
        return df
    if isinstance(apply_to, list):
        apply_to = apply_to[0]
    data_columns = args["data_columns"]
    if data_columns:
        df = df.copy()[data_columns]

    model = args.get(eKonf.Keys.MODEL)
    if model is None:
        log.warning("No model specified")
        return df
    model[eKonf.Keys.PREDICT][eKonf.Keys.INPUT] = apply_to
    model = eKonf.instantiate(model)

    df = model.predict(df)

    _save_data(df, args)

    return df


def aggregate_scores(df, args):
    groupby = args["groupby"]
    feature = args["feature"]
    min_examples = args["min_examples"]
    _method_ = args[eKonf.Keys.METHOD]
    if groupby is None:
        log.warning("No groupby columns specified")
        return df

    model = args[eKonf.Keys.MODEL].get("sentiment")
    if model is None:
        log.warning("No model specified")
        return df
    model = eKonf.instantiate(model)

    df = model.aggregate_scores(df, groupby, feature, min_examples, _method_=_method_)

    _save_data(df, args)

    return df


def _save_data(df, args):
    if df is None:
        log.info("No data to save")
        return
    _path = args[eKonf.Keys.PATH][eKonf.Keys.OUTPUT]
    _path[eKonf.Keys.SUFFIX.value] = args.get(eKonf.Keys.SUFFIX)
    if _path[eKonf.Keys.FILENAME]:
        log.info(f"Saving data to: {_path}")
        eKonf.save_data(df, **_path)
    else:
        log.info("filename not specified")


def segment(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    num_workers = args.get("num_workers", 1)
    use_batcher = args.get("use_batcher", True)
    minibatch_size = args.get("minibatch_size", None)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            log.warning("No columns specified")
        return df
    segmenter = args.get("preprocessor", {}).get("segmenter", None)
    if segmenter is None:
        log.warning("No segmenter specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    log.info("instantiating segmenter")
    segmenter = eKonf.instantiate(segmenter)
    for key in apply_to:
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = eKonf.apply(
                segmenter.segment_article,
                df[key],
                description=f"Splitting column: {key}",
                verbose=verbose,
                use_batcher=use_batcher,
                minibatch_size=minibatch_size,
                num_workers=num_workers,
            )
            log.info(" >> elapsed time to segment: {}".format(elapsed()))
    return df


def tokenize(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    use_batcher = args.get("use_batcher", True)
    minibatch_size = args.get("minibatch_size", None)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        log.warning("No columns specified")
        return df
    tokenizer = args.get("preprocessor", {}).get("tokenizer", None)
    if tokenizer is None:
        log.warning("No tokenizer specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    log.info("instantiating tokenizer")
    tokenizer = eKonf.instantiate(tokenizer)
    for key in apply_to:
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = eKonf.apply(
                tokenizer.tokenize_article,
                df[key],
                description=f"Tokenizing column: {key}",
                verbose=verbose,
                use_batcher=use_batcher,
                minibatch_size=minibatch_size,
            )
            log.info(" >> elapsed time to segment: {}".format(elapsed()))
    return df


def extract_tokens(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    use_batcher = args.get("use_batcher", True)
    nouns_only = args.get("nouns_only", True)
    filter_stopwords_only = args.get("filter_stopwords_only", False)
    minibatch_size = args.get("minibatch_size", None)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        log.warning("No columns specified")
        return df
    tokenizer = args.get("preprocessor", {}).get("tokenizer", None)
    if tokenizer is None:
        log.warning("No tokenizer specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    log.info("instantiating tokenizer")
    tokenizer = eKonf.instantiate(tokenizer)
    if filter_stopwords_only:
        extract_func = tokenizer.filter_article_stopwords
    elif nouns_only:
        extract_func = tokenizer.extract_nouns
    else:
        extract_func = tokenizer.extract_tokens
    log.info(f"extract_func: {extract_func}")

    for key in apply_to:
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = eKonf.apply(
                extract_func,
                df[key],
                description=f"Extracting column: {key}",
                verbose=verbose,
                use_batcher=use_batcher,
                minibatch_size=minibatch_size,
            )
            log.info(" >> elapsed time to extract tokens: {}".format(elapsed()))
    return df


def chunk(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    use_batcher = args.get("use_batcher", True)
    minibatch_size = args.get("minibatch_size", None)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        log.warning("No columns specified")
        return df
    segmenter = args.get("preprocessor", {}).get("segmenter", None)
    if segmenter is None:
        log.warning("No segmenter specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    log.info("instantiating segmenter")
    segmenter = eKonf.instantiate(segmenter)
    for key in apply_to:
        with elapsed_timer(format_time=True) as elapsed:
            df[key] = eKonf.apply(
                segmenter.chunk_article,
                df[key],
                description=f"Chunking column: {key}",
                verbose=verbose,
                use_batcher=use_batcher,
                minibatch_size=minibatch_size,
            )
            log.info(" >> elapsed time to segment: {}".format(elapsed()))
    return df


def general_function(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", None)
    method = args.get(eKonf.Keys.METHOD, None)
    if verbose:
        log.info(f"Running dataframe function: {args}")

    with elapsed_timer(format_time=True) as elapsed:
        if apply_to is None:
            df = getattr(df, method[eKonf.Keys.METHOD_NAME])(
                **method[eKonf.Keys.rcPARAMS]
            )
        else:
            if isinstance(apply_to, str):
                apply_to = [apply_to]
            for key in apply_to:
                log.info(f"processing column: {key}")
                df[key] = getattr(df[key], method[eKonf.Keys.METHOD_NAME])(
                    **method[eKonf.Keys.rcPARAMS]
                )

        log.info(" >> elapsed time to replace: {}".format(elapsed()))
        if verbose:
            print(df.head())
    return df


def replace_regex(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            log.warning("No columns specified")
        return df
    patterns = args.get("patterns", {})
    if patterns is None:
        if verbose:
            log.warning("No patterns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        log.info(f"Replacing regex: {args}")
    for key in apply_to:
        with elapsed_timer(format_time=True) as elapsed:
            for pat, repl in patterns.items():
                df[key] = df[key].str.replace(pat, repl, regex=True).str.strip()
            if verbose:
                log.info(" >> elapsed time to replace regex: {}".format(elapsed()))
    return df


def remove_startswith(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        if verbose:
            log.warning("No columns specified")
        return df
    startswith = args.get("startswith", {})
    if startswith is None:
        if verbose:
            log.warning("No startswith text specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        log.info(f"Remove startswith: {args}")
    for key in apply_to:
        with elapsed_timer(format_time=True) as elapsed:
            for starting_text in startswith:
                log.info(f"Remove text starting with {starting_text} from [{key}]")
                idx = df[key].str.lower().str.startswith(starting_text, na=False)
                start_pos = len(starting_text)
                df.loc[idx, key] = df.loc[idx, key].str[start_pos:].str.strip()
            if verbose:
                log.info(
                    " >> elapsed time to remove startswith: {}\n".format(elapsed())
                )
    return df


def filter_length(df, args, **kwargs):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    if apply_to is None:
        log.warning("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    min_length = args.get("min_length", None)
    max_length = args.get("max_length", None)
    if min_length is None and max_length is None:
        log.warning("No length specified")
        return df
    add_len_column = args.get("add_len_column", False)
    len_column = args.get("len_column", "num_bytes")
    func_name = args.get("len_func", "len_bytes")
    len_func = args[eKonf.Keys.FUNC].get(func_name, None)
    len_func = eKonf.instantiate(len_func)

    df = df.copy()
    if verbose:
        log.info(f"Filtering by length: {args}")
    for key in apply_to:
        with elapsed_timer(format_time=True) as elapsed:
            _len_column = f"{key}_{len_column}"
            df[_len_column] = eKonf.apply(
                len_func, df[key], description=f"Calculating length"
            )
            if min_length and min_length > 0:
                n_docs = df.shape[0]
                df = df.loc[df[_len_column] >= min_length]
                log.info(
                    f"removed {(n_docs-df.shape[0])} of {n_docs} documents with length < {min_length}"
                )
            if max_length and max_length > 0:
                n_docs = df.shape[0]
                df = df.loc[df[_len_column] <= max_length]
                log.info(
                    f"removed {(n_docs-df.shape[0])} of {n_docs} documents with length > {max_length}"
                )
            log.info(" >> elapsed time to filter length: {}".format(elapsed()))
        if not add_len_column:
            df = df.drop(_len_column, axis=1)

    return df


def filter_query(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    query = args.get("query", None)
    if query is None:
        log.warning("No query specified")
        return df
    if isinstance(query, str):
        query = [query]

    if verbose:
        log.info(f"Filtering by qeury: {args}")
    with elapsed_timer(format_time=True) as elapsed:
        for qry in query:
            if verbose:
                log.info(f"Preprocessing query: {qry}")
            n_docs = df.shape[0]
            df = df.query(qry, engine="python")
            if verbose:
                log.info(f"filtered {df.shape[0]} out of {n_docs} documents by {qry}")
        if verbose:
            log.info(" >> elapsed time to filter query: {}".format(elapsed()))
    return df


def drop_duplicates(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", None)
    if apply_to is None:
        if verbose:
            log.warning("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    if verbose:
        log.info(f"Dropping duplicates: {args}")
    with elapsed_timer(format_time=True) as elapsed:
        num_docs = df.shape[0]
        df = df.drop_duplicates(subset=apply_to)
        n_docs = df.shape[0]
        if verbose:
            log.info(
                f"{n_docs} documents after dropping {(num_docs-n_docs)} duplicates from [{apply_to}]"
            )
            log.info(" >> elapsed time to drop duplicates: {}".format(elapsed()))
    return df


def save_samples(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")

    sample_length_to_print = args.get("sample_length_to_print", 1000)
    if apply_to is None:
        if verbose:
            log.warning("No columns specified")
        return df
    if isinstance(apply_to, str):
        apply_to = [apply_to]
    num_samples_to_save = args.get("num_samples_to_save", None)
    if verbose:
        log.info(f"Saving samples: {args}")

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

    _path = args[eKonf.Keys.PATH][eKonf.Keys.OUTPUT]
    sample_file = _path.get("filepath") or "sample.txt"
    open(sample_file, "w", encoding="utf-8").write(sample_text)

    if verbose:
        print(sample_separator)
        print(print_text)
        log.info(f"Saved {num_samples_to_save} samples to {sample_file}")

    return df


def stdout_samples(df, args):
    from contextlib import redirect_stdout

    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    sample_length_to_print = args.get("sample_length_to_print", 1000)
    if apply_to is None:
        if verbose:
            log.warning("No columns specified")
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
        log.info(f"Print samples: {args}")

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
        log.info(f"Saved {num_samples} samples to {output_file}")

    return df


def save_as_text(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    apply_to = args.get("apply_to", "text")
    corpus_name = args.get("corpus_name", eKonf.Keys.CORPUS)
    output_dir = args.get("output_dir", ".")
    output_file = args.get("output_file", None)
    doc_separator = args.get("doc_separator", "\n\n")
    if isinstance(apply_to, list):
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
        log.info(f"Corpus is exported to {output_file_path}")
        n_sampled = len(df.index)
        status = [[" x ", corpus_name, n_loaded, n_sampled, elapsed(), filename]]

    if verbose:
        print_status(status)
    return df


def split_dataframe(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    num_splits = args.get("num_splits", 1)
    if num_splits <= 1:
        return df
    if verbose:
        log.info(f"Splitting dataframe into {num_splits} splits")
    return np.array_split(df, num_splits)


def concat_dataframes(dataframes, args):
    args = eKonf.to_dict(args)

    return eKonf.concat_data(data=dataframes, **args)


def merge_dataframe(df=None, args=None):
    if args is None:
        raise ValueError("args must be specified")
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    how = args.get("how", "inner")
    merge_on = args.get("merge_on", None)
    left_on = args.get("left_on", None)
    right_on = args.get("right_on", None)
    if merge_on is None and (right_on is None or left_on is None):
        raise ValueError("merge_on or (left_on and right_on) must be specified")
    if isinstance(merge_on, str):
        merge_on = [merge_on]
    if isinstance(left_on, str):
        left_on = [left_on]
    if isinstance(right_on, str):
        right_on = [right_on]

    df_to_merge = load_dataframe(None, args)
    if merge_on:
        df = df.merge(df_to_merge, how=how, on=merge_on)
    else:
        df = df.merge(df_to_merge, how=how, left_on=left_on, right_on=right_on)
    if verbose:
        print(df.tail())
    return df


def save_metadata(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    _path = args[eKonf.Keys.PATH][eKonf.Keys.OUTPUT]
    # filetype = args.get("filetype", None)
    column_info = args.get("column_info", None)
    split_name = args.get("split_name", None)

    if verbose:
        log.info(f"Saving metadata: {args}")

    meta_info = column_info.get("meta", None)
    meta_columns = []
    if isinstance(meta_info, dict):
        meta_columns = list(meta_info.keys())
        if eKonf.Keys.SPLIT in meta_columns and eKonf.Keys.SPLIT not in df.columns:
            df[eKonf.Keys.SPLIT] = split_name
        df_meta = df[meta_columns]
        eKonf.save_data(df_meta, **_path)

    data_info = column_info.get("data", None)
    if isinstance(data_info, dict):
        data_keys = list(data_info.keys())
        data_columns = [
            col for col in df.columns if col not in meta_columns or col in data_keys
        ]
        df = df[data_columns]

    return df


def save_dataframe(df, args):
    args = eKonf.to_dict(args)

    if df is None:
        log.warning("Dataframe is None")
        return df
    filepath = args.pop("filepath", None)
    output_dir = args.pop("output_dir", None)
    output_file = args.pop("output_file", None)

    if filepath:
        filename = filepath
        output_dir = None
    else:
        filename = output_file
    eKonf.save_data(df, filename, output_dir, **args)
    return df


def load_dataframe(df=None, args=None):
    if args is None:
        raise ValueError("args must be specified")

    args = eKonf.to_dict(args)
    filepath = args.pop("filepath", None)
    data_dir = args.pop("data_dir", None)
    data_file = args.pop("data_file", None)

    if data_file:
        filename = data_file
    else:
        filename = filepath
        data_dir = None
    return eKonf.load_data(filename, data_dir, **args)


def summary_stats(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    output_dir = args.get("output_dir", ".")
    output_file = args.get("output_file", None)
    stat_args = args.get("info", {}).get("stats", None)
    if stat_args is None:
        log.warning("No stats specified")
        return df
    if output_file is None:
        output_file = "summary_stats.csv"
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    info_path = f"{output_dir}/{output_file}"
    stat_fn = eKonf.instantiate(stat_args)
    stats = stat_fn(df)
    eKonf.save(stats, f=info_path)
    log.info(f"Saving summary stats: {info_path}")
    if verbose:
        eKonf.pprint(stats)

    return df


def save_as_json(df, args):
    args = eKonf.to_dict(args)
    verbose = args.get("verbose", False)
    corpus_name = args.get("corpus_name", eKonf.Keys.CORPUS)
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
    log.info(f"Corpus is exported to {output_file_path}")
    return df


def pipeline(data=None, **cfg):
    args = eKonf.to_dict(cfg)
    verbose = args.get("verbose", False)

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        if eKonf.is_instantiatable(data):
            data = eKonf.instantiate(data)
        df = data.data.copy()

    if df is None:
        raise ValueError("No dataframe to process")

    process_pipeline = args.get(eKonf.Keys.PIPELINE, [])
    if process_pipeline is None:
        process_pipeline = []

    if len(process_pipeline) > 0:
        df = apply_pipeline(df, process_pipeline, args)
    else:
        log.warning("No pipeline specified")

    return df
