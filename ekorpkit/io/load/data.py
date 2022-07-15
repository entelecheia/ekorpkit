import logging
from io import StringIO
from zipfile import ZipFile
import pandas as pd
from tqdm.auto import tqdm
from ekorpkit.io.parse.json import parse_data
from joblib import Parallel, delayed
from ekorpkit.utils.batch.batcher import tqdm_joblib
from ekorpkit.io.file import get_files_from_archive, get_filepaths
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def load_data(split_name=None, **loader_cfg):
    data_dir = loader_cfg["data_dir"]
    if split_name:
        data_files = loader_cfg["data_sources"][split_name]
    else:
        data_files = loader_cfg["data_sources"]
    filepaths = get_filepaths(data_files, data_dir)

    num_workers = loader_cfg.get("num_workers", 1)
    filetype = loader_cfg.get("filetype", None)
    # data_items =  args['data']['item'].keys()
    multiprocessing_at = loader_cfg.get("multiprocessing_at", "load_data")

    if split_name:
        default_items = {eKonf.Keys.SPLIT.value: split_name}
    else:
        default_items = {}
    documents = []
    num_workers = num_workers if num_workers else 1
    num_files = len(filepaths)
    processes = min(num_workers, num_files)
    if multiprocessing_at == "load_data" and num_files < processes // 2:
        multiprocessing_at = "_load_archive"
        loader_cfg["multiprocessing_at"] = multiprocessing_at
    if processes > 2 and multiprocessing_at == "load_data":
        log.info(
            f"Starting multiprocessing with {processes} processes at {multiprocessing_at}"
        )
        desciption = "::load_data()"
        with tqdm_joblib(tqdm(desc=desciption, total=num_files)):
            results = Parallel(n_jobs=num_workers)(
                delayed(_load_archive)(filepath, filetype, default_items, loader_cfg, 1)
                for filepath in filepaths
            )
            for result in results:
                documents += result
    else:
        for fno, filepath in enumerate(filepaths):
            log.info(f"==> processing {(fno+1)}/{num_files} files <==")
            documents += _load_archive(
                filepath, filetype, default_items, loader_cfg, num_workers
            )

    print(documents[0])
    df = pd.DataFrame(documents)
    return df


def _load_archive(filepath, filetype, default_items, loader_args, num_workers):
    from loky import get_reusable_executor

    multiprocessing_at = loader_args.get("multiprocessing_at", "load_data")
    limit = loader_args.get("limit", None)

    files, arch_handle, open_func = get_files_from_archive(filepath, filetype=filetype)
    num_files = len(files)
    if (
        num_workers > 2
        and multiprocessing_at == "_load_archive"
        and num_files > num_workers
    ):
        log.info(
            f"Starting multiprocessing with {num_workers} processes at {multiprocessing_at}"
        )
        desciption = "::_load_archive()"
        executor = get_reusable_executor(max_workers=num_workers)
        pbar = tqdm(total=num_files, desc=desciption)
    else:
        executor = None

    documents = []
    results = []
    for fno, (file, filename) in enumerate(files):
        if limit and fno >= limit:
            if len(documents) > 0:
                print(documents[0])
            break
        default_items["filename"] = filename

        try:
            if open_func is not None:
                with open_func(file) as f:
                    contents = f.read()
            else:
                with open(file, "rb") as f:
                    contents = f.read()
        except Exception as e:
            log.critical(f"Error reading {filename}", e)
            continue

        if contents is None:
            log.warning("contents is empty, skipping")
            continue
        if executor is not None:
            pbar.set_description(f"{filename}")
            results.append(
                executor.submit(parse_data, contents, loader_args, default_items, 1)
            )
            pbar.update(1)
        else:
            documents += parse_data(contents, loader_args, default_items, num_workers)

    if arch_handle is not None:
        arch_handle.close()
    if executor is not None:
        # pbar.close()
        for result in results:
            documents += result.result()
    return documents


def load_dataframe(split_name, **loader_cfg):
    data_dir = loader_cfg["data_dir"]
    data_files = loader_cfg["data_sources"][split_name]
    filepaths = get_filepaths(data_files, data_dir)

    documents = []
    log.info(f"Loading [{split_name}] documents from {len(filepaths)} files")
    for filepath in filepaths:
        log.info("==> processing {}".format(filepath))
        if filepath.endswith(".zip"):
            with ZipFile(filepath) as zf:
                for f_csv in zf.namelist():
                    with zf.open(f_csv, "r") as zfo:
                        content = zfo.read().decode(errors="ignore")
            df = pd.read_csv(StringIO(content), index_col=None)
        else:
            df = eKonf.load_data(filepath, index_col=None)
        documents.append(df)
    df = pd.concat(documents, ignore_index=True)
    return df


def load_hfds(split_name, **loader_cfg):
    from datasets import load_dataset

    dataset_name = loader_cfg["name"]
    split = loader_cfg["data_sources"][split_name]
    subsets = loader_cfg["subset"]
    download_mode = loader_cfg.get("download_mode", "force_redownload")
    ignore_verifications = loader_cfg.get("ignore_verifications", True)

    if isinstance(subsets, str):
        subsets = [subsets]
    elif not isinstance(subsets, list):
        subsets = [None]

    log.info(
        f"Loading [{split_name}] documents from huggingface datasets [{dataset_name}]"
    )

    dfs = []
    for subset in subsets:
        ds = load_dataset(
            dataset_name,
            subset,
            split=split,
            download_mode=download_mode,
            ignore_verifications=ignore_verifications,
        )
        print(ds)
        df = ds.to_pandas()
        df["subset"] = subset
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def load_tsv_data(split_name, **loader_cfg):
    data_dir = loader_cfg["data_dir"]
    data_files = loader_cfg["data_sources"][split_name]
    filepaths = get_filepaths(data_files, data_dir)
    data_args = loader_cfg["data"]
    sep = data_args.get("sep", "\t")
    sep = str(sep).encode("utf-8").decode("unicode_escape")
    dtype = dict(data_args.get("dtype", None))
    names = list(dtype.keys())
    constants = data_args.get("constants", None)

    documents = []
    log.info(f"Loading [{split_name}] documents from {len(filepaths)} files")
    for filepath in filepaths:
        log.info("==> processing {}".format(filepath))
        if filepath.endswith(".zip"):
            with ZipFile(filepath) as zf:
                for f_csv in zf.namelist():
                    with zf.open(f_csv, "r") as zfo:
                        content = zfo.read().decode(errors="ignore")
            df = pd.read_csv(
                StringIO(content), index_col=None, sep=sep, dtype=dtype, names=names
            )
        else:
            df = pd.read_csv(
                filepath, index_col=None, sep=sep, dtype=dtype, names=names
            )
        documents.append(df)
    df = pd.concat(documents, ignore_index=True)
    if constants is not None:
        for k, v in constants.items():
            df[k] = v
    return df


def read_conll(filename, **args):
    sep = "\t" if args["sep"] is None else args["sep"]
    column_dtypes = args["column_dtypes"]
    names = (
        ["words", "pos", "chunk", "labels"]
        if column_dtypes is None
        else column_dtypes.keys()
    )
    filter_query = args["filter_query"]

    df = pd.read_csv(
        filename,
        sep=sep,
        names=names,
        header=None,
        keep_default_na=False,
        quoting=3,
        skip_blank_lines=False,
        engine="python",
    )
    df = df[
        ~df["words"].astype(str).str.startswith("-DOCSTART-")
    ]  # Remove the -DOCSTART- header
    df["sentence_id"] = (df.words == "").cumsum()
    df = df[df.words != ""]
    df = df.astype(dict(column_dtypes), errors="raise")
    if filter_query:
        df = df.query(filter_query, engine="python")
    return df


def read_csv(filename, **args):
    sep = "," if args["sep"] is None else args["sep"]
    column_mapping = args["column_mapping"] if "column_mapping" in args else None
    filter_query = args["filter_query"] if "filter_query" in args else None
    index_col = args["index_col"] if "index_col" in args else None

    df = pd.read_csv(filename, sep=sep, index_col=index_col, engine="python")
    if column_mapping is not None:
        df.rename(columns=column_mapping, inplace=True)
    if filter_query:
        df = df.query(filter_query, engine="python")
    return df
