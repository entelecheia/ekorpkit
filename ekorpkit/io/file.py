import logging
import os
import pandas as pd
from glob import glob
from pathlib import Path
from ekorpkit.utils.func import elapsed_timer
from ekorpkit import eKonf

log = logging.getLogger(__name__)


def get_filepaths(
    filename_patterns, base_dir=None, recursive=True, verbose=True, **kwargs
):
    if isinstance(filename_patterns, str):
        filename_patterns = [filename_patterns]
    filepaths = []
    for file in filename_patterns:
        file = os.path.join(base_dir, file) if base_dir else file
        if os.path.exists(file):
            if Path(file).is_file():
                filepaths.append(file)
        else:
            filepaths += glob(file, recursive=recursive)
    filepaths = [fp for fp in filepaths if Path(fp).is_file()]
    if verbose:
        log.info(f"Processing [{len(filepaths)}] files from [{filename_patterns}]")

    return filepaths


def get_files_from_archive(archive_path, filetype=None):
    import tarfile
    from zipfile import ZipFile

    if ".tar.gz" in archive_path:
        log.info(f"::Extracting files from {archive_path} with tar.gz")
        archive_handle = tarfile.open(archive_path, "r:gz")
        files = [
            (file, file.name) for file in archive_handle.getmembers() if file.isfile()
        ]
        open_func = archive_handle.extractfile
    elif ".tar.bz2" in archive_path:
        log.info(f"::Extracting files from {archive_path} with tar.bz2")
        archive_handle = tarfile.open(archive_path, "r:bz2")
        files = [
            (file, file.name) for file in archive_handle.getmembers() if file.isfile()
        ]
        open_func = archive_handle.extractfile
    elif ".zip" in archive_path:
        log.info(f"::Extracting files from {archive_path} with zip")
        archive_handle = ZipFile(archive_path)
        files = [
            (file, file.encode("cp437").decode("euc-kr"))
            for file in archive_handle.namelist()
        ]
        open_func = archive_handle.open
    else:
        # print(f'::{archive_path} is not archive, use generic method')
        files = [(archive_path, os.path.basename(archive_path))]
        archive_handle = None
        open_func = None
    if filetype:
        files = [file for file in files if filetype in file[1]]

    return files, archive_handle, open_func


def save_dataframe(
    df,
    filepath=None,
    filetype=None,
    verbose=False,
    index=False,
    columns=None,
    name=None,
    _name_=None,
    output_dir=None,
    output_file=None,
    **kwargs,
):
    if df is None:
        log.warning("Dataframe is None")
        return df
    if isinstance(columns, list):
        df = df[columns]
    if verbose > 1:
        print(df.tail())

    if filepath:
        output_dir = os.path.dirname(filepath)
        output_file = os.path.basename(filepath)
    if output_file:
        fileinfo = os.path.splitext(output_file)
        filename = fileinfo[0]
        filetype = (
            fileinfo[1]
            if len(fileinfo) > 1
            else ("parquet" if not filetype else filetype)
        )
    else:
        filename = f"{name}"
        if not filetype:
            filetype = "parquet"
    filetype = "." + filetype.replace(".", "")
    if _name_ is not None:
        if _name_.endswith(filetype):
            filename = f"{filename}-{_name_}"
        else:
            filename = f"{filename}-{_name_}{filetype}"
    else:
        filename = f"{filename}{filetype}"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)

    log.info(f"Saving dataframe as {filepath}")
    with elapsed_timer(format_time=True) as elapsed:
        if "csv" in filetype or "tsv" in filetype:
            df.to_csv(filepath, index=index)
        elif "parquet" in filetype:
            df.to_parquet(filepath, compression="gzip", engine="pyarrow")
        else:
            raise ValueError("filetype must be .csv or .parquet")
        if verbose:
            log.info(" >> elapsed time to save data: {}".format(elapsed()))
    if verbose:
        log.info(f" >> saved dataframe to {filepath}")


def concat_dataframes(
    data,
    add_key_as_name=False,
    name_column=eKonf.Keys.NAME,
    concat={},
    columns=None,
    verbose=False,
    **kwargs,
):
    if isinstance(data, dict):
        log.info(f"Concatenating {len(data)} dataframes")
        dfs = []
        for df_name in data:
            df_each = data[df_name]
            if isinstance(columns, list):
                columns = [c for c in columns if c in df_each.columns]
                df_each = df_each[columns]
            if add_key_as_name:
                df_each[name_column] = df_name
            dfs.append(df_each)
        data = pd.concat(dfs, **concat)
        return data
    else:
        if verbose:
            log.info("Returning the original dataframe")
        return data


def load_dataframe(
    filepath=None,
    filetype=None,
    verbose=False,
    index_col=None,
    columns=None,
    name=None,
    data_dir=None,
    data_file=None,
    **kwargs,
):
    if filepath:
        data_dir = os.path.dirname(filepath)
        data_file = os.path.basename(filepath)
    if data_file:
        fileinfo = os.path.splitext(data_file)
        filename = fileinfo[0]
        filetype = (
            fileinfo[1]
            if len(fileinfo) > 1
            else ("parquet" if not filetype else filetype)
        )
    else:
        filename = f"{name}"
        if not filetype:
            filetype = "parquet"
    filetype = "." + filetype.replace(".", "")
    filename = f"{filename}{filetype}"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        log.warning(f"File {filepath} does not exist")
        return None
    log.info("Loading data from {}".format(filepath))
    with elapsed_timer(format_time=True) as elapsed:
        if "csv" in filetype:
            df = pd.read_csv(filepath, index_col=index_col, **kwargs)
        elif "tsv" in filetype:
            df = pd.read_csv(filepath, index_col=index_col, sep="\t", **kwargs)
        elif "parquet" in filetype:
            df = pd.read_parquet(filepath, engine="pyarrow")
        else:
            raise ValueError("filetype must be .csv or .parquet")
        if isinstance(columns, list):
            columns = [c for c in columns if c in df.columns]
            df = df[columns]
        if verbose:
            log.info(" >> elapsed time to load data: {}".format(elapsed()))
    return df
