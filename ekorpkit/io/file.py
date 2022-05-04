import logging
import os
import pandas as pd
from glob import glob
from pathlib import Path
from ekorpkit.utils.func import elapsed_timer


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
    filepath,
    filetype=None,
    verbose=False,
    index=False,
    columns_to_keep=None,
    **kwargs,
):
    if df is None:
        log.warning("Dataframe is None")
        return df
    if columns_to_keep is not None:
        df = df[columns_to_keep]
    if verbose > 1:
        print(df.tail())

    output_dir = os.path.dirname(filepath)
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    if filetype is None:
        filetype = os.path.splitext(filepath)[1]

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
        print(f" >> saved dataframe to {filepath}")


def load_dataframe(filepath, filetype=None, verbose=False, index_col=None, **kwargs):
    log.info("Loading data from {}".format(filepath))
    if filetype is None:
        filetype = os.path.splitext(filepath)[1]
    with elapsed_timer(format_time=True) as elapsed:
        if "csv" in filetype:
            df = pd.read_csv(filepath, index_col=index_col, **kwargs)
        elif "tsv" in filetype:
            df = pd.read_csv(filepath, index_col=index_col, sep="\t", **kwargs)
        elif "parquet" in filetype:
            df = pd.read_parquet(filepath, engine="pyarrow")
        else:
            raise ValueError("filetype must be .csv or .parquet")
        if verbose:
            log.info(" >> elapsed time to load data: {}".format(elapsed()))
    return df
