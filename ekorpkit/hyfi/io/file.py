"""File I/O functions"""
import os
import re
from glob import glob
from pathlib import Path, PosixPath, WindowsPath
from typing import List, Union

import pandas as pd

from ..utils.func import elapsed_timer
from ..utils.logging import getLogger

logger = getLogger(__name__)


def is_valid_regex(expr: str) -> bool:
    """Check if a string is a valid regular expression"""
    try:
        if expr.startswith("r:"):
            expr = expr[2:]
        else:
            return False
        re.compile(expr)
        return True
    except re.error:
        return False


def glob_re(pattern: str, base_dir: str, recursive: bool = False) -> list:
    """Glob files matching a regular expression"""
    if is_valid_regex(pattern):
        pattern = pattern[2:]
        pattern = re.compile(pattern)
        files = []
        if recursive:
            for (dirpath, dirnames, filenames) in os.walk(base_dir):
                files += [
                    os.path.join(dirpath, file)
                    for file in filenames
                    if pattern.search(file)
                ]
        else:
            files = [
                os.path.join(base_dir, file)
                for file in os.listdir(base_dir)
                if pattern.search(file)
            ]
    else:
        file = os.path.join(base_dir, pattern) if base_dir else pattern
        files = glob(file, recursive=recursive)
    return files


def get_filepaths(
    filename_patterns: Union[str, PosixPath, WindowsPath],
    base_dir: Union[str, PosixPath, WindowsPath] = None,
    recursive: bool = True,
    verbose: bool = False,
    **kwargs,
) -> List[str]:
    """Get a list of filepaths from a list of filename patterns"""
    if isinstance(filename_patterns, (PosixPath, WindowsPath)):
        filename_patterns = str(filename_patterns)
    if isinstance(filename_patterns, str):
        filename_patterns = [filename_patterns]
    filepaths = []
    base_dir = str(base_dir) if base_dir else None
    for file in filename_patterns:
        filepath = os.path.join(base_dir, file) if base_dir else file
        if os.path.exists(filepath):
            if Path(filepath).is_file():
                filepaths.append(filepath)
        else:
            if os.path.dirname(file) != "":
                _dir = os.path.dirname(file)
                file = os.path.basename(file)
                if base_dir:
                    base_dir = os.path.join(base_dir, _dir)
                else:
                    base_dir = _dir
            filepaths += glob_re(file, base_dir, recursive=recursive)
    filepaths = [fp for fp in filepaths if Path(fp).is_file()]
    if verbose:
        logger.info(f"Processing [{len(filepaths)}] files from {filename_patterns}")

    return filepaths


def get_files_from_archive(archive_path, filetype=None):
    """Get a list of files from an archive"""
    import tarfile
    from zipfile import ZipFile

    if ".tar.gz" in archive_path:
        logger.info(f"::Extracting files from {archive_path} with tar.gz")
        archive_handle = tarfile.open(archive_path, "r:gz")
        files = [
            (file, file.name) for file in archive_handle.getmembers() if file.isfile()
        ]
        open_func = archive_handle.extractfile
    elif ".tar.bz2" in archive_path:
        logger.info(f"::Extracting files from {archive_path} with tar.bz2")
        archive_handle = tarfile.open(archive_path, "r:bz2")
        files = [
            (file, file.name) for file in archive_handle.getmembers() if file.isfile()
        ]
        open_func = archive_handle.extractfile
    elif ".zip" in archive_path:
        logger.info(f"::Extracting files from {archive_path} with zip")
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


def is_dataframe(data) -> bool:
    """Check if data is a pandas dataframe"""
    return isinstance(data, pd.DataFrame)


def concat_data(
    data,
    columns=None,
    add_key_as_name: bool = False,
    name_column: str = "_name_",
    ignore_index: bool = True,
    **kwargs,
):
    """Concatenate dataframes"""
    if isinstance(data, dict):
        logger.info(f"Concatenating {len(data)} dataframes")
        dfs = []
        for df_name in data:
            df_each = data[df_name]
            if isinstance(columns, list):
                _columns = [c for c in columns if c in df_each.columns]
                df_each = df_each[_columns]
            if add_key_as_name:
                df_each[name_column] = df_name
            dfs.append(df_each)
        if len(dfs) > 0:
            return pd.concat(dfs, ignore_index=ignore_index)
        else:
            return None
    elif isinstance(data, list):
        logger.info(f"Concatenating {len(data)} dataframes")
        if len(data) > 0:
            return pd.concat(data, ignore_index=ignore_index)
        else:
            return None
    else:
        logger.warning("Warning: data is not a dict")
        return data


def load_data(filename, base_dir=None, filetype=None, verbose=False, **kwargs):
    """Load data from a file or a list of files"""
    concatenate = kwargs.pop("concatenate", False)
    ignore_index = kwargs.pop("ignore_index", False)
    if filename is not None:
        filename = str(filename)

    if filename.startswith("http"):
        if not filetype:
            filetype = filename.split(".")[-1]
        if filetype not in ["csv", "parquet"]:
            raise ValueError("`file` should be a csv or a parquet file.")
        kwargs["filetype"] = filetype
        return load_dataframe(filename, verbose=verbose, **kwargs)

    if base_dir:
        filepaths = get_filepaths(filename, base_dir)
    else:
        filepaths = get_filepaths(filename)
    if verbose:
        logger.info(f"Loading {len(filepaths)} dataframes from {filepaths}")

    data = {
        os.path.basename(f): load_dataframe(
            f, verbose=verbose, filetype=filetype, **kwargs
        )
        for f in filepaths
    }
    data = {k: v for k, v in data.items() if v is not None}
    if len(data) == 1:
        return list(data.values())[0]
    elif len(filepaths) > 1:
        if concatenate:
            return pd.concat(data.values(), ignore_index=ignore_index)
        else:
            return data
    else:
        logger.warning(f"No files found for {filename}")
        return None


def load_dataframe(
    filename: str,
    base_dir: str = None,
    columns: list = None,
    index_col=None,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Load a dataframe from a file"""
    dtype = kwargs.pop("dtype", None)
    if isinstance(dtype, list):
        dtype = {k: "str" for k in dtype}
    parse_dates = kwargs.pop("parse_dates", False)

    filetype = kwargs.pop("filetype", None) or "parquet"
    if filename.startswith("http"):
        filepath = filename
    else:
        fileinfo = os.path.splitext(filename)
        filename = fileinfo[0]
        filetype = fileinfo[1] if len(fileinfo) > 1 else filetype
        filetype = "." + filetype.replace(".", "")
        filename = f"{filename}{filetype}"
        if base_dir is not None:
            filepath = os.path.join(base_dir, filename)
        else:
            filepath = filename

        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} does not exist")
            return None
    if verbose:
        logger.info(f"Loading data from {filepath}")
    with elapsed_timer(format_time=True) as elapsed:
        if "csv" in filetype or "tsv" in filetype:
            delimiter = kwargs.pop("delimiter", "\t") if "tsv" in filetype else None
            data = pd.read_csv(
                filepath,
                index_col=index_col,
                dtype=dtype,
                parse_dates=parse_dates,
                delimiter=delimiter,
            )
        elif "parquet" in filetype:
            engine = kwargs.pop("engine", "pyarrow")
            data = pd.read_parquet(filepath, engine=engine)
        else:
            raise ValueError("filetype must be .csv or .parquet")
        if isinstance(columns, list):
            columns = [c for c in columns if c in data.columns]
            data = data[columns]
        if verbose:
            logger.info(" >> elapsed time to load data: {}".format(elapsed()))
    return data


def save_data(
    data: Union[pd.DataFrame, dict],
    filename: str,
    base_dir: str = None,
    columns=None,
    index=False,
    filetype="parquet",
    suffix: str = None,
    verbose: bool = False,
    **kwargs,
):
    """Save data to a file"""
    fileinfo = os.path.splitext(filename)
    filename = fileinfo[0]
    filetype = fileinfo[1] if len(fileinfo) > 1 else filetype
    filetype = "." + filetype.replace(".", "")
    if suffix is not None:
        filename = f"{filename}-{suffix}{filetype}"
    else:
        filename = f"{filename}{filetype}"
    if base_dir is not None:
        filepath = os.path.join(base_dir, filename)
    else:
        filepath = filename
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    os.makedirs(base_dir, exist_ok=True)

    if isinstance(data, dict):
        for k, v in data.items():
            save_data(
                v,
                filename,
                base_dir=base_dir,
                columns=columns,
                index=index,
                filetype=filetype,
                suffix=k,
                verbose=verbose,
                **kwargs,
            )
    elif is_dataframe(data):
        logger.info(f"Saving dataframe to {filepath}")
        if isinstance(columns, list):
            columns = [c for c in columns if c in data.columns]
            data = data[columns]
        with elapsed_timer(format_time=True) as elapsed:
            if "csv" in filetype or "tsv" in filetype:
                data.to_csv(filepath, index=index)
            elif "parquet" in filetype:
                compression = kwargs.get("compression", "gzip")
                engine = kwargs.get("engine", "pyarrow")
                data.to_parquet(filepath, compression=compression, engine=engine)
            else:
                raise ValueError("filetype must be .csv or .parquet")
            if verbose:
                logger.info(" >> elapsed time to save data: {}".format(elapsed()))
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def read(uri, mode="rb", encoding=None, head=None, **kwargs) -> bytes:
    """Read data from a file or url"""
    uri = str(uri)
    if uri.startswith("http"):
        import requests

        if mode == "r" and head is not None and isinstance(head, int):
            r = requests.get(uri, stream=True)
            r.raw.decode_content = True
            return r.raw.read(head)
        return requests.get(uri, **kwargs).content
    # elif uri.startswith("s3://"):
    #     import boto3

    #     s3 = boto3.resource("s3")
    #     bucket, key = uri.replace("s3://", "").split("/", 1)
    #     obj = s3.Object(bucket, key)
    #     return obj.get()["Body"].read()
    else:
        with open(uri, mode=mode, encoding=encoding) as f:
            if mode == "r" and head is not None and isinstance(head, int):
                return f.read(head)
            return f.read()


def is_file(a, *p) -> bool:
    """Check if path is a file"""
    _path = os.path.join(a, *p)
    return Path(_path).is_file()


def is_dir(a, *p) -> bool:
    """Check if path is a directory"""
    _path = os.path.join(a, *p)
    return Path(_path).is_dir()


def check_path(_path: str, alt_path: str = None) -> str:
    """Check if path exists, return alt_path if not"""
    if os.path.exists(_path):
        return _path
    else:
        return alt_path


def mkdir(_path: str) -> str:
    """Create directory if it does not exist"""
    if _path is None:
        return None
    Path(_path).mkdir(parents=True, exist_ok=True)
    return _path


def exists(a, *p) -> bool:
    """Check if path exists"""
    if a is None:
        return False
    _path = os.path.join(a, *p)
    return os.path.exists(_path)


def join_path(a, *p) -> str:
    """Join path components intelligently."""
    if p and p[0] is not None:
        p = [str(_p) for _p in p]
        if a is None:
            return os.path.join(*p)
        return os.path.join(a, *p)
    else:
        return a
