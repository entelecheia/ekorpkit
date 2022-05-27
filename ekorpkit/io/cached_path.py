import os
import pathlib
import logging

log = logging.getLogger(__name__)


def cached_path(
    url_or_filename,
    extract_archive: bool = False,
    force_extract: bool = False,
    return_dir: bool = False,
    cache_dir=None,
    verbose: bool = False,
):
    import cached_path

    if verbose:
        log.info(
            "caching path: {}, extract_archive: {}, force_extract: {}, cache_dir: {}".format(
                url_or_filename, extract_archive, force_extract, cache_dir
            )
        )

    if url_or_filename:
        try:
            if url_or_filename.startswith("gd://"):

                _path = cached_gdown(
                    url_or_filename,
                    verbose=verbose,
                    extract_archive=extract_archive,
                    force_extract=force_extract,
                    cache_dir=cache_dir,
                )
            else:
                if cache_dir is None:
                    cache_dir = (
                        pathlib.Path.home() / ".ekorpkit" / ".cache" / "cached_path"
                    )
                else:
                    cache_dir = pathlib.Path(cache_dir) / "cached_path"

                _path = cached_path.cached_path(
                    url_or_filename,
                    extract_archive=extract_archive,
                    force_extract=force_extract,
                    cache_dir=cache_dir,
                ).as_posix()

            log.info(f"cached path: {_path}")
            if return_dir and pathlib.Path(_path).is_file():
                _path = pathlib.Path(_path).parent

            return _path

        except Exception as e:
            log.error(e)
            return None


def cached_gdown(
    url, verbose=False, extract_archive=None, force_extract=False, cache_dir=None
):
    """
    :type url: str
          ex) gd://id:path
    :type verbose: bool
    :type extract_archive: bool
    :type force_extract: bool
    :type cache_dir: str
    :returns: str
    """
    import gdown

    if verbose:
        log.info(f"Downloading {url}...")
    if cache_dir is None:
        cache_dir = pathlib.Path.home() / ".ekorpkit" / ".cache" / "gdown"
    else:
        cache_dir = pathlib.Path(cache_dir) / "gdown"
    cache_dir.mkdir(parents=True, exist_ok=True)

    gd_prefix = "gd://"
    if url.startswith(gd_prefix):
        url = url[len(gd_prefix) :]
        _url = url.split(":")
        if len(_url) == 2:
            id, path = _url
        else:
            id = _url[0]
            path = id

        # If we're using the path!c/d/file.txt syntax, handle it here.
        fname = None
        extraction_path = path
        exclamation_index = path.find("!")
        if extract_archive and exclamation_index >= 0:
            extraction_path = path[:exclamation_index]
            fname = path[exclamation_index + 1 :]

        cache_path = cache_dir / f".{id}" / extraction_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache_path = gdown.cached_download(
            id=id,
            path=cache_path.as_posix(),
            quiet=not verbose,
        )

        if extract_archive:
            extraction_path, files = extractall(cache_path, force_extract=force_extract)

            if fname and files:
                for f in files:
                    if f.endswith(fname):
                        return f
            else:
                return extraction_path
        return cache_path

    else:
        log.warning(f"Unknown url: {url}")
        return None


def extractall(path, to=None, force_extract=False):
    """Extract archive file.

    Parameters
    ----------
    path: str
        Path of archive file to be extracted.
    to: str, optional
        Directory to which the archive file will be extracted.
        If None, it will be set to the parent directory of the archive file.
    """
    import tarfile
    from zipfile import ZipFile

    if to is None:
        to = os.path.dirname(path)

    if path.endswith(".zip"):
        opener, mode = ZipFile, "r"
    elif path.endswith(".tar"):
        opener, mode = tarfile.open, "r"
    elif path.endswith(".tar.gz") or path.endswith(".tgz"):
        opener, mode = tarfile.open, "r:gz"
    elif path.endswith(".tar.bz2") or path.endswith(".tbz"):
        opener, mode = tarfile.open, "r:bz2"
    else:
        log.warning(
            "Could not extract '%s' as no appropriate " "extractor is found" % path
        )
        return path, None

    def namelist(f):
        if isinstance(f, ZipFile):
            return f.namelist()
        return [m.path for m in f.members]

    def filelist(f):
        files = []
        for fname in namelist(f):
            fname = os.path.join(to, fname)
            files.append(fname)
        return files

    extraction_name = pathlib.Path(path).stem
    extraction_path = f"{to}/{extraction_name}"
    if extraction_path is not None:
        # If the extracted directory already exists (and is non-empty), then no
        # need to extract again unless `force_extract=True`.
        if (
            os.path.isdir(extraction_path)
            and os.listdir(extraction_path)
            and not force_extract
        ):
            files = [
                os.path.join(dirpath, filename)
                for dirpath, _, filenames in os.walk(extraction_path)
                for filename in filenames
            ]

            return to, files

    with opener(path, mode) as f:
        f.extractall(path=to)

    return to, filelist(f)
