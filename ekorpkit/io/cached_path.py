import os
import pathlib
import logging

log = logging.getLogger(__name__)


def cached_path(
    url_or_filename,
    extract_archive: bool = False,
    force_extract: bool = False,
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
                if extract_archive:
                    postprocess = extractall
                else:
                    postprocess = None

                _path = cached_gdown(
                    url_or_filename,
                    verbose=verbose,
                    postprocess=postprocess,
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

            if verbose:
                log.info(f"cached path: {_path}")

            return _path

        except Exception as e:
            log.error(e)
            return None


def cached_gdown(url, verbose=False, postprocess=None, cache_dir=None):
    """
    :type url: str
          ex) gd://id:path
    :type verbose: bool
    :type postprocess: callable
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
        _path = url.split(":")
        if len(_path) == 2:
            id, path = _path
        else:
            id = _path[0]
            path = id
        cache_path = cache_dir / path

        return gdown.cached_download(
            id=id,
            path=cache_path.as_posix(),
            quiet=not verbose,
            postprocess=postprocess,
        )
    else:
        log.warning(f"Unknown url: {url}")
        return None


def extractall(path, to=None):
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
        return path

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

    with opener(path, mode) as f:
        f.extractall(path=to)

    return filelist(f)
