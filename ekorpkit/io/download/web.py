import gzip
import shutil
import os
import requests
import tarfile
import zipfile
from tqdm import tqdm
from urllib import request
from ekorpkit import eKonf


GOOGLE_DRIVE_URL = "https://docs.google.com/uc?export=download"


def download_from_gcs(**cfg):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """
    import tensorflow as tf

    args = eKonf.to_config(cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.listdir(args.output_dir) or args.force_download:
        for filename, source_path in args.data_sources.items():
            output_path = f"{args.output_dir}/{filename}"
            print(f"Downloading {source_path} to {output_path}")
            tf.io.gfile.copy(source_path, output_path, overwrite=args.force_download)
    else:
        print(f" >> {args.output_dir} is not empty, skip downloading")


def download_from_web(**cfg):
    args = eKonf.to_config(cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.listdir(args.output_dir) or args.force_download:
        for filename, url in args.urls.items():
            file_path = f"{args.output_dir}/{filename}"
            print(f"Downloading {url} to {file_path}")
            # download_file(url, file_path)
            web_download(url, file_path, args.name, force_download=False)
        if args.extract_command:
            print(args.extract_command)
    else:
        print(f" >> {args.output_dir} is not empty, skip downloading")


def download_from_gdrive(**cfg):
    args = eKonf.to_config(cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    for filename, url in args.urls.items():
        file_path = f"{args.output_dir}/{filename}"
        google_drive_download(url, file_path, args.name, args.force_download)


def download_from_gdrive_untar(**cfg):
    args = eKonf.to_config(cfg)
    os.makedirs(args.output_dir, exist_ok=True)
    for filename, url in args.urls.items():
        file_path = f"{args.output_dir}/{filename}"
        google_download_untar(url, file_path, args.name, args.force_download)


def web_download(url, output_path, name="", force_download=False):
    site = request.urlopen(url)
    meta = site.info()
    remote_size = int(meta["Content-Length"])
    if (
        (not force_download)
        and os.path.exists(output_path)
        and (os.stat(output_path).st_size == remote_size)
    ):
        print(f"[eKorpkit] Corpus `{name}` is already downloaded at {output_path}")
        return None
    filename = os.path.basename(output_path)
    with tqdm(
        unit="B", unit_scale=True, miniters=1, desc=f"[{name}] download {filename}"
    ) as t:
        request.urlretrieve(url, filename=output_path, reporthook=_reporthook(t))


def web_download_unzip(url, zip_path, corpus_name="", force_download=False):
    web_download(url, zip_path, corpus_name, force_download)
    # assume that path/to/abc.zip consists path/to/abc
    data_path = zip_path[:-4]
    if (not force_download) and os.path.exists(data_path):
        print(f"[eKorpkit] Corpus `{corpus_name}` is already extracted at {data_path}")
        return None
    data_root = os.path.dirname(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root)
    print(f"unzip {data_path}")


def web_download_un7z(url, zip_path, corpus_name="", force_download=False):
    import py7zr

    web_download(url, zip_path, corpus_name, force_download)
    # assume that path/to/abc.zip consists path/to/abc
    data_path = zip_path[:-3]
    if (not force_download) and os.path.exists(data_path):
        print(f"[eKorpkit] Corpus `{corpus_name}` is already extracted at {data_path}")
        return None
    # data_root = os.path.dirname(zip_path)
    with py7zr.SevenZipFile(zip_path, mode="r") as z:
        z.extractall()
    print(f"un7z {data_path}")


def web_download_untar(url, tar_path, corpus_name="", force_download=False):
    web_download(url, tar_path, corpus_name, force_download)
    # assume that path/to/abc.tar consists path/to/abc
    data_path = tar_path[:-4]
    if (not force_download) and os.path.exists(data_path):
        print(f"[eKorpkit] Corpus `{corpus_name}` is already extracted at {data_path}")
        return None
    data_root = os.path.dirname(tar_path)
    with tarfile.open(tar_path) as tar:
        tar.extractall(data_root)
    print(f"decompress {tar_path}")


def web_download_ungzip(url, gzip_path, corpus_name="", force_download=False):
    web_download(url, gzip_path, corpus_name, force_download)
    # assume that path/to/abc.gzip consists path/to/abc
    data_path = gzip_path[:-3]
    if (not force_download) and os.path.exists(data_path):
        print(f"[eKorpkit] Corpus `{corpus_name}` is already extracted at {data_path}")
        return None
    with gzip.open(gzip_path, "rb") as fi:
        with open(data_path, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    print(f"decompress {gzip_path}")


def google_download_un7z(file_id, zip_path, corpus_name="", force_download=False):
    import py7zr

    google_drive_download(file_id, zip_path, corpus_name, force_download)
    # assume that path/to/abc.zip consists path/to/abc
    data_path = zip_path[:-3]
    if (not force_download) and os.path.exists(data_path):
        print(f"[eKorpkit] Corpus `{corpus_name}` is already extracted at {data_path}")
        return None
    # data_root = os.path.dirname(zip_path)
    with py7zr.SevenZipFile(zip_path, mode="r") as z:
        z.extractall(path=data_path)
    print(f"un7z {data_path}")


def google_download_untar(file_id, local_path, corpus_name="", force_download=False):
    google_drive_download(file_id, local_path, corpus_name, force_download)
    # assume that path/to/abc.tar consists path/to/abc
    data_path = local_path[: local_path.find(".tar")]
    if (not force_download) and os.path.exists(data_path):
        print(f"[eKorpkit] Corpus `{corpus_name}` is already extracted at {data_path}")
        return None
    data_root = os.path.dirname(local_path)
    with tarfile.open(local_path) as tar:
        tar.extractall(data_root)
    print(f"decompress {local_path}")


def google_drive_download(file_id, local_path, filename="", force_download=False):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    if (not force_download) and os.path.exists(local_path):
        print(f"[eKorpkit] `{filename}` is already downloaded at {local_path}")
        return None

    # init a HTTP session
    session = requests.Session()

    # make a request
    response = session.get(GOOGLE_DRIVE_URL, params={"id": file_id}, stream=True)

    # get confirmation token
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(GOOGLE_DRIVE_URL, params=params, stream=True)

    # download to disk
    with open(local_path, "wb") as f:
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(
            unit="B",
            unit_scale=True,
            total=total,
            initial=0,
            desc=f"[{filename}] download {local_path}",
        )
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                progress.update(len(chunk))
                f.write(chunk)
        progress.close()


def _reporthook(t):
    """``reporthook`` to use with ``urllib.request`` that prints the process of the download.

    Uses ``tqdm`` for progress bar.

    **Reference:**
    https://github.com/tqdm/tqdm

    Args:
        t (tqdm.tqdm) Progress bar.

    Example:
        >>> with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:  # doctest: +SKIP
        ...   urllib.request.urlretrieve(file_url, filename=full_path, reporthook=reporthook(t))
    """
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        Args:
            b (int, optional): Number of blocks just transferred [default: 1].
            bsize (int, optional): Size of each block (in tqdm units) [default: 1].
            tsize (int, optional): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner
