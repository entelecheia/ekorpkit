import ast
import time
import os
import re
import datetime
import chardet
from contextlib import contextmanager
from timeit import default_timer


def unescape_dict(d):
    return ast.literal_eval(repr(d).encode("utf-8").decode("unicode-escape"))


@contextmanager
def elapsed_timer(format_time=False):
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: str(
        datetime.timedelta(seconds=elapser())
    ) if format_time else elapser()
    end = default_timer()
    elapser = lambda: end - start


def lower_case_with_underscores(s):
    return re.sub(r"\s+", "_", s.lower()).replace("-", "_")


ordinal = lambda n: "%d%s" % (
    n,
    "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
)


def get_offset_ranges(count, num_workers):
    assert count > num_workers
    step_sz = int(count / num_workers)
    offset_ranges = [0]
    pv_cnt = 1
    for i in range(num_workers):
        if i == num_workers - 1:
            pv_cnt = count
        else:
            pv_cnt = pv_cnt + step_sz
        offset_ranges.append(pv_cnt)
    return offset_ranges


@contextmanager
def change_directory(directory):
    original = os.path.abspath(os.getcwd())

    fancy_print(" Change directory to {}".format(directory))
    os.chdir(directory)
    yield

    fancy_print(" Change directory back to {}".format(original))
    os.chdir(original)


def fancy_print(*args, color=None, bold=False, **kwargs):
    if bold:
        print("\033[1m", end="")

    if color:
        print("\033[{}m".format(color), end="")

    print(*args, **kwargs)

    print("\033[0m", end="")  # reset


# https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb/37423778
def humanbytes(B, units=None):
    "Return the given bytes as a human friendly KB, MB, GB, or TB string"
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if (B < KB and units is None) or units == "B":
        return "{0} {1}".format(B, "Bytes" if 0 == B > 1 else "Byte")
    elif (KB <= B < MB and units is None) or units == "KiB":
        return "{0:.2f} KiB".format(B / KB)
    elif (MB <= B < GB and units is None) or units == "MiB":
        return "{0:.2f} MiB".format(B / MB)
    elif (GB <= B < TB and units is None) or units == "GiB":
        return "{0:.2f} GiB".format(B / GB)
    elif (TB <= B and units is None) or units == "TiB":
        return "{0:.2f} TiB".format(B / TB)


def parse_size(sizestr):
    unit = sizestr[-1]
    size = float(sizestr[:-1])

    if unit.upper() == "B":
        return size
    if unit.upper() == "K":
        return size * 1024
    if unit.upper() == "M":
        return size * 1024 * 1024
    if unit.upper() == "G":
        return size * 1024 * 1024 * 1024
    if unit.upper() == "T":
        return size * 1024 * 1024 * 1024 * 1024


def check_min_len(s, len_func, min_len):
    return len_func(s) >= min_len


def check_max_len(s, len_func, max_len):
    return len_func(s) <= max_len


def utf8len(s):
    return len(str(s).encode("utf-8"))


def len_wospc(x):
    return utf8len(re.sub(r"\s", "", str(x)))


def len_bytes(x):
    return utf8len(x)


def len_words(x):
    if isinstance(x, str):
        return len(x.split())
    return 0


def len_sents(x, sep):
    sep = str(sep).encode("utf-8").decode("unicode-escape")
    return len(re.sub(r"(\r?\n|\r){1,}", sep, x).split(sep))


def len_segments(x, sep):
    sep = str(sep).encode("utf-8").decode("unicode-escape")
    return len(re.sub(r"(\r?\n|\r){2,}", sep, x).split(sep))


def any_to_utf8(b):
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        # try to figure out encoding if not urf-8

        guess = chardet.detect(b)["encoding"]

        if not guess or guess == "UTF-8":
            return

        try:
            return b.decode(guess)
        except (UnicodeDecodeError, LookupError):
            # still cant figure out encoding, give up
            return


def get_modified_time(path):
    if not os.path.exists(path):
        return None
    modTimesinceEpoc = os.path.getmtime(path)
    # Convert seconds since epoch to readable timestamp
    modificationTime = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(modTimesinceEpoc)
    )
    return modificationTime
