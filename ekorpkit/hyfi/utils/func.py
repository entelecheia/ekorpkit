import ast
import time
import os
import re
import datetime
import itertools
from typing import List, Type
from contextlib import contextmanager
from timeit import default_timer
from functools import partial
import chardet


def unescape_dict(d):
    return ast.literal_eval(repr(d).encode("utf-8").decode("unicode-escape"))


def _elapser_timer(start):
    return default_timer() - start


def _elapser(start, end):
    return end - start


@contextmanager
def elapsed_timer(format_time=False):
    """A context manager that yields a function that returns the elapsed time"""
    start = default_timer()
    # elapser = lambda: default_timer() - start
    elapser = partial(_elapser_timer, start)
    yield lambda: str(
        datetime.timedelta(seconds=elapser())
    ) if format_time else elapser()
    end = default_timer()
    # elapser = lambda: end - start
    elapser = partial(_elapser, start, end)


def lower_case_with_underscores(string):
    """Converts 'CamelCased' to 'camel_cased'."""
    return re.sub(r"\s+", "_", string.lower()).replace("-", "_")


def ordinal(num):
    """Return the ordinal of a number as a string."""
    return "%d%s" % (
        num,
        "tsnrhtdd"[(num // 10 % 10 != 1) * (num % 10 < 4) * num % 10 :: 4],
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

    fancy_print(f" Change directory to {directory}")
    os.chdir(directory)
    try:
        yield

    except Exception as e:
        fancy_print(" Exception: {}".format(e))
        raise e

    finally:
        fancy_print(" Change directory back to {}".format(original))
        os.chdir(original)


def fancy_print(*args, color=None, bold=False, **kwargs):
    """Print with color and bold"""
    if bold:
        print("\033[1m", end="")

    if color:
        print(f"\033[{color}m", end="")

    print(*args, **kwargs)

    print("\033[0m", end="")  # reset


# https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb/37423778
def humanbytes(B, units=None):
    "Return the given bytes as a human friendly KB, MB, GB, or TB string"
    B = float(B)
    KB = float(1024)
    MB = float(KB**2)  # 1,048,576
    GB = float(KB**3)  # 1,073,741,824
    TB = float(KB**4)  # 1,099,511,627,776

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
        # try to figure out encoding if not utf-8

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


def _today(_format="%Y-%m-%d"):
    from datetime import datetime

    if _format is None:
        return datetime.today().date()
    else:
        return datetime.today().strftime(_format)


def _now(_format="%Y-%m-%d %H:%M:%S"):
    from datetime import datetime

    if _format is None:
        return datetime.now()
    else:
        return datetime.now().strftime(_format)


def _strptime(
    _date_str: str,
    _format: str = "%Y-%m-%d",
):
    from datetime import datetime

    return datetime.strptime(_date_str, _format)


def _to_dateparm(_date, _format="%Y-%m-%d"):
    from datetime import datetime

    _dtstr = datetime.strftime(_date, _format)
    _dtstr = "${to_datetime:" + _dtstr + "," + _format + "}"
    return _dtstr


def _to_datetime(data, _format=None, _columns=None, **kwargs):
    import pandas as pd
    from datetime import datetime

    if isinstance(data, datetime):
        return data
    elif isinstance(data, str):
        if _format is None:
            _format = "%Y-%m-%d"
        return datetime.strptime(data, _format)
    elif isinstance(data, int):
        return datetime.fromtimestamp(data)
    elif isinstance(data, pd.DataFrame):
        if _columns:
            if isinstance(_columns, str):
                _columns = [_columns]
            for _col in _columns:
                data[_col] = pd.to_datetime(data[_col], format=_format, **kwargs)
        return data
    else:
        return data


def _to_numeric(data, _columns=None, errors="coerce", downcast=None, **kwargs):
    import pandas as pd

    if isinstance(data, str):
        return float(data)
    elif isinstance(data, int):
        return data
    elif isinstance(data, float):
        return data
    elif isinstance(data, pd.DataFrame):
        if _columns:
            if isinstance(_columns, str):
                _columns = [_columns]
            for _col in _columns:
                data[_col] = pd.to_numeric(data[_col], errors=errors, downcast=downcast)
        return data
    else:
        return data


def _human_readable_type_name(t: Type) -> str:
    """
    Generates a useful-for-humans label for a type.
    For builtin types, it's just the class name (eg "str" or "int").
    For other types, it includes the module (eg "pathlib.Path").
    """
    module = t.__module__
    if module == "builtins":
        return t.__qualname__
    elif module.split(".")[0] == "ekorpkit":
        module = "ekorpkit"

    try:
        return module + "." + t.__qualname__
    except AttributeError:
        return str(t)


def _readable_types_list(type_list: List[Type]) -> str:
    return ", ".join(_human_readable_type_name(t) for t in type_list)


def _dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def _dict_to_dataframe(data, orient="columns", dtype=None, columns=None):
    import pandas as pd

    return pd.DataFrame.from_dict(data, orient=orient, dtype=dtype, columns=columns)


def _records_to_dataframe(
    data, index=None, exclude=None, columns=None, coerce_float=False, nrows=None
):
    import pandas as pd

    return pd.DataFrame.from_records(
        data,
        index=index,
        exclude=exclude,
        columns=columns,
        coerce_float=coerce_float,
        nrows=nrows,
    )
