import os


def check_path(path):
    return os.path.exists(path)


def check_dir(filepath):
    dirname = os.path.abspath(os.path.dirname(filepath))
    if not os.path.exists(dirname):
        os.makedirs(dirname)
