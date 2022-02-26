import _pickle as cPickle
import bz2
import sys


def save_pickle(filename, data, add_suffix=False, suffix=".pkl.bz2"):
    """
    save object to file using pickle

    @param filename: name of destination file
    @type filename: str
    @param data: object to save (has to be pickleable)
    @type data: obj
    """
    filename = str(filename)
    if add_suffix and not filename.endswith(suffix):
        filename += suffix

    try:
        f = bz2.BZ2File(filename, "wb")
    except IOError:
        sys.stderr.write("File " + filename + " cannot be written\n")
        return

    cPickle.dump(data, f)
    f.close()


def load_pickle(filename):
    """
    Load from filename using pickle

    @param filename: name of file to load from
    @type filename: str
    """

    filename = str(filename)
    try:
        f = bz2.BZ2File(filename, "rb")
    except IOError:
        sys.stderr.write("File " + filename + " cannot be read\n")
        return

    data = cPickle.load(f)
    f.close()
    return data
