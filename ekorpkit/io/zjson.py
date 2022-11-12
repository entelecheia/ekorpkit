import zstandard
import orjson as json


def load(
    fp: str,
    **kw,
):
    """Deserialize ``fp`` (a file path) to a Python object."""
    if str(fp).endswith(".zst"):
        with open(fp, "rb") as fh:
            cctx = zstandard.ZstdDecompressor()
            data = cctx.decompress(fh.read())
            return json.loads(data, **kw)
    else:
        with open(fp, "r") as fh:
            return json.loads(fh.read(), **kw)


def dump(
    obj,
    fp: str,
    comress_level=3,
    **kw,
):
    """Serialize ``obj`` as a JSON formatted stream to ``fp`` (a file path)."""

    if str(fp).endswith(".zst"):

        cctx = zstandard.ZstdCompressor(level=comress_level)
        cdata = cctx.compress(json.dumps(obj, **kw))
        with open(fp, "wb") as fh:
            fh.write(cdata)
    else:

        with open(fp, "w") as fh:
            fh.write(json.dumps(obj, **kw))
