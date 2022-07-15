import logging

log = logging.getLogger(__name__)


def html_to_json(contents):
    import html_to_json

    doc = {}
    output_json = html_to_json.convert(contents)
    for k, v in output_json.items():
        if isinstance(v, list) and len(v) == 1:
            doc[k] = v[0].get("_value")
    return [doc]


def parse_plaintext(
    contents,
    split=False,
    meta_line=None,
    meta_key="meta",
    data_key="text",
    lineno_key="lineno",
    progress_per=1000,
    verbose=False,
    **kwargs,
):
    if split:
        docs = []
        for i, line in enumerate(contents.splitlines()):
            line = line.strip()
            doc = {lineno_key: i, data_key: line}
            if i < 2 and verbose:
                log.info(f"processing line {i}")
                log.info(doc)
            elif progress_per and i > 0 and i % progress_per == 0:
                log.info(f"processing line {i}")
            docs.append(doc)
        return docs
    elif meta_line is not None:
        text = []
        meta = []
        for i, line in enumerate(contents.splitlines()):
            line = line.strip()
            if i < meta_line:
                meta.append(line)
            else:
                text.append(line)
        doc = {meta_key: "\n".join(meta), data_key: "\n".join(text)}
        return [doc]
    else:
        doc = {data_key: "\n".join(contents.splitlines())}
        return [doc]


def parse_reuters_contents(contents):
    text = ""
    title = ""
    author = ""
    date = ""
    url = ""

    for i, line in enumerate(contents.split("\n")):
        if i == 0 and line.startswith("-- "):
            title = line.replace("-- ", "").strip()
        elif i == 1 and line.startswith("-- "):
            author = line.replace("-- ", "").replace("By ", "").strip()
        elif i == 2 and line.startswith("-- "):
            date = line.replace("-- ", "").strip()
        elif i == 3 and line.startswith("-- "):
            url = line.replace("-- ", "").strip()
        else:
            text += line + "\n"
    doc = {
        "title": title,
        "author": author,
        "date": date,
        "url": url,
        "text": text.strip(),
    }
    return [doc]
