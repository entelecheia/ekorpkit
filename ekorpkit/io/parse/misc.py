def html_to_json(contents):
    import html_to_json

    doc = {}
    output_json = html_to_json.convert(contents)
    for k, v in output_json.items():
        if isinstance(v, list) and len(v) == 1:
            doc[k] = v[0].get("_value")
    return [doc]


def parse_plaintext(contents, split=False):
    if split:
        docs = []
        for i, line in enumerate(contents.split("\n")):
            doc = {"lineno": i, "text": line}
            if i < 2:
                print(f"processing line {i}")
                print(doc)
            elif i % 10000 == 0:
                print(f"processing line {i}")
            docs.append(doc)
        return docs
    else:
        doc = {"text": contents.strip()}
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
