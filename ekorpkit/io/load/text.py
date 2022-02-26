def load_text(path, num_heads=0, num_samples=-1):
    lines = []
    with open(path, encoding="utf-8") as f:
        if num_heads > 0:
            for _ in range(num_heads):
                next(f)
        for i, line in enumerate(f):
            if (num_samples > 0) and (i >= num_samples):
                break
            lines.append(line.rstrip("\n"))
    return lines


def load_parallel_text(source_path, target_path, num_heads=0, num_samples=-1):
    sources = load_text(source_path, num_heads, num_samples)
    targets = load_text(target_path, num_heads, num_samples)
    if len(sources) != len(targets):
        raise ValueError("Parallel corpus must have same length two files")
    return sources, targets


def load_wikitext(path, num_lines=-1):
    """
    Wikitext format

         = Head1 =

        text ...
        text ...

         = = 2ead = =

        text ...
        text ...
    """
    if num_lines <= 0:
        with open(path, encoding="utf-8") as f:
            texts = f.read().split("\n =")
    else:
        lines = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                lines.append(line)
        texts = "".join(lines).split("\n =")
    # fix missing prefix
    texts = [texts[0]] + [f" ={text}" for text in texts[1:]]
    return texts
