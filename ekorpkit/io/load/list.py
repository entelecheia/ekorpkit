from pathlib import Path


def save_wordlist(words, filepath, sort=True, verbose=True, **kwargs):
    if sort:
        words = sorted(words)
    if verbose:
        print(
            "Save the list to the file: {}, no. of words: {}".format(
                filepath, len(words)
            )
        )
    with open(filepath, "w") as f:
        for word in words:
            f.write(word + "\n")


def load_wordlist(
    filepath,
    rewrite=False,
    max_ngram=None,
    remove_tag=False,
    remove_delimiter=False,
    remove_duplicate=True,
    sort=True,
    lowercase=False,
    verbose=True,
    **kwargs,
):
    filepath = Path(filepath)
    if filepath.is_file():
        with open(filepath) as f:
            words = [word.strip().split()[0] for word in f if len(word.strip()) > 0]
    else:
        words = []
        save_wordlist(words, filepath, verbose=verbose)

    if remove_delimiter:
        words = [word.replace(";", "") for word in words]
    if sort and remove_duplicate:
        words = sorted(set(words))
    if max_ngram:
        words = [word for word in words if len(word.split(";")) <= max_ngram]
    if verbose:
        print("Loaded the file: {}, No. of words: {}".format(filepath, len(words)))
    if rewrite:
        if sort:
            words = sorted(words)
        with open(filepath, "w") as f:
            for word in words:
                f.write(word + "\n")
        if verbose:
            print("Rewrite the file: {}, No. of words: {}".format(filepath, len(words)))

    if remove_tag:
        words = [word.split("/")[0] for word in words]
    words = [
        word.lower() if lowercase else word
        for word in words
        if not word.startswith("#")
    ]
    if remove_duplicate:
        words = list(set(words))
        if verbose:
            print("Remove duplicated words, No. of words: {}".format(len(words)))
    return words
