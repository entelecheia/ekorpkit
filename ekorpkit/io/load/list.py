'''Load and save the word list from the file.'''
from pathlib import Path


def save_wordlist(words, filepath, sort=True, verbose=True, **kwargs):
    """Save the word list to the file."""
    if sort:
        words = sorted(words)
    if verbose:
        print(f"Save the list to the file: {filepath}, no. of words: {len(words)}")
    with open(filepath, "w", encoding="utf-8") as fo_:
        for word in words:
            fo_.write(word + "\n")


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
    """Load the word list from the file."""
    filepath = Path(filepath)
    if filepath.is_file():
        with open(filepath, encoding="utf-8") as fo_:
            words = [word.strip().split()[0] for word in fo_ if len(word.strip()) > 0]
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
        print(f"Loaded the file: {filepath}, No. of words: {len(words)}")
    if rewrite:
        if sort:
            words = sorted(words)
        with open(filepath, "w", encoding="utf-8") as fo_:
            for word in words:
                fo_.write(word + "\n")
        if verbose:
            print(f"Rewrite the file: {filepath}, No. of words: {len(words)}")

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
            print(f"No. of unique words: {len(words)}")
    return words
