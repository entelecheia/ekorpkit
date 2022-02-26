from pathlib import Path


def save_wordlist(words, file_path, sort=True):
    if sort:
        words = sorted(words)
    print('Save the list to the file: {}, no. of words: {}'.format(file_path, len(words)))
    with open(file_path, 'w') as f:
        for word in words:
            f.write(word + "\n")

def load_wordlist(file_path, rewrite=False, max_ngram=None,
                  remove_tag=False, remove_delimiter=False,
                  remove_duplicate=True, sort=True,
                  lowercase=False):
    file_path = Path(file_path)
    if file_path.is_file():
        with open(file_path) as f:
            words = [word.strip().split()[0] for word in f if len(word.strip()) > 0]
    else:
        words = []
        save_wordlist(words, file_path)

    if remove_delimiter:
        words = [word.replace(';', '') for word in words]
    if sort and remove_duplicate:
        words = sorted(set(words))
    if max_ngram:
        words = [word for word in words if len(word.split(';')) <= max_ngram]
    print('Loaded the file: {}, No. of words: {}'.format(file_path, len(words)))
    if rewrite:
        if sort:
            words = sorted(words)
        with open(file_path, 'w') as f:
            for word in words:
                f.write(word + "\n")
        print('Rewrite the words to the file: {}'.format(file_path))

    if remove_tag:
        words = [word.split('/')[0] for word in words]
    words = [word.lower() if lowercase else word
             for word in words if not word.startswith('#')]
    if remove_duplicate:
        words = set(words)
    return words
