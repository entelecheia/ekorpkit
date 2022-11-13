from ekorpkit import eKonf


def test_brancing_entropy():
    from ekorpkit.tokenizers.branching import BranchingEntropyTokenizer

    cfg = eKonf.compose("path")
    cfg.cache.uri = "https://github.com/entelecheia/ekorpkit-book/raw/main/assets/data/us_equities_news_sampled.zip"
    data = eKonf.load_data("us_equities_news_sampled.parquet", cfg.cached_path)
    texts = data.text[:100]

    bet = BranchingEntropyTokenizer()
    bet.train_from_iterator(texts, min_frequency=1, verbose=True)

    sequence = "Investment opportunities in the company."
    tokens = bet.tokenize(sequence, flatten=False, direction="forward")
    assert tokens == [
        ("▁invest", "ment"),
        ("▁opportunit", "ies"),
        ("▁in",),
        ("▁the",),
        ("▁company.",),
    ]


def test_bpe():
    from ekorpkit.tokenizers.bpe import BPETokenizer

    cfg = eKonf.compose("path")
    cfg.cache.uri = "https://github.com/entelecheia/ekorpkit-book/raw/main/assets/data/us_equities_news_sampled.zip"
    data = eKonf.load_data("us_equities_news_sampled.parquet", cfg.cached_path)
    texts = data.text[:100]

    bpe = BPETokenizer()

    bpe.train_from_iterator(texts, vocab_size=1000, verbose=500)

    tokens = bpe.tokenize("Investment opportunities in the company")

    assert tokens == [
        "investment</w>",
        "op",
        "port",
        "un",
        "ities</w>",
        "in</w>",
        "the</w>",
        "company</w>",
    ]


def test_sentencepiece():
    from ekorpkit.tokenizers.sentencepiece import SentencePieceUnigramTokenizer

    cfg = eKonf.compose("path")
    cfg.cache.uri = "https://github.com/entelecheia/ekorpkit-book/raw/main/assets/data/us_equities_news_sampled.zip"
    data = eKonf.load_data("us_equities_news_sampled.parquet", cfg.cached_path)
    texts = data.text[:100]

    vocab_size = 5000

    sp = SentencePieceUnigramTokenizer()
    sp.train_from_iterator(
        texts, vocab_size=vocab_size, min_frequency=10, initial_alphabet=[".", "!", "?"]
    )

    sequence = "Investment opportunities in the company."
    tokens = sp.tokenize(sequence, nbest_size=1)

    assert tokens == [
        "▁investment",
        "▁opportunit",
        "ies",
        "▁in",
        "▁the",
        "▁company",
        ".",
    ]
