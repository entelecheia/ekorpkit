from ekorpkit import eKonf


def test_compose_config():
    cfg = eKonf.compose()
    assert type(cfg) == dict


def test_mecab_cfg():
    config_group = "preprocessor/tokenizer=mecab"
    cfg = eKonf.compose(config_group=config_group)
    mecab = eKonf.instantiate(cfg)
    text = "IMF가 推定한 우리나라의 GDP갭률은 今年에도 소폭의 마이너스(−)를 持續하고 있다."
    tokens = mecab.tokenize(text)
    assert type(tokens) == list


def test_mecab():
    from ekorpkit.preprocessors.tokenizer import MecabTokenizer

    mecab = MecabTokenizer()
    text = "IMF가 推定한 우리나라의 GDP갭률은 今年에도 소폭의 마이너스(−)를 持續하고 있다."
    tokens = mecab.tokenize(text)
    assert type(tokens) == list


def test_kss():
    from ekorpkit.preprocessors.segmenter import KSSSegmenter

    seg = KSSSegmenter()
    text = "일본기상청과 태평양지진해일경보센터는 3월 11일 오후 2시 49분경에 일본 동해안을 비롯하여 대만, 알래스카, 하와이, 괌, 캘리포니아, 칠레 등 태평양 연안 50여 국가에 지진해일 주의보와 경보를 발령하였다. 다행히도 우리나라는 지진발생위치로부터 1,000km 이상 떨어진데다 일본 열도가 가로막아 지진해일이 도달하지 않았다. 지진해일은 일본 소마항에 7.3m, 카마이시항에 4.1m, 미야코항에 4m 등 일본 동해안 전역에서 관측되었다. 지진해일이 원해로 전파되면서 대만(19시 40분)에서 소규모 지진해일과 하와이 섬에서 1.4m(23시 9분)의 지진해일이 관측되었다. 다음날인 3월 12일 새벽 1시 57분경에는 진앙지로부터 약 7,500km 떨어진 캘리포니아 크레센트시티에서 2.2m의 지진해일이 관측되었다."
    sents = seg(text)
    assert type(sents) == list


def test_normalizer():
    from ekorpkit.preprocessors.normalizer import Normalizer

    text = "IMF가 推定한 우리나라의 GDP갭률은 今年에도 소폭의 마이너스(−)를 持續하고 있다."
    text = Normalizer().normalize(text)
    assert type(text) == str


def test_about():
    from ekorpkit.cli import about

    cfg = eKonf.compose()
    about(**cfg)
    assert True
