from ekorpkit import eKonf
from ekorpkit.info.docs import make_table

from .hyfi import getLogger, hydra_main

logger = getLogger(__name__)


def listup(**args):
    ekorpkit_dir = eKonf.__ekorpkit_path__
    corpus_info = {}
    for name in args[eKonf.Keys.CORPUS]["preset"][eKonf.Keys.CORPUS]:
        info_path = f"{ekorpkit_dir}/resources/corpora/{name}.yaml"
        info = eKonf.load(info_path)
        corpus_info[name] = info
    make_table(corpus_info.values(), args["info"]["table"])


if __name__ == "__main__":
    hydra_main()
