import os
import subprocess
import pandas as pd
from collections import namedtuple
from pathlib import Path
import logging


log = logging.getLogger(__name__)


Feature = namedtuple(
    "Feature",
    [
        "pos",
        "semantic",
        "has_jongseong",
        "reading",
        "type",
        "start_pos",
        "end_pos",
        "expression",
    ],
)
# ('합니다',
#   Feature(pos='XSA+EF', semantic=None, has_jongseong=False, reading='합니다', type='Inflect',
#           start_pos='XSA', end_pos='EF',
#           expression='하/XSA/*+ᄇ니다/EF/*')),


def _extract_feature(values):
    # Reference:
    # - http://taku910.github.io/mecab/learn.html
    # - https://docs.google.com/spreadsheets/d/1-9blXKjtjeKZqsf4NzHeYJCrr49-nXeRF6D80udfcwY
    # - https://bitbucket.org/eunjeon/mecab-ko-dic/src/master/utils/dictionary/lexicon.py

    # feature = <pos>,<semantic>,<has_jongseong>,<reading>,<type>,<start_pos>,<end_pos>,<expression>
    assert len(values) == 8

    values = [value if value != "*" else None for value in values]
    feature = dict(zip(Feature._fields, values))
    feature["has_jongseong"] = {"T": True, "F": False}.get(feature["has_jongseong"])

    return Feature(**feature)


class MeCabError(Exception):
    pass


class MeCab:  # APIs are inspried by KoNLPy
    def __init__(
        self,
        dicdir=None,
        userdic_path=None,
        backend="mecab-python3",
        verbose=False,
        **kwargs,
    ):
        import mecab_ko_dic

        self.dicdir = None
        self.userdic_path = None
        self.verbose = verbose

        DICDIR = mecab_ko_dic.DICDIR
        mecabrc = os.path.join(DICDIR, "mecabrc")

        self.backend = backend
        assert self.backend in [
            "fugashi",
            "mecab-python3",
        ], "Wrong backend! Currently, we support [`fugashi`, `mecab-python3`] backend."
        if self.verbose:
            log.info(f"MeCab uses {self.backend} as backend.")

        if not dicdir:
            dicdir = DICDIR
        self.dicdir = dicdir
        MECAB_ARGS = '-r "{}" -d "{}" '.format(mecabrc, dicdir)
        if userdic_path:
            self.userdic_path = userdic_path
            MECAB_ARGS += '-u "{}" '.format(userdic_path)
        if self.verbose:
            log.info(
                f"Mecab uses system dictionary: {dicdir}, user dictionary: {userdic_path}"
            )
        try:
            if self.backend == "fugashi":
                try:
                    import fugashi as _mecab
                except ImportError:
                    raise ImportError(
                        "\n"
                        "You must install `fugashi` if you want to use `fugashi` backend.\n"
                    )
                self.tagger = _mecab.GenericTagger(MECAB_ARGS)
                self.dictionary_info = self.tagger.dictionary_info
                self.sysdic_path = self.dictionary_info[0]["filename"]
            else:
                try:
                    import MeCab as _mecab
                except ImportError:
                    raise ImportError(
                        "\n"
                        "You must install `mecab-python3` if you want to use `mecab-python3` backend.\n"
                    )
                self.tagger = _mecab.Tagger(MECAB_ARGS)
                self.version = _mecab.VERSION
        except RuntimeError:
            raise Exception(
                'The MeCab dictionary does not exist at "%s". Is the dictionary correctly installed?\nYou can also try entering the dictionary path when initializing the MeCab class: "MeCab(\'/some/dic/path\')"'
                % self.dicdir
            )
        except NameError:
            raise Exception("Check if MeCab is installed correctlly.")

    def _parse_fugashi(self, text):
        return [
            (node.surface, _extract_feature(node.feature)) for node in self.tagger(text)
        ]

    def _parse_mecab(self, text):
        m = self.tagger.parseToNode(text)
        nodes = []
        while m:
            feature = m.feature.split(",")
            if "BOS/EOS" != feature[0]:
                nodes.append((m.surface, _extract_feature(feature)))
            m = m.next
        return nodes

    def parse(self, text):
        if self.backend == "fugashi":
            return self._parse_fugashi(text)
        else:
            return self._parse_mecab(text)

    def pos(
        self,
        sentence,
        flatten=True,
        concat_surface_and_pos=False,
        include_whitespace_token=True,
    ):
        sentence = sentence.strip()
        if include_whitespace_token:
            sent_ptr = 0
            res = []

            for token, pos in self._pos(sentence, flatten=flatten):
                if sentence[sent_ptr] == " ":
                    # Move pointer to whitespace token to reserve whitespace
                    # cf. to prevent double white-space, move pointer to next eojeol
                    while sentence[sent_ptr] == " ":
                        sent_ptr += 1
                    res.append((" ", "SP"))
                res.append((token, pos))
                sent_ptr += len(token)
        else:
            res = self._pos(sentence, flatten=flatten)

        return [s[0] + "/" + s[1] if concat_surface_and_pos else s for s in res]

    def _pos(self, sentence, flatten=True):
        if flatten:
            return [(surface, feature.pos) for surface, feature in self.parse(sentence)]
        else:
            res = []
            for surface, feature in self.parse(sentence):
                if feature.expression is None:
                    res.append((surface, feature.pos))
                else:
                    for elem in feature.expression.split("+"):
                        s = elem.split("/")
                        res.append((s[0], s[1]))
            return res

    def morphs(self, sentence, flatten=True, include_whitespace_token=True):
        return [
            surface
            for surface, _ in self.pos(
                sentence,
                flatten=flatten,
                concat_surface_and_pos=False,
                include_whitespace_token=include_whitespace_token,
            )
        ]

    def nouns(
        self,
        sentence,
        flatten=True,
        include_whitespace_token=True,
        noun_pos=["NNG", "NNP", "XSN", "SL", "XR", "NNB", "NR"],
    ):
        return [
            surface
            for surface, pos in self.pos(
                sentence,
                flatten=flatten,
                concat_surface_and_pos=False,
                include_whitespace_token=include_whitespace_token,
            )
            if pos in noun_pos
        ]


DicEntry = namedtuple(
    "DicEntry",
    [
        "surface",
        "left_id",
        "right_id",
        "cost",
        "pos",
        "semantic",
        "has_jongseong",
        "reading",
        "type",
        "start_pos",
        "end_pos",
        "expression",
    ],
    defaults=[None, None, None, None, "NNP", "*", "T", None, "*", "*", "*", "*"],
)

ContextEntry = namedtuple(
    "ContextEntry",
    [
        "id",
        "pos",
        "semantic",
        "has_jongseong",
        "reading",
        "type",
        "start_pos",
        "end_pos",
        "expression",
    ],
    defaults=[None, "*", "*", "*", "*", "*", "*", "*"],
)


def iternamedtuples(df):
    Row = namedtuple("Row", df.columns)
    for row in df.itertuples():
        yield Row(*row[1:])


def has_jongseong(c):
    return int((ord(c[-1]) - 0xAC00) % 28) != 0


class MecabDicConfig:
    def __init__(self, userdic_path=None):
        import mecab_ko_dic

        if userdic_path:
            self.load_userdic(userdic_path)
        else:
            self.userdic = {}
        self.dicdir = mecab_ko_dic.DICDIR
        self.left_ids = self.load_context_ids("left-id.def")
        self.right_ids = self.load_context_ids("right-id.def")

    def load_context_ids(self, id_file):
        id_file = os.path.join(self.dicdir, id_file)
        context_ids = []
        with open(id_file, "r", encoding="utf-8") as f:
            for line in f:
                id, vals = line.split()
                entry = ContextEntry(id, *vals.split(","))
                context_ids.append(entry)
        return context_ids

    def find_left_context_id(self, search):
        for entry in self.left_ids:
            if entry.pos == search.pos and entry.semantic == search.semantic:
                return entry.id

    def find_right_context_id(self, search):
        for entry in self.right_ids:
            if (
                entry.pos == search.pos
                and entry.semantic == search.semantic
                and entry.has_jongseong == search.has_jongseong
            ):
                return entry.id

    def load_userdic(self, userdic_path):
        userdic_path = Path(userdic_path)

        if userdic_path.is_dir():
            self.userdic = {}
            for f in userdic_path.glob("*.csv"):
                df = pd.read_csv(f, names=DicEntry._fields)
                dic = {e.surface: e for e in iternamedtuples(df)}
                self.userdic = {**self.userdic, **dic}
        else:
            df = pd.read_csv(userdic_path, names=DicEntry._fields)
            self.userdic = {e.surface: e for e in iternamedtuples(df)}
        print(" No. of user dictionary entires loaded: %d" % len(self.userdic))

    def add_entry_to_userdic(
        self, surface, pos="NNP", semantic="*", reading=None, cost=1000
    ):
        entry = DicEntry(
            surface=surface,
            cost=cost,
            pos=pos,
            semantic=semantic,
            has_jongseong={True: "T", False: "F"}.get(has_jongseong(surface)),
            reading=surface if reading is None else reading,
        )
        entry = entry._replace(
            left_id=self.find_left_context_id(entry),
            right_id=self.find_right_context_id(entry),
        )
        self.userdic[surface] = entry

    def adjust_context_ids(self):
        for entry in self.userdic.values():
            entry = entry._replace(
                left_id=self.find_left_context_id(entry),
                right_id=self.find_right_context_id(entry),
            )
            self.userdic[entry.surface] = entry

    def adjust_costs(self, cost=1000):
        for surface, entry in self.userdic.items():
            self.userdic[surface] = entry._replace(cost=cost)

    def save_userdic(self, save_path):
        if len(self.userdic) > 0:
            df = pd.DataFrame(self.userdic.values())
            df.to_csv(save_path, header=False, index=False)
            self.userdic_path = save_path
            print("Saved the userdic to {}".format(save_path))
        else:
            print("No userdic to save...")

    def build_userdic(self, built_userdic_path, userdic_path=None):
        if userdic_path:
            self.userdic_path = userdic_path
        args = '-d "{}" -u "{}" {}'.format(
            self.dicdir, built_userdic_path, self.userdic_path
        )
        # print(args)
        subprocess.run(["fugashi-build-dict", args])
