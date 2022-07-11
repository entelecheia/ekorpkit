"""the_pile dataset"""

import io
import os
import pandas as pd
from ekorpkit import eKonf
from ekorpkit.io.fetch.web import web_download
from tqdm.auto import tqdm

try:
    import simdjson as json
except ImportError:
    print("Installing simdjson library")
    os.system("pip install -q pysimdjson")
    import json as json


parser = json.Parser()


def json_parser(x):
    try:
        line = parser.parse(x).as_dict()
        return line
    except ValueError:
        return x


class PileReader:
    def __init__(self, filenames, subset=None, segment_separator="\n\n"):
        if not isinstance(filenames, list):
            filenames = [filenames]
        self.filenames = filenames
        self.subset = subset
        self.segment_separator = segment_separator

    def _read_fn(self):
        import zstandard
        import jsonlines

        for filename in self.filenames:
            print(f"iterating over {filename}")
            with open(filename, "rb") as f:
                cctx = zstandard.ZstdDecompressor()
                reader_stream = io.BufferedReader(cctx.stream_reader(f))
                reader = jsonlines.Reader(reader_stream, loads=json_parser)
                for i, item in enumerate(reader):
                    result = dict()
                    if isinstance(item, str):
                        result["text"] = item
                        result["subset"] = "the_pile"
                    else:
                        text = item["text"]
                        if isinstance(text, list):
                            text = self.segment_separator.join(text)
                        result["text"] = text
                        result["subset"] = item.get("meta", {}).get(
                            "pile_set_name", "the_pile"
                        )
                    if self.subset is None:
                        yield result
                    else:
                        if self.subset == result["subset"]:
                            yield result

    def __iter__(self):
        return self._read_fn()


class ThePile:
    def __init__(self, **args):
        self.cfg = eKonf.to_config(args)
        self.name = self.cfg.name
        self.subsets = self.cfg.get("subsets", [])
        if self.name == "the_pile":
            self.subset = None
        else:
            self.subset = self.name
            if self.subset not in self.subsets:
                raise ValueError(f"{self.subset} not in {self.subsets}")
        self._parse_split_urls()

    def _parse_split_urls(self):
        self.splits = {}
        for split, info in self.cfg.data_sources.items():
            if info.get("splits", None):
                urls = [
                    info["url"].format(str(i).zfill(info.zfill))
                    for i in range(info.splits)
                ]
            else:
                urls = [info["url"]]
            paths = {}
            for url in urls:
                path = os.path.join(self.cfg.data_dir, url.split("/")[-1])
                paths[path] = url
            self.splits[split] = paths

    def download(self):
        for split, paths in self.splits.items():
            for path, url in paths.items():
                print(f"Downloading {split} from {url} to {path}")
                web_download(url, path, self.name)

    def load(self, split="train"):
        paths = list(self.splits[split].keys())
        return self._generate_examples(paths)

    def _generate_examples(self, paths):
        pipeline = PileReader(paths, self.subset)
        for result in pipeline:
            if result:
                yield result


def load_pile_data(split_name, **args):
    pile = ThePile(**args)

    ds_iter = pile.load(split_name)
    documents = []
    for sample in tqdm(ds_iter):
        documents.append(sample)
    df = pd.DataFrame(documents)
    return df
