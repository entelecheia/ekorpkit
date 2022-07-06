import abc
import collections
import orjson as json
import os
import time
import random

from datasets import load_dataset
from pytablewriter import MarkdownTableWriter
from tqdm.auto import tqdm

from ekorpkit.utils.func import humanbytes, parse_size, utf8len

from ekorpkit import eKonf
from ekorpkit.utils.func import change_directory
from hydra.utils import instantiate

from tqdm.auto import tqdm
import zstandard


class Archive:
    def __init__(self, output_dir, compression_level=3):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.i = 0

        self.fh = open(self.output_dir + "/current_chunk_incomplete", "wb")
        self.cctx = zstandard.ZstdCompressor(level=compression_level, threads=8)
        self.compressor = self.cctx.stream_writer(self.fh)

    def add_data(self, data, meta={}):
        self.compressor.write(
            json.dumps({"text": data, "meta": meta}).encode("UTF-8") + b"\n"
        )

    def commit(self, archive_name="default"):
        fname = (
            self.output_dir
            + "/data_"
            + str(self.i)
            + "_time"
            + str(int(time.time()))
            + "_"
            + archive_name
            + ".jsonl.zst"
        )
        self.compressor.flush(zstandard.FLUSH_FRAME)

        self.fh.flush()
        self.fh.close()
        os.rename(self.output_dir + "/current_chunk_incomplete", fname)
        self.fh = open(self.output_dir + "/current_chunk_incomplete", "wb")
        self.compressor = self.cctx.stream_writer(self.fh)

        self.i += 1


def dummy_meta(xs):
    return ((x, {}) for x in xs)


class Dataset(abc.ABC):
    @abc.abstractmethod
    def name(self):
        """Human-readable name of the dataset"""
        pass

    @abc.abstractmethod
    def documents(self):
        """A generator producing all documents in the dataset."""
        pass

    @abc.abstractmethod
    def clean(self):
        """Remove any dataset files."""
        pass

    def size(self):
        """Return an estimate of the dataset size. Implementations may use a faster, less accurate estimate."""

        size = sum(map(utf8len, tqdm(self.documents())))
        print("size", self.name(), size)
        return size

    def num_docs(self):
        """Return an estimate of the number of documents in the dataset. Implementations may use a faster, less accurate estimate."""

        size = len(list(map(lambda x: None, tqdm(self.documents()))))
        print("docs", self.name(), size)
        return size

    def shuffled(self):
        """Datasets where the source is already shuffled should override this to return True so that it isn't shuffled again."""
        return False


class HFDS(Dataset):
    def __init__(self, **args):
        self.name = args["name"]
        self.path = args["path"]
        self.subset = args["subset"]
        self.split = args[eKonf.Keys.SPLIT]
        self.download_mode = args["download_mode"]
        self.cache_dir = args["cache_dir"]
        self.streaming = args["streaming"]

    def name(self):
        return self.name

    def _load(self):
        self.dataset = load_dataset(
            self.name, self.subset, download_mode=self.download_mode
        )

    def documents(self):
        self._load()

        for doc in self.dataset:
            yield doc

    def clean(self):
        if not self.streaming:
            self.dataset.cleanup_cache_files()


def take(n, iter):
    ret = []
    for i in range(n):
        try:
            ret.append(next(iter))
        except StopIteration:
            break
    return ret


def mk_table(datasets, train_chars, latex=None, print_latex=False):
    values = []

    total_weight = sum([x[1] * x[0].size() for x in datasets])

    for dataset, weight in datasets:
        size = dataset.size()
        relative_weight = size * weight / total_weight
        values.append(
            [
                dataset.name(),
                size,
                "{:.2%}".format(relative_weight),
                "{:.4f}".format(train_chars / size * relative_weight),
                size * weight,
                humanbytes(size / dataset.num_docs(), "KiB"),
            ]
        )

    values.sort(key=lambda x: -x[4])
    values.append(
        [
            "**Total**",
            "",
            "",
            "",
            sum([x[4] for x in values]),
            humanbytes(
                sum([x[1] for x in values]) / sum(x[0].num_docs() for x in datasets),
                "KiB",
            ),
        ]
    )
    values = [
        [
            x[0],
            humanbytes(x[1], "GiB") if x[1] else "",
            x[2],
            x[3],
            humanbytes(x[4], "GiB"),
            x[5],
        ]
        for x in values
    ]

    writer = MarkdownTableWriter()
    writer.table_name = "The Pile™"
    writer.headers = [
        "Component",
        "Raw Size",
        "Weight",
        "Epochs",
        "Effective Size",
        "Mean Document Size",
    ]
    writer.value_matrix = values

    if print_latex:
        rows = []
        for row in values[:-1]:
            rows.append(
                "        "
                + " & ".join(map(lambda x: str(x).replace("%", r"\%"), row))
                + r" \\"
            )
        totalrow = (
            " & ".join(
                map(
                    lambda x: r"\textbf{%s}" % str(x).replace("%", r"\%") if x else "",
                    values[-1][1:],
                )
            )
            + r" \\"
        )
        latex = latex.format(rows="\n".join(rows), totalrow=totalrow)
        print(latex)
    return writer.dumps()


def dataset_tqdm(dset):
    if isinstance(dset, LMDS):
        return dset.documents()
    pbar = tqdm(total=dset.size(), unit="B", unit_scale=True, unit_divisor=1024)
    for doc in dset.documents():
        pbar.update(utf8len(doc))
        yield doc


class Profiler:
    def __init__(self, profile):
        self.i = 0
        self.profile = profile
        self.time_per_dataset = collections.defaultdict(lambda: [0, 0])

    def measured_next(self, name, iter):
        if not self.profile:
            # no-op
            return next(iter)
        else:
            self.i += 1
            start = time.time()
            doc = next(iter)
            elapsed = time.time() - start

            self.time_per_dataset[name][0] += elapsed
            self.time_per_dataset[name][1] += 1

            if self.i % 100000 == 0:
                times = [
                    (dsname, total, ct)
                    for dsname, (total, ct) in self.time_per_dataset.items()
                ]
                times.sort(key=lambda x: x[1])
                for name, total, ct in times:
                    print(
                        name.ljust(22),
                        "{:.8f}".format(total / ct),
                        str(ct).rjust(8),
                        "{:.4f}".format(total),
                    )

            return doc


class LMDS(Dataset):
    def __init__(self, datasets, dataset_bytes, profile=False):
        self.datasets = datasets
        self.dataset_bytes = dataset_bytes
        self.profile = profile
        self.rnd = random.Random(42)

    def name(self):
        return "Custom Pile"

    def documents(self):
        datasets = []
        weights = []

        # calculate relative_weight for each
        total_weight = sum([x[1] * x[0].num_docs() for x in self.datasets])
        for dataset, weight in self.datasets:
            size = dataset.size()
            relative_weight = weight * dataset.num_docs() / total_weight
            datasets.append((dataset.name(), dataset))
            weights.append(relative_weight)

        # yield from dataset until right number of bytes
        total_bytes = 0
        pbar = tqdm(
            total=self.dataset_bytes, unit="B", unit_scale=True, unit_divisor=1024
        )

        profiler = Profiler(profile=self.profile)
        while True:
            chunk = self.rnd.choices(population=datasets, weights=weights, k=1000)
            for name, dset in chunk:
                doc, meta = profiler.measured_next(name, dset)

                size = utf8len(doc)
                total_bytes += size
                pbar.update(size)

                meta["pile_set_name"] = name

                yield doc, meta

                if total_bytes > self.dataset_bytes:
                    return

    def clean(self):
        for dataset, _ in self.datasets:
            dataset.clean()

    def size(self):
        return self.dataset_bytes


class LimitedDataset(Dataset):
    def __init__(self, dataset, limit_size):
        self.dataset = dataset
        self.limit_size = limit_size
        self.rnd = random.Random(42)

    def name(self):
        return self.dataset.name() + " (truncated)"

    def documents(self):
        numer = self.limit_size
        denom = self.dataset.size()
        for doc, meta in dataset_tqdm(self.dataset):
            docsize = utf8len(doc)
            if self.rnd.random() < numer / denom:
                yield doc, meta
                numer -= docsize
            denom -= docsize

            if numer <= 0 or denom <= 0:
                break

    def clean(self):
        self.dataset.clean()

    def size(self):
        return self.limit_size


def sample_from_sets(datasets, n_docs):
    random.seed(42)
    for dset, _ in datasets:
        print(dset.name())
        fname = "dataset_samples/{}.json".format(dset.name().replace(" ", "_"))
        if os.path.exists(fname):
            continue

        n = dset.num_docs()

        indices = set(random.sample(range(n), n_docs))
        pbar = tqdm(total=n_docs)

        docs = []
        for i, (doc, meta) in enumerate(dset.documents()):
            if i > max(indices):
                break
            if i in indices:
                docs.append((doc, meta))
                pbar.update(1)

        try:
            os.mkdir("dataset_samples")
        except:
            pass

        with open(fname, "w") as fh:
            json.dump(docs, fh)

        pbar.close()


def build_pile(**args):
    args = eKonf.to_config(args)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    with change_directory(args.output_dir):
        main(args)


def main(args):

    if args.datasets:
        datasets = []
        for dataset in args.datasets:
            dataset = instantiate(dataset)
            datasets.append((dataset, 1.0))

    random.seed(42)

    # if args.using != 'pile_reprod_no_cc':
    #     # add CC
    #     datasets.append((CommonCrawlDataset(), 1.))

    if args.read_amount is None:
        args.read_amount = sum([ds.size() * epochs for ds, epochs in datasets])
    else:
        args.read_amount = parse_size(args.read_amount)

    print(mk_table(datasets, args.read_amount))

    lmd = LMDS(datasets, args.read_amount, profile=args.profile)

    if args.force_download:
        for dset, _ in datasets:
            dset._download()

    if args.limit:
        size_limit = parse_size(args.limit)
        lmd = LimitedDataset(lmd, size_limit)

    if args.make_lmd:
        assert not (args.interleave_output and args.chunk)  # can't chunk and interleave

        if args.interleave_output:
            ars = [
                Archive("pile_pass1/chunk{}".format(i))
                for i in range(args.interleave_output)
            ]
        else:
            ar = Archive("pile_output")

        if args.chunk:
            chunk_size = parse_size(args.chunk)

        cursize = 0
        for doc, meta in lmd.documents():
            if args.interleave_output:
                ar = random.choice(ars)

            ar.add_data(doc, meta)

            cursize += len(doc)
            if args.chunk and cursize > chunk_size:
                # interleave will not be on
                cursize = 0
                ar.commit(archive_name=args.using)

        if args.interleave_output:
            for ar in ars:
                ar.commit(archive_name=args.using)
        else:
            ar.commit(archive_name=args.using)

    if args.make_dataset_samples:
        sample_from_sets(datasets, args.make_dataset_samples)
