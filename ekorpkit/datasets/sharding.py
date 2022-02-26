import subprocess
import omegaconf
from hydra.utils import instantiate
from pathlib import Path


def _load_dataset(
    data_files,
    extension="json",
    test_split_percentage=10,
    min_tokens=5,
    download_mode="reuse_dataset_if_exists",
    num_proc=1,
):
    from datasets import load_dataset

    dataset = {}
    if test_split_percentage > 0:
        dataset["test"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{test_split_percentage}%]",
            download_mode=download_mode,
        )
        dataset["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{test_split_percentage}%:]",
        )
    else:
        dataset = load_dataset(
            extension, data_files=data_files, download_mode=download_mode
        )
    for name in dataset:
        n_proc = min(num_proc, len(dataset[name]) - 1)
        cols = dataset[name].column_names
        cols.remove("text")
        if len(cols) > 0:
            dataset[name] = dataset[name].remove_columns(cols)
        dataset[name] = dataset[name].filter(
            lambda x: len(x["text"].split()) > min_tokens, num_proc=n_proc
        )
        # print(dataset[name]['text'][0])
        print(
            "Loading Datasets: There are",
            len(dataset[name]),
            f"records in {name} dataset.",
            data_files,
        )
    return dataset


class ShardingDataset:
    def __init__(
        self, corpus, output, dataset, n_shards, n_processes, segmenter, **args
    ):

        self.input_extension = corpus["corpus_filetype"]
        corpus_files = corpus["corpus_files"]
        corpus_dir = corpus["corpus_dir"]
        if corpus_files:
            print(type(corpus_files))
            if isinstance(corpus_files, str):
                corpus_files = {corpus_files: 1}
            elif (
                isinstance(corpus_files, list)
                or type(corpus_files) == omegaconf.listconfig.ListConfig
            ):
                corpus_files = {f: 1 for f in corpus_files}
            else:
                corpus_files = dict(corpus_files)
            corpus_files = {corpus_dir + "/" + f: d for f, d in corpus_files.items()}
        else:
            corpus_files = {
                str(f): 1 for f in Path(corpus_dir).glob("**/*") if f.is_file()
            }
        assert (
            len(corpus_files) > 0
        ), "The input file list must contain at least one file."
        self.input_files = corpus_files

        assert n_shards["train"] > 0, "There must be at least one output shard."
        # assert n_test_shards > 0, 'There must be at least one output shard.'
        self.n_shards = n_shards

        self.download_mode = dataset["download_mode"]
        self.fraction_test_set = dataset["fraction_test_set"]
        if not self.fraction_test_set > 0:
            self.n_shards["test"] = 0
        self.shuffle_dataset = dataset["shuffle_dataset"]
        self.seed = dataset["seed"]
        self.min_tokens = dataset["min_tokens"]

        if n_processes:
            self.n_processes = n_processes
        else:
            self.n_processes = 7
        self.segmenter = segmenter

        self.output_dir = Path(output["output_dir"])
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_name_prefix = output["name_prefix"]
        self.output_identifier = output["identifier"]
        self.output_file_extension = output["file_extension"]

        self.datasets = {}  # key: split name, value: dataset
        self.output_files = {
            "train": {},
            "test": {},
        }  # key: filename, value: list of articles to go into file

        self.init_output_files()

    # Remember, the input files contain one article per line (the whitespace check is to skip extraneous blank lines)
    def load_datasets(self):
        from datasets import concatenate_datasets

        print("Start: Loading Datasets")

        datasets = {"train": [], "test": []}
        for input_file, dupe_factor in self.input_files.items():
            print(f"processing {input_file} with {dupe_factor} dupe factor")
            dset = _load_dataset(
                input_file,
                extension=self.input_extension,
                test_split_percentage=int(self.fraction_test_set * 100),
                min_tokens=self.min_tokens,
                download_mode=self.download_mode,
                num_proc=self.n_processes,
            )
            # self.datasets[input_file] = dset
            for name in dset:
                for i in range(dupe_factor):
                    datasets[name].append(dset[name])
                    print(f"{i}: adding {len(dset[name])} records in {name} dataset.")
        for name in datasets:
            if len(datasets[name]) > 0:
                self.datasets[name] = concatenate_datasets(datasets[name])
                print(f"Concatenating {len(datasets[name])} {name} datasets.")
        if self.shuffle_dataset:
            for name in self.datasets:
                print(f"Shuffling {name} dataset.")
                self.datasets[name] = self.datasets[name].shuffle(self.seed)

        print("End: Loading Datasets.")

    def segment_articles_into_sentences(self):
        if not self.datasets:
            self.load_articles()

        assert (
            len(self.datasets["train"]) > 0
        ), "Please check that input files are present and contain data."
        assert self.segmenter[
            "_target_"
        ], "Please check that the segmenter is configured correctly."
        print("Start: Sentence Segmentation")

        seg = instantiate(self.segmenter)

        def segment_into_sentences(examples):
            sentences = []
            for text in examples["text"]:
                sentences += seg.segment_article(text.replace("\t", " "))
                sentences.append("")
            return {"text": sentences}

        num_articles = 0
        num_sentences = 0
        for name in self.datasets:
            num_articles += len(self.datasets[name])
            num_proc = min(self.n_processes, len(self.datasets[name]) - 1)
            self.datasets[name] = self.datasets[name].map(
                segment_into_sentences, batched=True, num_proc=num_proc
            )
            num_sentences += len(self.datasets[name])

        print(
            f"Number of articles: {num_articles}, Number of sentences: {num_sentences}"
        )
        print("End: Sentence Segmentation")

    def init_output_files(self):
        print("Start: Init Output Files")
        for name in self.output_files:
            assert (
                len(self.output_files[name]) == 0
            ), "Internal storage self.output_files already contains data."

            for i in range(self.n_shards[name]):
                filename = (
                    self.output_name_prefix
                    + self.output_identifier[name]
                    + "_"
                    + str(i)
                    + self.output_file_extension
                )
                self.output_files[name][filename] = None

        print("End: Init Output Files")

    def distribute_datasets_over_shards(self):
        print("Start: Distribute Datasets Over Shards")
        for name in self.datasets:
            assert (
                len(self.datasets[name]) > self.n_shards[name]
            ), "There are fewer articles than shards. Please add more data or reduce the number of shards requested."

            for i, filename in enumerate(self.output_files[name]):
                self.output_files[name][filename] = self.datasets[name].shard(
                    self.n_shards[name], i, contiguous=True
                )

            for shard in self.output_files[name]:
                print(f"{name} shard:", shard, len(self.output_files[name][shard]))

        print("End: Distribute Datasets Over Shards")

    def write_shards_to_disk(self):
        print("Start: Write Shards to Disk")
        for name in self.output_files:
            for shard in self.output_files[name]:
                self.write_single_shard(shard, self.output_files[name][shard])

        print("End: Write Shards to Disk")
        for split, n in self.n_shards.items():
            if n > 0:
                if not (self.output_dir / split).is_dir():
                    (self.output_dir / split).mkdir(exist_ok=True, parents=True)
                absolute_dir = str(self.output_dir)
                command = (
                    "mv "
                    + absolute_dir
                    + "/*"
                    + split
                    + "*.txt"
                    + " "
                    + absolute_dir
                    + "/"
                    + split
                )
                print(command)
                mv_process = subprocess.Popen(command, shell=True)

                mv_process.wait()

    def write_single_shard(self, shard_name, shard_dataset):
        with open(shard_name, mode="w", newline="\n") as f:
            for text in shard_dataset["text"]:
                f.write(text + "\n")
