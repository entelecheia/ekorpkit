import os
import subprocess
import orjson as json
import pandas as pd
from multiprocessing import Pool
from ekorpkit import eKonf
from glob import glob
from ekorpkit.io.fetch.web import web_download, gdrive_download_un7z


class Wiki:
    def __init__(self, lang="en", output_dir=None, **args):
        cfg = eKonf.compose("io/fetcher=wiki")
        cfg.lang = lang
        if output_dir:
            cfg.output_dir = output_dir
        self.args = eKonf.merge(cfg, args)
        self.name = self.args.name
        self.autoload = self.args.get("autoload", True)
        self.url = self.args.dump.url
        self.output_dir = self.args.output_dir
        self.dump_dir = self.args.dump.dump_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dump_dir, exist_ok=True)
        if self.args.output_file:
            self.output_file = os.path.join(self.output_dir, self.args.output_file)
        self.dump_file = os.path.join(self.dump_dir, self.args.dump.dump_file)
        self.force_download = self.args.force_download

        if self.autoload:
            self.fetch()

    def fetch(self):
        if not os.listdir(self.output_dir) or self.force_download:
            self.download_dump()
            if self.args.extract._target_ == "extract_wiki":
                self.extract_wiki()
            elif self.args.extract._target_ == "extract_namuwiki":
                self.extract_namuwiki()
            else:
                raise Exception("Invalid target")

    def extract_namuwiki(self):
        extracted_dir = self.dump_file[:-3]
        if os.path.exists(extracted_dir):
            json_file = glob(extracted_dir + "/*.json")[0]
        else:
            raise Exception("Extracted json file doesn't exist")

        with open(json_file, "r", encoding="utf-8") as input_file:
            namu_wiki = json.load(input_file)

        with Pool(processes=self.args.num_workers) as pool:
            documents = pool.map(work, namu_wiki)
        print(f"Extracted {self.name} from dump file {self.dump_file}")

        df = pd.DataFrame(documents)
        print(df.tail())
        df.to_csv(self.output_file, header=True, index=False, encoding="utf-8")
        print(f"Saved {self.name} to {self.output_file}")

    def extract_wiki(self):
        command = f"python -m wikiextractor.WikiExtractor {self.dump_file} --json --output {self.output_dir}"
        if self.args.num_workers > 1:
            command += f" --processes {self.args.num_workers}"
        if self.args.compress:
            command += " --compress"
        subprocess.run(command, shell=True)
        print(f"Extracted {self.name} from dump file {self.dump_file}")

    def download_dump(self):
        if self.args.dump._target_ == "web_download":
            web_download(self.url, self.dump_file, self.name, self.force_download)
        elif self.args.dump._target_ == "gdrive_download_un7z":
            gdrive_download_un7z(
                self.url, self.dump_file, self.name, self.force_download
            )
        else:
            raise Exception("Invalid target")


def work(document):
    from namuwiki.extractor import extract_text

    return {"title": document["title"], "text": extract_text(document["text"])}
