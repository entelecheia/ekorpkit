import os
import re
import pysbd
from pathlib import Path
import pandas as pd
from ekorpkit import eKonf
from glob import glob


class Pathobook:
    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        self.input_path = self.args.input_path
        self.output_dir = Path(self.args.output_dir)
        self.chapter_dir = self.output_dir / "chapters"
        self.chapter_dir.mkdir(parents=True, exist_ok=True)
        self.norm_dir = self.output_dir / "normalized"
        self.norm_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / self.args.output_file
        # self.download()
        self.chapter_info = pd.read_csv(self.args.chapter_info_path, index_col=None)
        # self.normalizer = instantiate(self.args.normalize)

        if os.path.exists(self.output_file) and not self.args.force_download:
            print(f"output file {self.output_file} already exists, skipping")
        else:
            self.build()

    def build(self):
        self.split_chapters()
        self.make_clean_corpus()

    def split_chapters(self):
        contents = ""
        filepaths = glob(self.input_path, recursive=True)
        filepaths = [fp for fp in filepaths if Path(fp).is_file()]
        filepaths = sorted(filepaths)
        print(filepaths)
        self.norm_files = []
        for fp in filepaths:
            with open(fp, "r") as f:
                contents += f.read()

        print(contents[:100])
        self.chapters = {}
        self.chapter_files = {}
        for i, row in self.chapter_info.iterrows():
            str_ch, str_beg, str_end = row[["Chapter", "Beginning", "Ending"]]
            # str_beg = self.normalizer.normalize(str_beg)
            # str_end = self.normalizer.normalize(str_end)
            pos_beg = contents.find(str_beg)
            pos_end = contents.find(str_end)
            print(i + 1, pos_beg, pos_end)
            if pos_beg == -1:
                print(str_ch, str_beg)
            if pos_end == -1:
                print(str_ch, str_end)
            if pos_beg > -1:
                if pos_end > -1:
                    ch_txt = contents[pos_beg : pos_end + len(str_end) + 1]
                else:
                    ch_txt = contents[pos_beg:]
                filename = "chapter_{:0>2d}.txt".format(str_ch)
                self.chapters[str_ch] = ch_txt
                self.chapter_files[str_ch] = filename
                print(f"{filename}: {len(ch_txt)}")

            if pos_beg == -1 and pos_end == -1:
                print("failed to find chapter {}".format(str_ch))
                print(str_ch, str_beg, str_end)

    def make_clean_corpus(self):
        seg = pysbd.Segmenter(language="en", clean=False)
        pattern_remove_starting_with = "^\. ?"
        # pattern_split_para = '^\s*[\w+]+[\w+\s]*[^.:]$'
        pattern_split_para = "\n{2,}"
        # pattern_replace_hyphen = '(?<=\w)- '
        documents = []
        for chapter, doc in self.chapters.items():
            filename = self.chapter_files[chapter]
            print("processing {}".format(filename))

            # doc = re.sub(pattern_replace_hyphen, '-', doc)
            paras = re.split(pattern_split_para, doc, maxsplit=0, flags=re.MULTILINE)

            o_file = self.chapter_dir / filename
            out_paras = []
            for para in paras:
                if len(para) > 0:
                    sents = []
                    for sent in para.split("\n"):
                        sent = sent.strip()
                        sent = re.sub(pattern_remove_starting_with, "", sent)
                        # if len(sent) > 0:
                        #     seg_sent = seg.segment(sent)
                        #     for ss in seg_sent:
                        #         ss = re.sub(pattern_remove_starting_with, '', ss)
                        sents.append(sent)
                    out_paras.append("\n".join(sents))

            with o_file.open("w", encoding="utf-8") as fo:
                fo.write("\n\n".join(out_paras))
            doc = {"chapter": chapter, "text": "\n\n".join(out_paras)}
            documents.append(doc)
        df = pd.DataFrame(documents)
        df.to_csv(self.output_file, index=False)
