import os
import codecs
import pandas as pd
from pathlib import Path
from glob import glob
from ekorpkit.utils.func import ordinal


class ESGReport:
    def __init__(self, name, output_dir, output_file, input_path, txt_info, **kwargs):

        self.name = name
        self.input_path = input_path
        self.txt_info = txt_info
        self.output_path = f"{output_dir}/{output_file}"
        os.makedirs(output_dir, exist_ok=True)

        self.parse_text_files()

    def parse_text_files(self):
        filepaths = glob(self.input_path, recursive=True)
        filepaths = [fp for fp in filepaths if Path(fp).is_file()]
        txt_info = self.txt_info

        initial_file_num = txt_info.get("initial_file_num", 0)
        segment_separator = txt_info.get("segment_separator", None)
        segment_separator = codecs.decode(segment_separator, "unicode_escape")
        doc_id_format = txt_info.get("doc_id_format", None)
        file_num_increment = txt_info.get("file_num_increment", 1)

        file_num = initial_file_num
        reports = []
        for i, file in enumerate(filepaths):
            file = Path(file)
            print(" >> processing {} file: {}".format(ordinal(i + 1), file.name))

            texts = file.open().read()
            for seg_num, doc in enumerate(texts.split(segment_separator)):
                doc = doc.strip()
                if len(doc) > 0:
                    doc_id = doc_id_format.format(file_num=file_num, seg_num=seg_num)
                    rpt = {"doc_id": doc_id, "filename": file.stem, "text": doc}
                    reports.append(rpt)
            file_num += file_num_increment

        df = pd.DataFrame(reports)
        print(df.tail())
        df.to_csv(self.output_path, header=True, index=False)
        print(
            f"Corpus [{self.name}] is built to [{self.output_path}] from [{self.input_path}]"
        )
